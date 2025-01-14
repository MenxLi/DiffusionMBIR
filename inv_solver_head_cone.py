import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import numpy as np
import controllable_generation_TV

from utils import restore_checkpoint, clear
from pathlib import Path
from models import utils as mutils
from models import ncsnpp   # this one is needed
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector)
import datasets
from physics.ct import CBCT
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

###############################################
# Configurations
###############################################
problem = 'cone_sparseview_CBCT_head_TV_total'
config_name = 'cone_head_256_ncsnpp_continuous'
sde = 'VESDE'
num_scales = 2000
ckpt_num = 50
N = num_scales
vol_name = '99cab51a7f78ec04ef5b0431f07a6737'
root = Path(f'./data/CBCT_head/{vol_name}')

# Parameters for the inverse problem
Nview = 40
size = 256
lamb = 0.04
rho = 10

if sde.lower() == 'vesde':
    from configs.ve import AAPM_256_ncsnpp_continuous as configs
    ckpt_filename = f"exp/ve/{config_name}/checkpoint_{ckpt_num}.pth"
    config = configs.get_config()
    config.model.num_scales = N
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sde.N = N
    sampling_eps = 1e-5
predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
probability_flow = False
snr = 0.16
n_steps = 1

batch_size = 12
config.training.batch_size = batch_size
config.eval.batch_size = batch_size
random_seed = 0

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)  ## model

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True, skip_optimizer=True)
ema.copy_to(score_model.parameters())

# Specify save directory for saving generated samples
# save_root = Path(f'./results/{config_name}/{problem}/m{Nview}/rho{rho}/lambda{lamb}')
save_root = Path(f'./results/{config_name}/{problem}/{vol_name}/m{Nview}/')
save_root.mkdir(parents=True, exist_ok=True)

irl_types = ['input', 'recon', 'label', 'BP', 'sinogram']
for t in irl_types:
    if t == 'recon':
        save_root_f = save_root / t / 'progress'
        save_root_f.mkdir(exist_ok=True, parents=True)
        save_root_f = save_root / t / 'slices'
        save_root_f.mkdir(exist_ok=True, parents=True)
    else:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)

# read all data
fname_list = os.listdir(root)
fname_list = sorted(fname_list, key=lambda x: float(x.split(".")[0]))
# print(fname_list)
all_img = []

print("Loading all data")
for fname in tqdm(fname_list):
    just_name = fname.split('.')[0]
    img = torch.from_numpy(np.load(os.path.join(root, fname), allow_pickle=True))
    h, w = img.shape
    img = img.view(1, 1, h, w)
    all_img.append(img)
    plt.imsave(os.path.join(save_root, 'label', f'{just_name}.png'), clear(img), cmap='gray')

all_img = torch.cat(all_img, dim=0)

radon = CBCT(geo = dict(
        camera_distance=1000,
        im_shape_HW=(512, 512),
        pixel_size=0.012,
        volume_shape=(all_img.shape[0], all_img.shape[-2], all_img.shape[-1]),
        voxel_size=1,
        focal_len=20,
    ), radon_view=Nview, device=config.device)

predicted_sinogram = []
label_sinogram = []
img_cache = None

img = all_img.to(config.device)
pc_radon = controllable_generation_TV.get_pc_radon_ADMM_TV_vol(sde,
                                                               predictor, corrector,
                                                               inverse_scaler,
                                                               snr=snr,
                                                               n_steps=n_steps,
                                                               probability_flow=probability_flow,
                                                               continuous=config.training.continuous,
                                                               denoise=True,
                                                               radon=radon,
                                                               save_progress=True,
                                                               save_root=save_root,
                                                               final_consistency=True,
                                                               img_shape=img.shape,
                                                               lamb_1=lamb,
                                                               rho=rho)
print("image shape", img.shape)
# Sparse by masking
sinogram = radon.A(img)

print("measured sinogram shape", sinogram.shape)

# A_dagger
bp = radon.AT(sinogram)

# save images
print("Saving images for BP, label, and sinogram")
for i in range(len(bp)):
    plt.imsave(save_root / 'BP' / f'{i}.png', clear(bp[i]), cmap='gray')
for i in range(len(sinogram.squeeze().permute(2,0,1))):
    plt.imsave(save_root / 'sinogram' / f'{i}.png', clear(sinogram.squeeze().permute(2,0,1)[i]), cmap='gray')

# Recon Image
x = pc_radon(score_model, scaler(img), measurement=sinogram)
img_cahce = x[-1].unsqueeze(0)

count = 0
for i, recon_img in enumerate(x):
    plt.imsave(save_root / 'BP' / f'{count}.png', clear(bp[i]), cmap='gray')
    plt.imsave(save_root / 'label' / f'{count}.png', clear(img[i]), cmap='gray')
    plt.imsave(save_root / 'recon' / 'slices' / f'{count}.png', clear(recon_img), cmap='gray')

    count += 1

# save reconstructed image as npy
np.save(str(save_root / 'recon' / 'all.npy'), x.detach().squeeze().cpu().numpy())
print("Reconstructed image saved, size: ", x.shape)

# Recon and Save Sinogram
label_sinogram.append(radon.A_all(img))
predicted_sinogram.append(radon.A_all(x))

original_sinogram = torch.cat(label_sinogram, dim=0).detach().cpu().numpy()
recon_sinogram = torch.cat(predicted_sinogram, dim=0).detach().cpu().numpy()

np.save(str(save_root / 'sinogram' / f'original_{count}.npy'), original_sinogram)
np.save(str(save_root / 'sinogram' / f'recon_{count}.npy'), recon_sinogram)