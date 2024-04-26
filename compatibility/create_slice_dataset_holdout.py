import os
from tqdm import tqdm
import numpy as np
import platform

if platform.node() == "monsoon-extreme":
    source_dir = "/mnt/storage-ssd/Exp/cbctrec/datasets/head"
elif platform.node() in ["7eda8815a8eb", "7fab99823a5b"]:
    source_dir = "/remote-home/limengxun/Data/cbctrec-npz/head"
else:
    raise ValueError("Unknown platform: {}".format(platform.node()))

source_eval_dir = os.path.join(source_dir, "holdout")
if not os.path.exists(source_eval_dir):
    raise ValueError("Source directory does not exist")

dest_eval_dir = os.path.join("./data", "CBCT_head")

for _p in [dest_eval_dir]:
    if not os.path.exists(_p):
        os.makedirs(_p)

def formatNum(num):
    # format as 4 digit number, with leading zeros
    return f"{num:04d}"

file_list = list(os.listdir(source_eval_dir))
for file in tqdm(file_list):
    if not file.endswith(".npz"):
        print("Skipping file", file)
        continue
    file_path = os.path.join(source_eval_dir, file)
    _file_name_base = ".".join(os.path.basename(file).split(".")[:-1])
    data = np.load(file_path)["volume"].transpose(2, 0, 1)
    this_dest_dir = os.path.join(dest_eval_dir, _file_name_base)
    if not os.path.exists(this_dest_dir):
        os.makedirs(this_dest_dir)
    for i in range(data.shape[0]):
        # save as npy
        slice_data = data[i]
        slice_file = os.path.join(this_dest_dir, f"{formatNum(i)}.npy")
        np.save(slice_file, slice_data)