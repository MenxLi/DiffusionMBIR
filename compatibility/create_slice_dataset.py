"""
Make slice dataset for SDM out of our dataset
"""
import os, random
import numpy as np
from tqdm import tqdm

source_dir = "/mnt/storage-ssd/Exp/cbctrec/datasets/head"
source_train_dir = os.path.join(source_dir, "train")
if not os.path.exists(source_train_dir):
    raise ValueError("Source directory does not exist")

dest_dir = "/mnt/storage-ssd/Exp/cbctrec/datasets/head_slice"
dest_train_dir = os.path.join(dest_dir, "train")
dest_test_dir = os.path.join(dest_dir, "test")

for _p in [dest_train_dir, dest_test_dir]:
    if not os.path.exists(_p):
        os.makedirs(_p)

def formatNum(num):
    return f"{num:04d}"

file_list = list(os.listdir(source_train_dir))
random.shuffle(file_list)
for file in tqdm(file_list[:-10]):
    if not file.endswith(".npz"):
        print("Skipping file", file)
        continue
    file_path = os.path.join(source_train_dir, file)
    _file_name_base = ".".join(os.path.basename(file).split(".")[:-1])
    data = np.load(file_path)["volume"].transpose(2, 0, 1)
    for i in range(data.shape[0]):
        # save as npy
        slice_data = data[i]
        slice_file = os.path.join(dest_train_dir, f"{_file_name_base}_{formatNum(i)}.npy")
        np.save(slice_file, slice_data)

for file in tqdm(file_list[-10:]):
    if not file.endswith(".npz"):
        print("Skipping file", file)
        continue
    file_path = os.path.join(source_train_dir, file)
    _file_name_base = ".".join(os.path.basename(file).split(".")[:-1])
    data = np.load(file_path)["volume"].transpose(2, 0, 1)
    for i in range(data.shape[0]):
        # save as npy
        slice_data = data[i]
        slice_file = os.path.join(dest_test_dir, f"{_file_name_base}_{formatNum(i)}.npy")
        np.save(slice_file, slice_data)
