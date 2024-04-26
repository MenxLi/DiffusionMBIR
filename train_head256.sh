#!/bin/bash

# python main.py \
#   --config=configs/ve/AAPM_256_ncsnpp_continuous.py \
#   --eval_folder=eval/AAPM256 \
#   --mode='train' \
#   --workdir=workdir/AAPM256

python main.py \
  --config=configs/ve/cone_head_256_ncsnpp_continuous.py \
  --eval_folder=eval/exp_head256 \
  --mode='train' \
  --workdir=workdir/exp_head256