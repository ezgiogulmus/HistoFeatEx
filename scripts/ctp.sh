#!/bin/bash
output_file="./scripts/errors_ctp.txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}
# You must activate the ctp environment before running this script
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --model_type ctp --ckpt_path ./model_ckpt/ctranspath.pth --data_root_dir /media/nfs/SURV/TCGA_OV/"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --model_type ctp --ckpt_path ./model_ckpt/ctranspath.pth --data_root_dir /media/nfs/SURV/BasOVER"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --model_type ctp --ckpt_path ./model_ckpt/ctranspath.pth --data_root_dir /media/nfs/SURV/TCGA_UCEC/"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --model_type ctp --ckpt_path ./model_ckpt/ctranspath.pth --data_root_dir /media/nfs/SURV/TCGA_BRCA/"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --model_type ctp --ckpt_path ./model_ckpt/ctranspath.pth --data_root_dir /media/nfs/SURV/TCGA_CESC/"

python ./scripts/check_errors.py "$output_file"