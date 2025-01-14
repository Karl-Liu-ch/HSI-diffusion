#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"
### -- set the job Name --
#BSUB -J sstransformer_ori_ablation
### -- ask for number of cores (default: 1) --
#BSUB -n 16
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=2:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
###request 40GB of system-memory
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s212645@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -B
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o sstransformer_ori_ablation%J.out
#BSUB -e sstransformer_ori_ablation%J.err
# -- end of LSF options --
nvidia-smi
module load cuda/11.8
module load cudnn/v8.9.1.23-prod-cuda-11.X 
cd /zhome/02/b/164706/
source ./miniconda3/bin/activate
conda activate pytorch
cd /zhome/02/b/164706/Master_Courses/thesis/HSI-diffusion/
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
NCCL_P2P_DISABLE=1 python -u main.py -c configs/sst/sstransformer_lightning_ori_no_spectral.yaml --gpu_id 0,1 --batch_size 16 --learning_rate 4e-4 --mode train --val_check_interval 1000 --stride 8 --end_epoch 7 --patch_size 128 -r /work3/s212645/Spectral_Reconstruction/checkpoint/SSTransformer_orig_no_spectral/lightning_logs/version_0/checkpoints/last.ckpt
NCCL_P2P_DISABLE=1 python -u main.py -c configs/sst/sstransformer_lightning_ori_no_spectral.yaml --gpu_id 0,1 --batch_size 4 --learning_rate 4e-5 --mode finetune --stride 128 --end_epoch 100 --patch_size 256 -r /work3/s212645/Spectral_Reconstruction/checkpoint/SSTransformer_orig_no_spectral/lightning_logs/version_0/checkpoints/last.ckpt

# export CUDA_VISIBLE_DEVICES=2,3
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
NCCL_P2P_DISABLE=1 python -u main.py -c configs/sst/sstransformer_lightning_ori_no_rpe.yaml --gpu_id 0,1 --batch_size 16 --learning_rate 4e-4 --mode train --val_check_interval 2000 --stride 8 --end_epoch 7 --patch_size 128 -r /work3/s212645/Spectral_Reconstruction/checkpoint/SSTransformer_orig_no_rpe/lightning_logs/version_0/checkpoints/last.ckpt
NCCL_P2P_DISABLE=1 python -u main.py -c configs/sst/sstransformer_lightning_ori_no_rpe.yaml --gpu_id 0,1 --batch_size 4 --learning_rate 4e-5 --mode finetune --stride 128 --end_epoch 100 --patch_size 256 -r /work3/s212645/Spectral_Reconstruction/checkpoint/SSTransformer_orig_no_rpe/lightning_logs/version_0/checkpoints/last.ckpt