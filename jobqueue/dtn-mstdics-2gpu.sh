#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
#BSUB -R "select[gpu40gb]"
### -- set the job Name --
#BSUB -J dtn-mstdics-2gpu
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=2:mode=exclusive_process:mps=yes"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 40GB of system-memory
#BSUB -R "rusage[mem=20GB]"
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
#BSUB -o dtn-mstdics-2gpu%J.out
#BSUB -e dtn-mstdics-2gpu%J.err
# -- end of LSF options --
nvidia-smi
module load cuda/11.8
module load cudnn/v8.9.1.23-prod-cuda-11.X 
cd /zhome/02/b/164706/
source ./miniconda3/bin/activate
conda activate pytorch
cd /zhome/02/b/164706/Master_Courses/thesis/HSI-diffusion/

GPUID=0,1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
LOGDIR=/work3/s212645/Spectral_Reconstruction/checkpoint/SwinGAN/DTN_BN-SNTransformerDiscriminator/
CKPT=/work3/s212645/Spectral_Reconstruction/checkpoint/SwinGAN/DTN_BN-SNTransformerDiscriminator/lightning_logs/version_0/checkpoints/last-v1.ckpt
python -u finetune.py -c configs/sncwgan_dtn.yaml -l $LOGDIR --batch_size 16 --gpu_id $GPUID --in_channels 6 -r $CKPT --mode tuning512