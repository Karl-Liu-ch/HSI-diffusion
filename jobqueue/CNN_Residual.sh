#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
### -- set the job Name --
#BSUB -J  CNN_Residual
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
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
#BSUB -o  CNN_Residual%J.out
#BSUB -e  CNN_Residual%J.err
# -- end of LSF options --
nvidia-smi
module load cuda/11.8
module load cudnn/v8.9.1.23-prod-cuda-11.X 
cd /zhome/02/b/164706/
source ./miniconda3/bin/activate
conda activate pytorch
cd /zhome/02/b/164706/Master_Courses/thesis/HSI-diffusion/
export PYTHONUNBUFFERED=1
# python -u train.py -c configs/cnn_residual/cnn_residual.yaml --gpu_id 0 --batch_size 128 --mode train --learning_rate 1e-4 -r
# python -u train.py -c configs/cnn_residual/cnn_residual_256.yaml --gpu_id 0 --batch_size 128 --mode tuning --learning_rate 1e-5 -r
# python -u train.py -c configs/cnn_residual/cnn_residual_512.yaml --gpu_id 0 --batch_size 128 --mode tuning --learning_rate 1e-5 -r

# python -u train.py -c configs/cnn_residual/cnn_residual_multistage.yaml --gpu_id 0 --batch_size 64 --mode train --learning_rate 2e-4 -r
# python -u train.py -c configs/cnn_residual/cnn_residual_multistage_256.yaml --gpu_id 0 --batch_size 64 --mode tuning --learning_rate 4e-5 -r
# python -u train.py -c configs/cnn_residual/cnn_residual_multistage_512.yaml --gpu_id 0 --batch_size 64 --mode tuning --learning_rate 4e-5 -r

# python -u train.py -c configs/cnn_residual/transformer.yaml --gpu_id 0 --batch_size 64 --mode train --learning_rate 2e-4 -r

python -u train.py -c configs/cnn_residual/Transformer.yaml --gpu_id 0 --batch_size 64 --mode train --learning_rate 4e-4 -r
# python -u train.py -c configs/cnn_residual/Transformer_256.yaml --gpu_id 0 --batch_size 64 --mode tuning --learning_rate 4e-5 -r
