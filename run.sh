nvidia-smi
module load cuda/11.8
module load cudnn/v8.9.1.23-prod-cuda-11.X 
cd /zhome/02/b/164706/
source ./miniconda3/bin/activate
conda activate pytorch
cd /zhome/02/b/164706/Master_Courses/thesis/HSI-diffusion/

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
GPUID=0,1

# # hsi_vae_perceptual
LOGDIR=/work3/s212645/Spectral_Reconstruction/checkpoint/vae_perceptual_32/
CKPT=/work3/s212645/Spectral_Reconstruction/checkpoint/vae_perceptual_32/lightning_logs/version_0/checkpoints/last.ckpt
python -u main.py -c configs/ae_kl/hsi_vae_perceptual_32.yaml -l $LOGDIR --end_epoch 500 --batch_size 32 --gpu_id $GPUID --in_channels 31 --mode train
#  -r $CKPT

# # hsi_vae_perceptual
# LOGDIR=/work3/s212645/Spectral_Reconstruction/checkpoint/vae_perceptual_skipconnect/
# CKPT=/work3/s212645/Spectral_Reconstruction/checkpoint/vae_perceptual_skipconnect/lightning_logs/version_0/checkpoints/last.ckpt
# python -u main.py -c configs/ae_kl/hsi_vae_perceptual_skipconnect.yaml -l $LOGDIR --end_epoch 500 --batch_size 16 --gpu_id $GPUID --in_channels 31 --mode train -r $CKPT

# # hsi_vae_perceptual_cond
# LOGDIR=/work3/s212645/Spectral_Reconstruction/checkpoint/vae_perceptual_cond/
# CKPT=/work3/s212645/Spectral_Reconstruction/checkpoint/vae_perceptual_cond/lightning_logs/version_0/checkpoints/last.ckpt
# python -u main.py -c configs/ae_kl/hsi_vae_perceptual_cond.yaml -l $LOGDIR --end_epoch 500 --batch_size 32 --gpu_id $GPUID --in_channels 6 --mode train -r $CKPT

# # hsi_ldm
# LOGDIR=/work3/s212645/Spectral_Reconstruction/checkpoint/ldm/ldm/
# CKPT=/work3/s212645/Spectral_Reconstruction/checkpoint/ldm/ldm/lightning_logs/version_0/checkpoints/last.ckpt
# python main.py -c configs/ldm/hsi_ldm.yaml --batch_size 64 --end_epoch 500 --gpu_id $GPUID -l $LOGDIR
# #  -r $CKPT
# # #  --check_val 5

# hsi_bbdm
LOGDIR=/work3/s212645/Spectral_Reconstruction/checkpoint/bbdm/
CKPT=/work3/s212645/Spectral_Reconstruction/checkpoint/bbdm/lightning_logs/version_0/checkpoints/last.ckpt
python main.py -c configs/bbdm/hsi_bbdm.yaml --batch_size 64 --gpu_id $GPUID -l $LOGDIR
#  -r $CKPT

# # hsi_ldm
# LOGDIR=/work3/s212645/Spectral_Reconstruction/checkpoint/ldm/ldm_skipconnect/
# CKPT=/work3/s212645/Spectral_Reconstruction/checkpoint/ldm/ldm_skipconnect/lightning_logs/version_0/checkpoints/last.ckpt
# python main.py -c configs/ldm/hsi_ldm_skipconnect.yaml --batch_size 16 --gpu_id $GPUID -l $LOGDIR
# #  -r $CKPT
# #  --check_val 5

# # sncwgan dtn
# LOGDIR=/work3/s212645/Spectral_Reconstruction/checkpoint/SwinGAN/DTN_BN-SNTransformerDiscriminator/
# CKPT=/work3/s212645/Spectral_Reconstruction/checkpoint/SwinGAN/DTN_BN-SNTransformerDiscriminator/lightning_logs/version_0/checkpoints/last.ckpt
# python -u main.py -c configs/sncwgan_dtn.yaml -l $LOGDIR --batch_size 12 --gpu_id $GPUID --in_channels 6 --mode train -r $CKPT
# # CKPT=/work3/s212645/Spectral_Reconstruction/checkpoint/SwinGAN/DTN_BN-SNTransformerDiscriminator/lightning_logs/version_0/checkpoints/last-v1.ckpt
# # python -u finetuning.py -c configs/sncwgan_dtn.yaml -l $LOGDIR --batch_size 12 --gpu_id $GPUID --in_channels 6 -r $CKPT --mode tuning512

# # sncwgan dtn no ycrcb
# LOGDIR=/work3/s212645/Spectral_Reconstruction/checkpoint/SwinGAN/DTN_no_ycrcb-SNTransformerDiscriminator/
# CKPT=/work3/s212645/Spectral_Reconstruction/checkpoint/SwinGAN/DTN_no_ycrcb-SNTransformerDiscriminator/lightning_logs/version_0/checkpoints/last.ckpt
# python -u main.py -c configs/sncwgan_dtn_no_ycrcb.yaml -l $LOGDIR --batch_size 8 --gpu_id 1,0 -r $CKPT --mode train

# # # sncwgan dtn no sam
# LOGDIR=/work3/s212645/Spectral_Reconstruction/checkpoint/SwinGAN/DTN_no_sam-SNTransformerDiscriminator/
# CKPT=/work3/s212645/Spectral_Reconstruction/checkpoint/SwinGAN/DTN_no_sam-SNTransformerDiscriminator/lightning_logs/version_0/checkpoints/last.ckpt
# PTH=/work3/s212645/Spectral_Reconstruction/checkpoint/gan/DTN-SNTransformerDiscriminator_no_sam/net.pth
# python -u main.py -c configs/sncwgan_dtn_no_sam.yaml -l $LOGDIR --batch_size 8 --gpu_id 0 --mode train --load-pth $PTH
# #  -r $CKPT


# python -u train.py -c configs/dtgan/dtn_snmstdisc_no_sam.yaml --gpu_id 0 --batch_size 14 --learning_rate 2e-4 -r --mode train
exit