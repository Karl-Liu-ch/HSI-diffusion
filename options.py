import sys
sys.path.append('./')
import argparse
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='sncwgan_dtn')
parser.add_argument("-c",'--config', type=str, default='configs/sncwgan_dtn.yaml')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--G', type=str, default='res')
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--in_channels", type=int, default=6, help="inchannels")
parser.add_argument("--check_val", type=int, default=1, help="inchannels")
parser.add_argument("--end_epoch", type=int, default=101, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("-l", "--logdir", type=str, default='/work3/s212645/Spectral_Reconstruction/checkpoint/', help='path log files')
parser.add_argument("--data_root", type=str, default='/work3/s212645/Spectral_Reconstruction/')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
parser.add_argument("--local-rank", type=int, default=-1)
parser.add_argument("--multigpu", action='store_true')
parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?", help="resume from logdir or checkpoint in logdir",)
parser.add_argument("--load-pth", type=str, const=True, default="", nargs="?", help="load from pth",)
parser.add_argument("--nonoise", action='store_true')
parser.add_argument("--learning_rate", type=float, default=4e-4, help="initial learning rate")
parser.add_argument('--datanames', type=list, default=['ARAD/'])
parser.add_argument("--notone", action='store_false')
parser.add_argument("--val_check_interval", type=int, const=True, default=0, nargs="?", help="val_check_interval",)
# parser.add_argument("--ckpath", type=str, default='/work3/s212645/Spectral_Reconstruction/checkpoint/gan/msdtn/')
opt = parser.parse_args()
# opt = parser.parse_args(args=[])

if __name__ == '__main__':
    if opt.val_check_interval:
        val_check_interval=opt.val_check_interval
        check_val_every_n_epoch=None
    else:
        val_check_interval=1.0
        check_val_every_n_epoch=opt.check_val
    print(val_check_interval, check_val_every_n_epoch)

    from omegaconf import OmegaConf

    cfg_path = 'configs/ae_kl/hsi_vae_perceptual.yaml'
    cfg = OmegaConf.load(cfg_path)
    aeconfig = OmegaConf.create()
    aeconfig['target'] = 'models.vae.networks.DualTransformerEncoder'
    aeconfig['params'] = cfg.model.params.ddconfig.encoder.params
    from utils import *
    instantiate_from_config(aeconfig)