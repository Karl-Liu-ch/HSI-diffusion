from utils import instantiate_from_config
import torch
import os
from omegaconf import OmegaConf
import argparse
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='sncwgan_dtn')
parser.add_argument('--datanames', type=list, default=['ARAD/'])
parser.add_argument("-c",'--config', type=str, default='configs/dtn_sndisc.yaml')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--end_epoch", type=int, default=101, help="number of epochs")
parser.add_argument("--learning_rate", type=float, default=4e-4, help="initial learning rate")
# parser.add_argument("--ckpath", type=str, default='/work3/s212645/Spectral_Reconstruction/checkpoint/gan/msdtn/')
parser.add_argument("--data_root", type=str, default='/work3/s212645/Spectral_Reconstruction/')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0,1', help='path log files')
parser.add_argument("--local-rank", type=int)
parser.add_argument("--multigpu", action='store_true')
parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?", help="resume from logdir or checkpoint in logdir",)
opt = parser.parse_args()

if opt.multigpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


if __name__ == '__main__':
    cfg_path = opt.config
    cfg = OmegaConf.load(cfg_path)
    cfg.params.update(vars(opt))
    model = instantiate_from_config(cfg)
    modelname = str(cfg.params.genconfig.target).split('.')[-1]
    if opt.resume:
        try:
            model.load_checkpoint()
        except Exception as ex:
            print(ex)
    if opt.mode == 'train':
        # try:
        #     model.load_checkpoint()
        # except Exception as ex:
        #     print(ex)
        model.train()
    elif opt.mode == 'test':
        model.load_checkpoint(best=True)
        model.test(modelname)
    elif opt.mode == 'testfull':
        model.load_checkpoint(best=True)
        model.test_full_resol(modelname)