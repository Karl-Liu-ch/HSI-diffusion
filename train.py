from utils import instantiate_from_config
from results.figures import gen_resutls
import torch
import os
from omegaconf import OmegaConf
import argparse
from torch.utils.data import DataLoader
from dataset.datasets import TestDataset
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='sncwgan_dtn')
# parser.add_argument('--datanames', type=list, default=['ARAD/'])
parser.add_argument("-c",'--config', type=str, default='configs/dtgan/dtn_sndisc.yaml')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--end_epoch", type=int, default=200, help="number of epochs")
parser.add_argument("--learning_rate", type=float, default=4e-4, help="initial learning rate")
# parser.add_argument("--ckpath", type=str, default='/work3/s212645/Spectral_Reconstruction/checkpoint/gan/msdtn/')
parser.add_argument("--data_root", type=str, default='/work3/s212645/Spectral_Reconstruction/')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=128, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
parser.add_argument("--local-rank", type=int)
parser.add_argument("--multigpu", action='store_true')
parser.add_argument("--notone", action='store_false')
parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?", help="resume from logdir or checkpoint in logdir",)
opt = parser.parse_args()

if __name__ == '__main__':
    cfg_path = opt.config
    opt_cfg = OmegaConf.create()
    opt_cfg.params = OmegaConf.create()
    opt_cfg.params.update(vars(opt))
    yaml_cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(opt_cfg, yaml_cfg)
    # cfg = OmegaConf.merge(yaml_cfg, opt_cfg)
    if opt.mode == 'tuning':
        cfg.params.data.params.train.params.crop_size = opt.patch_size
        cfg.params.data.params.train.params.stride = opt.stride
    print(OmegaConf.to_yaml(cfg))
    
    # cfg = OmegaConf.load(cfg_path)
    # cfg.params.update(vars(opt))
    
    model = instantiate_from_config(cfg)
    # modelname = str(cfg.params.genconfig.target).split('.')[-1]
    modelname = str(cfg.params.ckpath).split('/')[-2]
    print(modelname)
    modelnames = [modelname]
    if opt.resume:
        try:
            model.load_checkpoint()
        except Exception as ex:
            print(ex)
    run = True
    while run:
        match opt.mode:
            case 'train':
                # try:
                #     model.load_checkpoint()
                # except Exception as ex:
                #     print(ex)
                model.train()
                opt.mode = 'tuning'
            case 'tuning':
                model.finetuning()
                # opt.mode = 'test'
                opt.mode = 'valid'
            case 'test':
                model.load_checkpoint(best=True)
                model.test(modelname)
                opt.mode = 'testfull'
            case 'testfull':
                model.load_checkpoint(best=True)
                test_data_arad = TestDataset(data_root=opt.data_root, crop_size=1e8, valid_ratio = 0.1, test_ratio=0.1, datanames=['ARAD/'], cave=False)
                test_data_bgu = TestDataset(data_root=opt.data_root, crop_size=1e8, valid_ratio = 0.1, test_ratio=0.1, datanames=['BGU/'], cave=False)
                test_data_cave = TestDataset(data_root=opt.data_root, crop_size=1e8, valid_ratio = 0, test_ratio=1, datanames=['CAVE/'], cave=False)
                test_loader_arad = DataLoader(dataset=test_data_arad, batch_size=1, shuffle=False, num_workers=32, pin_memory=True)
                test_loader_bgu = DataLoader(dataset=test_data_bgu, batch_size=1, shuffle=False, num_workers=32, pin_memory=True)
                test_loader_cave = DataLoader(dataset=test_data_cave, batch_size=1, shuffle=False, num_workers=32, pin_memory=True)

                test_loaders = {
                    'ARAD': test_loader_arad, 
                    'BGU': test_loader_bgu,
                    'CAVE': test_loader_cave, 
                    }
                model.test_full_resol(modelname, test_loaders)
                # gen_resutls(modelnames, datanames = ['ARAD/'])
                gen_resutls(modelnames, datanames = ['ARAD/', 'BGU/', 'CAVE/'])
                opt.mode = 'stop'
            case 'valid':
                model.load_checkpoint(best=True)
                test_data_arad = instantiate_from_config(cfg.params.data.params.validation)
                test_loader_arad = DataLoader(dataset=test_data_arad, batch_size=1, shuffle=False, num_workers=32, pin_memory=True)
                test_loaders = {
                    'ARAD-orig': test_loader_arad, 
                    }
                model.test_full_resol(modelname, test_loaders)
                opt.mode = 'gen_results'
            case 'gen_results':
                gen_resutls(modelnames, datanames = ['ARAD-origin/'], valid_ratio=0.0, test_ratio=0.053, random_split=False)
                opt.mode = 'stop'
            case _:
                run = False
                print('finish running')