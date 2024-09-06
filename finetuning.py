from utils import *
import os
import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from options import opt

os.environ['NCCL_P2P_DISABLE'] = str(1)

if __name__ == '__main__':
    cfg_path = opt.config
    cfg = OmegaConf.load(cfg_path)
    try:
        logdir = cfg.model.params.logdir
    except:
        logdir = opt.logdir
    devices = []
    for gpu_id in opt.gpu_id.split(','):
        devices.append(int(gpu_id))
    if len(devices) > 1:
        strategy = 'ddp_find_unused_parameters_true'
    else:
        strategy = 'auto'
    model = instantiate_from_config(cfg.model)
    if opt.resume:
        ckpt_path = opt.resume
        # model.init_from_ckpt(ckpt_path)
        # model = get_obj_from_str(cfg.model["target"]).load_from_checkpoint(ckpt_path, **cfg.model.get("params", dict()))
    else:
        ckpt_path = None

    tblogger = TensorBoardLogger(save_dir=logdir, version=0)
    try:
        epochs = cfg.model.params.epochs
    except:
        epochs = opt.end_epoch
    cfg.data.params.batch_size = opt.batch_size
    if opt.load_pth:
        model.load_from_pth(opt.load_pth)
        model.load_pth = True
    match opt.mode:
        case 'tuning256':
            checkpoint_callback = ModelCheckpoint(
                monitor=cfg.model.params.monitor,
                save_top_k=1,
                mode='min',
                save_last=True,
                dirpath=logdir + 'lightning_logs/version_0/checkpoints/finetune/',
                filename='epoch{epoch:02d}-mrae_avg{val/mrae_avg:.2f}',
                auto_insert_metric_name=False
            )
            model.init_from_ckpt(ckpt_path)
            trainer = Trainer(accelerator="gpu",
                            devices=devices,
                            strategy=strategy,
                            max_epochs=20,
                            enable_checkpointing=True,
                            default_root_dir=logdir, 
                            callbacks=[checkpoint_callback], 
                            logger=tblogger)
            cfg.data.params.train.params.crop_size = 256
            cfg.data.params.batch_size = cfg.data.params.batch_size // 4
            if cfg.data.params.batch_size < 1:
                cfg.data.params.batch_size = 1
                model.generator.apple(freeze_norm_stats)
            data = instantiate_from_config(cfg.data)
            model.finetune = True
            model.freezeD = True
            model.loss.finetune = True
            model.loss.threshold = 0
            model.lr = 4e-5
            model.end_epoch = 20
            trainer.fit(model, data)
        case 'tuning512':
            checkpoint_callback = ModelCheckpoint(
                monitor=cfg.model.params.monitor,
                save_top_k=1,
                mode='min',
                save_last=True,
                dirpath=logdir + 'lightning_logs/version_0/checkpoints/finetune/',
                filename='epoch{epoch:02d}-mrae_avg{val/mrae_avg:.2f}',
                auto_insert_metric_name=False
            )
            model.init_from_ckpt(ckpt_path)
            trainer = Trainer(accelerator="gpu",
                            devices=devices,
                            strategy=strategy,
                            max_epochs=20,
                            enable_checkpointing=True,
                            default_root_dir=logdir, 
                            callbacks=[checkpoint_callback], 
                            logger=tblogger)
            cfg.data.params.train.params.crop_size = 512
            cfg.data.params.batch_size = cfg.data.params.batch_size // 16
            if cfg.data.params.batch_size < 1:
                cfg.data.params.batch_size = 1
                model.generator.apply(freeze_norm_stats)
            data = instantiate_from_config(cfg.data)
            model.finetune = True
            model.freezeD = True
            model.loss.finetune = True
            model.loss.threshold = 0
            model.lr = 4e-5
            model.end_epoch = 20
            trainer.fit(model, data)