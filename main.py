from utils import *
import os
import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from options import opt
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

os.environ['NCCL_P2P_DISABLE'] = str(1)
BASE_BS = 32

if __name__ == '__main__':
    if opt.val_check_interval:
        val_check_interval=opt.val_check_interval
        check_val_every_n_epoch=None
    else:
        val_check_interval=1.0
        check_val_every_n_epoch=opt.check_val
    print(val_check_interval, check_val_every_n_epoch)
    
    cfg_path = opt.config
    cfg = OmegaConf.load(cfg_path)
    try:
        logdir = cfg.logdir
    except:
        logdir = opt.logdir
    print(logdir)
    devices = []
    for gpu_id in opt.gpu_id.split(','):
        devices.append(int(gpu_id))
    ngpus = len(devices)
    if len(devices) > 1:
        strategy = 'ddp_find_unused_parameters_true'
        sync_batchnorm=False
    else:
        strategy = 'auto'
        sync_batchnorm=False

    cfg.data.params.batch_size = opt.batch_size
    print(opt.batch_size * ngpus / BASE_BS)
    cfg.model.params.learning_rate = cfg.model.params.learning_rate * opt.batch_size * ngpus / BASE_BS
    print(cfg.model.params.learning_rate)

    model = instantiate_from_config(cfg.model)
    if opt.resume:
        ckpt_path = opt.resume
        try:
            model.init_from_ckpt(ckpt_path)
            ckpt_path = None
        except Exception as ex:
            ckpt_path = None
            print(ex)
        # model = get_obj_from_str(cfg.model["target"]).load_from_checkpoint(ckpt_path, **cfg.model.get("params", dict()), map_location = 'cpu')
    else:
        ckpt_path = None

    tblogger = TensorBoardLogger(save_dir=logdir, version=0)
    try:
        epochs = cfg.model.params.epochs
    except:
        epochs = opt.end_epoch
    # if opt.load_pth:
    #     model.load_from_pth(opt.load_pth)
    #     model.load_pth = True
    match opt.mode:
        case 'train':
            early_stop_callback = EarlyStopping(monitor=cfg.model.params.monitor, min_delta=0.002, patience=20, verbose=False, mode="min")
            checkpoint_callback = ModelCheckpoint(
                monitor=cfg.model.params.monitor,
                save_top_k=1,
                mode='min',
                save_last=True,
                dirpath=logdir + 'lightning_logs/version_0/checkpoints/',
                filename='epoch{epoch:02d}-mrae_avg{val/mrae_avg:.2f}',
                auto_insert_metric_name=False
            )
            trainer = Trainer(accelerator="gpu",
                            devices=devices,
                            strategy=strategy,
                            max_epochs=epochs,
                            enable_checkpointing=True,
                            sync_batchnorm=sync_batchnorm,
                            default_root_dir=logdir, 
                            callbacks=[checkpoint_callback], 
                            # callbacks=[checkpoint_callback, early_stop_callback], 
                            logger=tblogger, 
                            val_check_interval = val_check_interval, 
                            check_val_every_n_epoch=check_val_every_n_epoch)
            data = instantiate_from_config(cfg.data)
            trainer.fit(model, data)
            opt.mode = 'test'
        case 'tuning':
            early_stop_callback = EarlyStopping(monitor=cfg.model.params.monitor, min_delta=0.002, patience=20, verbose=False, mode="min")
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
            model._temp_epoch = 0
            for i in range(2):
                trainer = Trainer(accelerator="gpu",
                                devices=devices,
                                strategy=strategy,
                                max_epochs=20,
                                enable_checkpointing=True,
                                sync_batchnorm=sync_batchnorm,
                                default_root_dir=logdir, 
                                callbacks=[checkpoint_callback], 
                                val_check_interval = val_check_interval, 
                                check_val_every_n_epoch=check_val_every_n_epoch,
                                logger=tblogger)
                cfg.data.params.train.params.crop_size *= 2
                cfg.data.params.batch_size = cfg.data.params.batch_size // 4
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
                model._temp_epoch = 0
                trainer.fit(model, data)
        case "test":
            from torch.utils.data import DataLoader
            early_stop_callback = EarlyStopping(monitor=cfg.model.params.monitor, min_delta=0.002, patience=20, verbose=False, mode="min")
            checkpoint_callback = ModelCheckpoint(
                monitor=cfg.model.params.monitor,
                save_top_k=1,
                mode='min',
                save_last=True,
                dirpath=logdir + 'lightning_logs/version_0/checkpoints/',
                filename='epoch{epoch:02d}-mrae_avg{val/mrae_avg:.2f}',
                auto_insert_metric_name=False
            )
            trainer = Trainer(accelerator="gpu",
                            devices=devices,
                            strategy=strategy,
                            max_epochs=epochs,
                            enable_checkpointing=True,
                            sync_batchnorm=sync_batchnorm,
                            default_root_dir=logdir, 
                            callbacks=[checkpoint_callback], 
                            # callbacks=[checkpoint_callback, early_stop_callback], 
                            logger=tblogger, 
                            val_check_interval = val_check_interval, 
                            check_val_every_n_epoch=check_val_every_n_epoch)
            data = instantiate_from_config(cfg.data.params.test)
            dataloader = DataLoader(dataset=data, batch_size=opt.batch_size, shuffle=False, num_workers=32, pin_memory=True)
            trainer.test(model, dataloaders=dataloader)