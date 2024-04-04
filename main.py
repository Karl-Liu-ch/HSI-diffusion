from utils import *
import os
import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from options import opt

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

if __name__ == '__main__':
    logdir = opt.logdir
    print(torch.cuda.mem_get_info())
    cfg_path = opt.config
    cfg = OmegaConf.load(cfg_path)
    if opt.resume:
        ckpt_path = opt.resume
        model = get_obj_from_str(cfg.model["target"]).load_from_checkpoint(ckpt_path, **cfg.model.get("params", dict()))
    else:
        ckpt_path = None
        model = instantiate_from_config(cfg.model)

    vae_path = logdir

    checkpoint_callback = ModelCheckpoint(
        monitor='val/mrae_avg',
        # monitor='val/loss_ema',
        save_top_k=3,
        mode='min',
        save_last=True,
        dirpath=logdir + 'lightning_logs/version_0/checkpoints/',
        filename='epoch{epoch:02d}-mrae_avg{val/mrae_avg:.2f}',
        auto_insert_metric_name=False
    )

    tblogger = TensorBoardLogger(save_dir=logdir, version=0)

    trainer = Trainer(accelerator="gpu",
                    # devices=[1],
                    enable_checkpointing=True,
                    default_root_dir=logdir, 
                    callbacks=[checkpoint_callback], 
                    logger=tblogger)
    cfg.data.params.batch_size = opt.batch_size
    data = instantiate_from_config(cfg.data)
    trainer.fit(model, data, ckpt_path=ckpt_path)