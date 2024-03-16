from utils import *
import os
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from omegaconf import OmegaConf
from options import opt as args

cfg_path = args.config
cfg = OmegaConf.load(cfg_path)
model = instantiate_from_config(cfg.model)
data = instantiate_from_config(cfg.data)
trainer = Trainer(accelerator="gpu", enable_checkpointing=True, default_root_dir="/work3/s212645/Spectral_Reconstruction/checkpoint/sncwgan-dtn/")
trainer.fit(model, data)