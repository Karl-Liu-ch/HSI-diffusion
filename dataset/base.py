import sys
sys.path.append('./')
import torch.nn as nn
import torch
import pytorch_lightning as pl
from utils import instantiate_from_config
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset
import collections
import torch
from taming.data.helper_types import Annotation
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from tqdm import tqdm
string_classes = str

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers)

# def custom_collate(batch):
#     r"""source: pytorch 1.9.0, only one modification to original code """

#     elem = batch[0]
#     elem_type = type(elem)
#     if isinstance(elem, torch.Tensor):
#         out = None
#         if torch.utils.data.get_worker_info() is not None:
#             # If we're in a background process, concatenate directly into a
#             # shared memory tensor to avoid an extra copy
#             numel = sum([x.numel() for x in batch])
#             storage = elem.storage()._new_shared(numel)
#             out = elem.new(storage)
#         return torch.stack(batch, 0, out=out)
#     elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
#             and elem_type.__name__ != 'string_':
#         if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
#             # array of string classes and object
#             if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
#                 raise TypeError(default_collate_err_msg_format.format(elem.dtype))

#             return custom_collate([torch.as_tensor(b) for b in batch])
#         elif elem.shape == ():  # scalars
#             return torch.as_tensor(batch)
#     elif isinstance(elem, float):
#         return torch.tensor(batch, dtype=torch.float64)
#     elif isinstance(elem, int):
#         return torch.tensor(batch)
#     elif isinstance(elem, string_classes):
#         return batch
#     elif isinstance(elem, collections.abc.Mapping):
#         return {key: custom_collate([d[key] for d in batch]) for key in elem}
#     elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
#         return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
#     if isinstance(elem, collections.abc.Sequence) and isinstance(elem[0], Annotation):  # added
#         return batch  # added
#     elif isinstance(elem, collections.abc.Sequence):
#         # check to make sure that the elements in batch have consistent size
#         it = iter(batch)
#         elem_size = len(next(it))
#         if not all(len(elem) == elem_size for elem in it):
#             raise RuntimeError('each element in list of batch should be of equal size')
#         transposed = zip(*batch)
#         return [custom_collate(samples) for samples in transposed]

#     raise TypeError(default_collate_err_msg_format.format(elem_type))
