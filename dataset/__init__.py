import logging
import torch
import importlib


def get_dataloader(cfgs, mode='train'):
    name = cfgs['name']
    module = importlib.import_module(f'dataset.{name.lower()}_dataset')
    assert hasattr(module, f'{name}Dataset'), "Invalid dataset."
    module_class = getattr(module, f'{name}Dataset')
    dataset = module_class(cfgs, mode)
    shuffle = False if mode=='test' else cfgs.get('shuffle', None)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfgs[f'batch_size_{mode}'],
                                             sampler=None, num_workers=cfgs['n_workers'],
                                             shuffle=shuffle,
                                             collate_fn=dataset.collate_batch)
    return dataloader


