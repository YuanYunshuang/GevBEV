import random

import numpy as np
import torch


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_optimizer(model, cfg):
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,  lr=cfg['lr'],
                                  weight_decay=cfg['weight_decay'],
                                  betas=tuple(cfg['betas']))

    # construct a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=cfg['milestones'],
                                                        gamma=cfg['gamma'])

    return optimizer, lr_scheduler


def load_tensors_to_gpu(batch_dict):
    """
    Load all tensors in batch_dict to gpu
    :param batch_dict: batched data dict
    :return:
    """
    for k, v in batch_dict.items():
        if isinstance(v, torch.Tensor):
            batch_dict[k] = v.cuda()


def load_model_dict(model, pretrained_dict):
    # 1. filter out unnecessary keys
    model_dict = model.state_dict()
    matched_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            matched_dict[k] = v
    # 2. overwrite entries in the existing state dict
    model_dict.update(matched_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model