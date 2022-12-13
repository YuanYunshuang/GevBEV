import logging
import os, glob, json
import os.path as osp
import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from dataset.const import LABEL_COLORS_carla, VALID_CLS_nuscenes
from dataset.base_dataset import BaseDataset
from utils.box_utils import limit_period


class NuscenesDataset(BaseDataset):
    LABEL_COLORS = None
    VALID_CLS = VALID_CLS_nuscenes

    def __init__(self, cfgs, mode):
        super(NuscenesDataset, self).__init__(cfgs, mode)

    def init_dataset(self):
        # load annotation mapping info
        with open(osp.join(self.cfgs['path'], 'anno_mapping.json'), mode='r') as fh:
            anno_mapping = json.load(fh)

        self.LABEL_COLORS = anno_mapping['colormap']

        # load all data
        file = osp.join(osp.dirname(osp.abspath(__file__)), f'nuscenes/splits/{self.mode}.txt')
        self.data = []
        with open(file, "r") as f:
            scenes = f.read().splitlines()
        # load meta info
        with open(osp.join(self.cfgs['path'], f'v1.0-trainval.json'), mode='r') as fh:
            meta = json.load(fh)

        self.samples = []
        for s in scenes:
            self.samples.extend([osp.join(s, sample['sample_token'])for sample in meta[s]])

    def load_one_sample(self, item):
        sample = self.samples[item]
        pcdf = os.path.join(self.cfgs['path'], 'lidar', sample + '.bin')
        lidar = np.fromfile(pcdf, dtype="float32").reshape(-1, 5)
        lidar_idx = np.zeros_like(lidar[:, 0])

        boxf = os.path.join(self.cfgs['path'], 'anno', sample + '.txt')
        if os.stat(boxf).st_size == 0:
            boxes = np.zeros((0, 7))
        else:
            boxes = np.loadtxt(boxf, dtype=np.float).reshape(-1, 8)
            mask = np.logical_and(boxes[:, 7] >= 15, boxes[:, 7] <= 23)
            boxes = boxes[mask, :7]
            boxes[:, 6] = limit_period(boxes[:, 6], 0.5, 2 * np.pi)

        mapf = os.path.join(self.cfgs['path'], 'map', sample + '.png')
        bev_map = Image.open(mapf).__array__() #.transpose(1, 0, 2)[::-1]

        return {
            'frame_id': sample.split('/'),
            'lidar': lidar,
            'lidar_idx': lidar_idx,
            'boxes': boxes,
            'bevmap_static': bev_map[::2, ::2, 0],
            'bevmap_dynamic': bev_map[::2, ::2, 1],
        }


