import logging
import os, glob, json
import torch
import hydra
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from dataset.const import LABEL_COLORS_carla, VALID_CLS_carla
from dataset.openv2v.opv2v_base import OpV2VBase


class OpV2VDataset(OpV2VBase):
    LABEL_COLORS = LABEL_COLORS_carla
    VALID_CLS = VALID_CLS_carla

    def __init__(self, cfgs, mode):
        super(OpV2VDataset, self).__init__(cfgs, mode)

    def load_one_sample(self, item):
        data_dict = self.load_data(item)['ego']

        # convert lidar list to one array and a corresponding idx array
        lidars = data_dict['projected_lidar']
        lidar_idx = np.concatenate([np.ones_like(lidar[:, 0]) * i
                                    for i, lidar in enumerate(lidars)],
                                   axis=0)
        lidars = np.concatenate(lidars, axis=0)
        tf_matrices = np.stack(data_dict['tf_matrices'], axis=0)

        # import matplotlib.pyplot as plt
        # C = np.array(list(self.LABEL_COLORS.values())) / 255
        # colors = C[lidars[:, -1].astype(int)]
        # plt.scatter(lidars[:, 0], lidars[:, 1], c=colors, s=1)
        # plt.show()
        # plt.close()

        # get bounding boxes of ego vehicle
        bbx_mask = data_dict['object_bbx_mask'].astype(bool)
        boxes = data_dict['object_bbx_center'][bbx_mask]

        return {
            'frame_id': (data_dict['scenario'], data_dict['timestamp']),
            'lidar': lidars,
            'lidar_idx': lidar_idx,
            'boxes': boxes,
            'bevmap_static': data_dict['bev_map'][..., 2],
            'bevmap_dynamic': data_dict['bev_map'][..., 1],
            'tf_matrices': tf_matrices,
        }


@hydra.main(config_path=str(Path(__file__).parents[1] / 'config'),
            config_name='spconv_opv2v.yaml')
def main(cfg):
    dataset = OpV2VDataset(cfg['DATASET'], mode='train')
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=8,
                                             sampler=None, num_workers=4,
                                             shuffle=False,
                                             collate_fn=dataset.collate_batch)
    for batch in dataloader:
        print(batch.keys())


if __name__ == '__main__':
    main()
