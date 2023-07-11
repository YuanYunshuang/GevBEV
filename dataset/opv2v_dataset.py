import logging
import os, glob, json
import torch
import logging

import cv2
import numpy as np
from torch.utils.data import Dataset
from dataset.const import LABEL_COLORS_carla, VALID_CLS_carla
from dataset.openv2v.opv2v_base import OpV2VBase
from dataset.openv2v.utils import boxes_to_corners_3d
from utils.misc import print_exec_time


class OpV2VDataset(OpV2VBase):
    LABEL_COLORS = LABEL_COLORS_carla
    VALID_CLS = VALID_CLS_carla

    def __init__(self, cfgs, mode, use_cuda):
        super(OpV2VDataset, self).__init__(cfgs, mode, use_cuda)
        logging.info(f"Loaded {len(self)} {mode} samples.")

    def load_one_sample(self, item):
        data_dict = self.load_data(item)['ego']

        # convert lidar list to one array and a corresponding idx array
        lidars = data_dict['projected_lidar']
        lidar_idx = np.concatenate([np.ones_like(lidar[:, 0]) * i
                                    for i, lidar in enumerate(lidars)],
                                   axis=0)
        lidars = np.concatenate(lidars, axis=0)
        # pad point label column with -1 if not available
        if lidars.shape[1] == 4:
            lidars = np.concatenate([lidars, - np.ones_like(lidars[:, :1])], axis=-1)

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
        if self.order == 'hwl':  # standard order for training is 'lwh'
            boxes = boxes[:, [0, 1, 2, 5, 4, 3, 6]]

        if data_dict['bev_map'] is not None:
            bev_map_static = data_dict['bev_map'][..., 2]
            bev_map_dynamic = data_dict['bev_map'][..., 1]
        else:
            bev_map_dynamic = self.get_dynamic_bev_map(boxes, 'lwh')
            bev_map_static = None
            # save bev_map
            # img = np.zeros(bev_map_dynamic.shape[:2] + (3,), dtype=np.int8)
            # img[:, :, 1] = bev_map_dynamic
            # filename = os.path.join(self.cfgs['path'], self.mode, data_dict['scenario'],
            #                         str(data_dict['object_ids'][0]), f"{data_dict['timestamp']}_bev_map.jpg")
            # cv2.imwrite(filename, img)

        return {
            'frame_id': (data_dict['scenario'], data_dict['timestamp']),
            'lidar': lidars,
            'lidar_idx': lidar_idx,
            'boxes': boxes,
            'bevmap_static': bev_map_static,
            'bevmap_dynamic': bev_map_dynamic,
            'tf_matrices': tf_matrices,
        }

    def get_dynamic_bev_map(self, bbxs, order='lwh'):
        bbxs = boxes_to_corners_3d(bbxs, order)[:, :4]
        lidar_range = self.cfgs['lidar_range']
        resolution = self.cfgs['bev_res']

        w = round((lidar_range[3] - lidar_range[0]) / resolution)
        h = round((lidar_range[4] - lidar_range[1]) / resolution)
        buf = np.zeros((h, w), dtype=np.uint8)
        bev_map = np.zeros((h, w), dtype=np.uint8)

        for box in bbxs:
            box[:, 0] = (box[:, 0] - lidar_range[0]) / resolution
            box[:, 1] = (box[:, 1] - lidar_range[1]) / resolution
            buf.fill(0)
            cv2.fillPoly(buf, [box[:, :2].round().astype(np.int32)], 1, cv2.INTER_LINEAR)
            bev_map[buf > 0] = 1

        return bev_map.T


if __name__ == '__main__':
    import argparse
    from config import load_yaml
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/minkunet_evigausbev_v2v4real.yaml")
    parser.add_argument("--cuda_loader", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args)
    cfg['DATASET']['visualize'] = False
    cfg['DATASET']['loc_err_flag'] = False
    cfg['DATASET']['loc_err_t_std'] = 0.5
    cfg['DATASET']['loc_err_r_std'] = 0.0
    dataset = OpV2VDataset(cfg['DATASET'], mode='test', use_cuda=True)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             sampler=None, num_workers=1,
                                             shuffle=False,
                                             collate_fn=dataset.collate_batch)
    i = 1
    for batch in dataset:
        print(i)
        i += 1
        # if i > 5:
        #     break
        img = batch['bevmap_dynamic'][::2, ::2] * 255
        cv2.imwrite(
            os.path.join("/mars/projects20/evibev_exp/v2vreal/gt_bev", '_'.join(batch['frame_id']) + '.jpg'),
            img.T
        )

    # dataset.add_free_space_points.get_mean_runtime()
    # dataset.sample_bev_pts.get_mean_runtime()
