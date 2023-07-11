import logging, tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
import torch.nn.functional as F
from scipy.spatial.transform.rotation import Rotation as R

import dataset.processors as PP
from torch.utils.data import Dataset
from dataset.data_utils import project_points_by_matrix
from utils.pclib import rotate_points_along_z
from utils.box_utils import limit_period
from utils.vislib import draw_points_boxes_plt, draw_img
from utils.misc import print_exec_time
from ops.utils import points_in_boxes_cpu


class BaseDataset(Dataset):
    LABEL_COLORS = {}
    VALID_CLS = []

    def __init__(self, cfgs, mode, use_cuda=False):
        self.cfgs = cfgs
        self.mode = mode
        self.device = torch.device('cuda') if use_cuda else torch.device('cpu')
        self.visualize = cfgs['visualize']

        self.samples = []
        self.data = []

        self.init_dataset()

        # load all data
        if self.cfgs.get('load_all', False) and mode=='train':
            logging.info('Loading data ...')
            for s in tqdm.tqdm(self.samples):
                self.data.append(self.load_one_sample(s))

        pre_processes = []
        if mode in cfgs['preprocessors']:
            for name in cfgs['preprocessors'][mode]:
                pre_processes.append(getattr(PP, name)(**cfgs['preprocessors']['args'][name]))
            self.pre_processes = PP.Compose(pre_processes)
        post_processes = []
        if mode in cfgs['postprocessors']:
            for name in cfgs['postprocessors'][mode]:
                post_processes.append(getattr(PP, name)(**cfgs['postprocessors']['args'][name]))
            self.post_processes = PP.Compose(post_processes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        data = self.prepare_data(item)
        if getattr(self, 'visualize', False):
            self.visualize_data(data)
        if self.pre_processes is not None:
            self.pre_processes(data)
        return data

    def init_dataset(self):
        """Load all necessary meta information"""
        raise NotImplementedError

    def load_one_sample(self, item):
        """
        Load the data dictionary of one sample.
        :param item: sample index
        :return: dict
            - lidar: np.ndarray [N, 5], columns are (x, y, z, intensity, point_cls)
            - lidar_idx: np.ndarray [N,], indices for lidar data from different cavs,
              in the case of mono view, this can be None or an array of zeros
            - boxes: np.ndarray [M, 7+c], first 7 columns must be
              (x, y, z, dx, dy, dz, det_r) of the bounding boxes
            - bevmap_static: boolean np.ndarray [H, W]
            - bevmap_dynamic: boolean np.ndarray [H, W]
        """
        raise NotImplementedError

    def prepare_data(self, item):
        if self.cfgs.get('load_all', False) and self.mode == 'train':
            data_dict = self.data[item]
        else:
            data_dict = self.load_one_sample(item)
        data_dict = self.add_free_space_points(data_dict)
        data_dict = self.map_cls(data_dict)
        data_dict = self.sample_bev_pts(data_dict)
        if self.mode == 'train' and self.cfgs['augmentation']:
            data_dict = self.augmentation(data_dict)
        if self.cfgs.get('loc_err_flag', False):
            data_dict = self.add_loc_err(data_dict)
        data_dict = self.crop_points_range(data_dict)
        data_dict = self.remove_empty_boxes(data_dict)
        data_dict = self.positive_z(data_dict)
        coords, features = self.compose_coords_features(self.cfgs['voxel']['coords'],
                                                        self.cfgs['voxel']['features'],
                                                        data_dict['lidar'])

        labels = data_dict['lidar'][:, -1]

        mask = self.crop_points_by_features(features)
        lidar_idx = data_dict.get('lidar_idx', None)
        if lidar_idx is not None:
            lidar_idx = lidar_idx[mask]

        return {
            'frame_id': data_dict['frame_id'],
            'xyz': data_dict['lidar'][mask, :3], # for box regression tgt generation
            'coords': coords[mask],
            'coords_idx': lidar_idx,
            'features': features[mask],
            'target_semantic': labels[mask],
            'target_boxes': data_dict['boxes'],
            'target_bev_pts': data_dict.get('pts', None),
            'target_bev_pts_idx': data_dict.get('bev_pts_idx', None),
            'bevmap_static': data_dict.get('bevmap_static', None),
            'bevmap_dynamic': data_dict.get('bevmap_dynamic', None),
        }

    def remove_empty_boxes(self, data_dict):
        point_indices = points_in_boxes_cpu(
            data_dict['lidar'][:, :3],
            data_dict['boxes']
        )
        n_pts = point_indices.sum(axis=1)
        data_dict['boxes'] = data_dict['boxes'][n_pts > 3]
        return data_dict

    # @print_exec_time
    def sample_bev_pts(self, data_dict):
        """
        Sample BEV points based on bev-opv2v map and lidar points.
        This function can runs on GPU to speed up runtime.

        :param data_dict: dict
        :return: data_dict: updated with sampled bev_points -
                np.ndarray [K, 3], columns are (x, y, bev_cls)
        """
        bevmap_static = data_dict['bevmap_static']
        bevmap_dynamic = data_dict['bevmap_dynamic']
        device = self.device
        if isinstance(bevmap_dynamic, np.ndarray):
            bevmap_dynamic = torch.from_numpy(bevmap_dynamic).to(device)
            if bevmap_static is not None:
                bevmap_static = torch.from_numpy(bevmap_static).to(device)

        x_lim = self.cfgs['lidar_range'][3]
        y_lim = self.cfgs['lidar_range'][4]
        lidar = data_dict['lidar']
        lidar_idx = data_dict['lidar_idx']
        res = 0.4

        # generate random point samples
        points2d = np.concatenate([lidar_idx[:, None], lidar[:, :2]], axis=1)
        points2d = torch.from_numpy(points2d).to(device)

        offsets = torch.randn((len(points2d), 10, 3), device=device) * 3
        offsets[..., 0] = 0  # set idx column to 0
        points2d = points2d.reshape(-1, 1, 3) + offsets
        res_vec = torch.tensor([[[1, res, res]]], device=device)
        points2d = torch.unique(torch.div(points2d, res_vec, rounding_mode='floor'
                                          ).reshape(-1, 3), dim=0) * res_vec.reshape(1, 3)
        if not self.cfgs.get("discrete_bev_pts", False):
            points2d[:, 1:] = points2d[:, 1:] + torch.randn((len(points2d), 2), device=device)

        # retrieve labels from bev-opv2v maps and downsample is necessary
        h, w = bevmap_dynamic.shape
        pixels_per_meter = 1 / self.cfgs['bev_res']
        xs = torch.clip(torch.floor((points2d[:, 1] + x_lim) * pixels_per_meter).long(), 0, h - 1)
        ys = torch.clip(torch.floor((points2d[:, 2] + y_lim) * pixels_per_meter).long(), 0, w - 1)
        labels = torch.zeros_like(points2d[:, :2])
        sample_idx = []
        is_dynamic = bevmap_dynamic[xs, ys] > 0
        labels[is_dynamic, 1] = 1  # 2nd col. --> dynamic label
        sample_idx.append(torch.where(is_dynamic)[0])
        is_pos = is_dynamic
        if bevmap_static is not None:
            is_static = bevmap_static[xs, ys] > 0
            labels[is_static, 0] = 1   # 1st col. --> static label
            is_pos = torch.logical_or(is_dynamic, is_static)
            static_idx = torch.where(is_static)[0]
            if len(static_idx) > 3000:
                # only sample <=3000 static
                static_idx = static_idx[torch.randperm(len(static_idx))[:3000]]
            sample_idx.append(static_idx)
        neg_idx = torch.where(torch.logical_not(is_pos))[0]
        sample_idx.append(neg_idx[torch.randperm(len(neg_idx))[:3000]])  # only sample <=3000 negative

        sample_idx = torch.cat(sample_idx, dim=0)
        # bev-opv2v pts col. attr. (idx, x, y, static_lbl, dynamic_lbl)
        bev_points = torch.cat([points2d[sample_idx], labels[sample_idx]], dim=1).cpu().numpy()

        data_dict.update({
            'pts': bev_points[:, 1:],
            'bev_pts_idx': bev_points[:, 0],
        })
        return data_dict

    def positive_z(self, data_dict):
        """
        Shift all lidar points and bbx center along z to ensure thet z coordinates
        are larger equal zero. This ensures that ME can compress z-axis into one
        voxel via convolutions.
        :param data_dict: minimum input key-value pairs are 'lidar', 'boxes'
        :return: same dict as input with shifted z-coordinates for 'lidar' and 'boxes'
        """
        data_dict['lidar'][:, 2] -= self.cfgs['crop']['z'][0]
        data_dict['boxes'][:, 2] -= self.cfgs['crop']['z'][0]
        return data_dict

    def augmentation(self, data_dict):
        """
        Augment dataset by random rotation, scaling and noising
        :param data_dict: a dict contains:
                         - 'lidar': np.ndarray [N1, 3+c], columns 1-3 are x, y, z
                         - 'pts': np.ndarray [N2, 2], columns 1-2 are x, y
                         - 'boxes': np.ndarray [N3, 7+c], columns 1-7 are x, y, z, dx, dy, dz, det_r
        :return: the same dict with augmented data
        """
        lidars = data_dict['lidar']
        bev_pts = data_dict['pts']
        boxes = data_dict['boxes']
        rot = 2 * np.pi * np.random.random()

        lidars = rotate_points_along_z(lidars, rot)
        bev_pts = rotate_points_along_z(bev_pts, rot)
        boxes[:, :3] = rotate_points_along_z(boxes[:, :3], rot)
        boxes[:, 6] = limit_period(boxes[:, 6] + rot)

        flip = np.random.choice(4, 1)
        if flip == 1:
            lidars[:, 0] *= -1
            bev_pts[:, 0] *= -1
            boxes[:, 0] *= -1
            boxes[:, 6] = np.pi - boxes[:, 6]
        elif flip == 2:
            lidars[:, 1] *= -1
            bev_pts[:, 1] *= -1
            boxes[:, 1] *= -1
            boxes[:, 6] = - boxes[:, 6]
        elif flip == 3:
            lidars[:, :2] *= -1
            bev_pts[:, :2] *= -1
            boxes[:, :2] *= -1
            boxes[:, 6] = np.pi + boxes[:, 6]

        scaling = np.random.uniform(0.95, 1.05, (1, 3))
        noise = np.random.normal(0, 0.2, (1, 3))
        lidars[:, :3] = lidars[:, :3] * scaling + noise
        bev_pts[:, :2] = bev_pts[:, :2] * scaling[:, :2] + noise[:, :2]
        boxes[:, :3] = boxes[:, :3] * scaling + noise
        boxes[:, 3:6] = boxes[:, 3:6] * scaling

        data_dict.update({
            'lidar': lidars,
            'pts': bev_pts,
            'boxes': boxes
        })

        return data_dict

    def add_loc_err(self, data_dict):
        """Add loc noise to cav"""
        tf_matrices = data_dict['tf_matrices']

        lidar = data_dict['lidar']
        lidar_idx = data_dict['lidar_idx']

        for i in range(len(tf_matrices[1:])):
            loc_noise = np.random.normal(0, self.cfgs['loc_err_t_std'], 3)
            rot_noise = np.random.normal(0, self.cfgs['loc_err_r_std'], 3)
            rot_noise[:2] = 0
            rot_noise = np.deg2rad(rot_noise)
            # loc_noise[2] = 0
            noise_pose = np.eye(4)
            rot_mat = R.from_euler('xyz', rot_noise, degrees=True).as_matrix()
            noise_pose[:3, :3] = rot_mat @ tf_matrices[i + 1, :3, :3]
            noise_pose[:3, 3] = tf_matrices[i + 1, :3, 3] + loc_noise
            tf_matrices[i + 1] = noise_pose
            mask = lidar_idx == i + 1
            cur_lidar = lidar[mask]
            cur_lidar = rotate_points_along_z(cur_lidar, rot_noise[2])
            cur_lidar[:, :3] = cur_lidar[:, :3] + loc_noise.reshape(1, 3)
            lidar[mask] = cur_lidar
        data_dict['lidar'] = lidar
        return data_dict

    def crop_points_by_features(self, features):
        mask = np.ones_like(features[:, 1])
        for k, v in self.cfgs['crop'].items():
            if k == 'd' or k == 'z':
                continue
            symbols = [s.strip() for s in self.cfgs['voxel']['features'].split(',')]
            assert k in symbols
            coor_idx = ''.join(symbols).find(k)
            mask = np.logical_and(
                features[:, coor_idx] > v[0],
                features[:, coor_idx] < v[1]
            ) * mask
        return mask.astype(bool)

    def crop_points_range(self, data_dict):
        lidar = data_dict['lidar']
        lidar_idx = data_dict['lidar_idx']
        for s in self.cfgs['crop'].keys():
            mask = self.get_crop_mask(lidar, s)
            lidar = lidar[mask]
            lidar_idx = lidar_idx[mask]
        keep = np.ones_like(lidar_idx)
        for i in np.unique(lidar_idx):
            mask = lidar_idx == i
            if mask.sum() < 10:
                keep[mask] = 0
        keep = keep.astype(bool)
        lidar = lidar[keep]
        lidar_idx = lidar_idx[keep]
        data_dict['lidar'] = lidar
        data_dict['lidar_idx'] = lidar_idx
        if self.mode == 'train':
            for s in self.cfgs['crop'].keys():
                if s == 'z':
                    continue
                mask = self.get_crop_mask(data_dict['pts'], s)
                data_dict['pts'] = data_dict['pts'][mask]
                data_dict['bev_pts_idx'] = data_dict['bev_pts_idx'][mask]
            # after augmentation, in some cavs lidar points are all removed by cropping,
            # make sure the bev-opv2v pts for these cavs are also removed
            rm_idx = set(np.unique(data_dict['bev_pts_idx'])) - set(np.unique(lidar_idx))
            for i in rm_idx:
                mask = data_dict['bev_pts_idx'] == i
                data_dict['pts'] = data_dict['pts'][mask]
                data_dict['bev_pts_idx'] = data_dict['bev_pts_idx'][mask]
        return data_dict

    def crop_boxes_range(self, data_dict):
        boxes = data_dict['boxes']
        for s in self.cfgs['crop'].keys():
            mask = self.get_crop_mask(boxes[:, :3], s)
            if mask.sum() == 0:
                data_dict['boxes'] = np.zeros((0, 7))
                return data_dict
            else:
                boxes = boxes[mask].reshape(-1, 7)
        data_dict['boxes'] = boxes
        return data_dict

    def get_crop_mask(self, points, symbol):
        """
        Get crop mask for points
        :param points: np.ndarray [N, 3+c], column 1-3 must be x, y, z
        :param symbol: one of
        - 'x'(coordinate),
        - 'y'(coordinate),
        - 'z'(coordinate),
        - 't'(theta in degree),
        - 'c'(cos(t)),
        - 's'(sin(t)).
        :return: mask
        """
        points = getattr(self, f'get_feature_{symbol}')(points).squeeze()
        mask = np.logical_and(
            points > self.cfgs['crop'][symbol][0],
            points < self.cfgs['crop'][symbol][1]
        )
        return mask

    def map_cls(self, data_dict):
        """
        Mapping class numbers from a larger set to a smaller one
        :param lidar: np.ndarray [N, c+1], last column ist the original class label
        :return: np.ndarray [N, c+1], class labels in the smaller class set
        """
        lidar = data_dict['lidar']
        if (lidar[:, -1] == -1).all():
            return data_dict
        assert len(self.VALID_CLS) > 0, 'VALID_CLS not set!'
        cls = np.zeros_like(lidar[:, -1])
        for tgt_cls, src_cls_list in enumerate(self.VALID_CLS):
            for src_cls in src_cls_list:
                cls[lidar[:, -1] == src_cls] = tgt_cls + 1
        cls[lidar[:, -1] == -1] = -1 # free space points
        lidar[:, -1] = cls
        data_dict['lidar'] = lidar
        return data_dict

    # @print_exec_time
    def add_free_space_points(self, data_dict):
        # transform lidar points from ego to local, and to torch tensor to speed up runtime
        device = self.device
        lidar = torch.from_numpy(data_dict['lidar']).to(device).float()
        lidar_idx = torch.from_numpy(data_dict['lidar_idx']).to(device).float()
        tf_matrices = torch.from_numpy(data_dict['tf_matrices']).to(device).float()
        lidar_idx, lidar = self.lidar_transform(lidar, lidar_idx, tf_matrices)
        # get point lower than z_min=1.5m
        z_min = self.cfgs['free_space_h']
        m = lidar[:, 2] < z_min
        points = lidar[m][:, :3]
        points_idx = lidar_idx[m]

        # generate free space points based on points
        d = torch.norm(points[:, :2], dim=1).reshape(-1, 1)
        free_space_d = self.cfgs.get('free_space_d', 3)
        free_space_step = self.cfgs.get('free_space_step', 1)
        delta_d = torch.arange(1, free_space_d + free_space_step,
                               free_space_step,
                               device=device).reshape(1, -1)
        steps = delta_d.shape[1]
        tmp = (d - delta_d) / d  # Nxsteps
        xyz_new = points[:, None, :] * tmp[:, :, None] # Nx3x3
        points_idx = torch.tile(points_idx.reshape(-1, 1), (1, steps))
        ixyz = torch.cat([points_idx.reshape(-1, steps, 1), xyz_new], dim=-1)

        # 1.remove free space points with negative distances to lidar center
        # 2.remove free space points higher than z_min
        # 3.remove duplicated points with resolution 1m
        ixyz = ixyz[tmp > 0]
        ixyz = ixyz[ixyz[..., 3] < z_min]
        ixyz = ixyz[torch.randperm(len(ixyz))]
        selected = torch.unique(torch.floor(ixyz * 2.5).long(), return_inverse=True, dim=0)[1]
        ixyz = ixyz[torch.unique(selected)]

        # pad free space point intensity as -1
        ixyz = torch.cat([ixyz, - torch.ones_like(ixyz[:, :2])], dim=-1)

        # pad lidar with intensity 1 if not given
        if lidar.shape[1] == 4: #xyzl
            lidar = torch.cat([lidar[:, :3], np.ones_like(lidar[:, :1]), lidar[:, 3:]], dim=-1)
        assert lidar.shape[1] == 5 #xyzil

        # transform augmented points back to ego and numpy
        lidar = torch.cat([lidar, ixyz[:, 1:]], dim=0)
        lidar_idx = torch.cat([lidar_idx, ixyz[:, 0]], dim=0)
        lidar_idx, lidar = self.lidar_transform(lidar, lidar_idx, tf_matrices, ego2local=False)
        data_dict['lidar'] = lidar.cpu().numpy()
        data_dict['lidar_idx'] = lidar_idx.cpu().numpy()
        return data_dict

    def lidar_transform(self,
                        lidar: torch.Tensor,
                        lidar_idx: torch.Tensor,
                        tf_matrices: torch.Tensor,
                        ego2local=True):
        lidar_tf = []
        lidar_tf_idx = []
        uniq_idx = [int(x.item()) for x in torch.unique(lidar_idx)]
        for i in uniq_idx:
            mat = torch.inverse(tf_matrices[i]) if ego2local else tf_matrices[i]
            mask = lidar_idx == i
            cur_lidar = project_points_by_matrix(lidar[mask], mat, True)
            cur_idx = lidar_idx[mask]
            lidar_tf_idx.append(cur_idx)
            lidar_tf.append(cur_lidar)

        lidar_tf_idx = torch.cat(lidar_tf_idx, dim=0)
        lidar_tf = torch.cat(lidar_tf, dim=0)
        return lidar_tf_idx, lidar_tf

    def visualize_data(self, data_dict):
        boxes = data_dict['target_boxes']
        lidar = data_dict['xyz']
        lidar_idx = data_dict.get('coords_idx')
        bev_points = data_dict['target_bev_pts']
        labels_road = bev_points[:, -2]
        labels_vehicle = bev_points[:, -1]

        lr = self.cfgs['lidar_range']
        fig = plt.figure(figsize=((lr[3] - lr[0]) / 10, (lr[4] - lr[1]) / 10))
        ax = fig.add_subplot()
        ax = draw_points_boxes_plt(lr, points=bev_points[labels_road == 0],
                                   points_c='gray', ax=ax, return_ax=True)
        ax = draw_points_boxes_plt(lr, points=bev_points[labels_road == 1],
                                   points_c='blue', ax=ax, return_ax=True)
        ax = draw_points_boxes_plt(lr, points=bev_points[labels_vehicle == 1],
                                   points_c='red', ax=ax, return_ax=True)
        if lidar_idx is not None:
            for i in np.unique(lidar_idx):
                ax = draw_points_boxes_plt(lr, points=lidar[lidar_idx == i],
                                           points_c='c', ax=ax, return_ax=True,
                                           marker_size=1)
            else:
                ax = draw_points_boxes_plt(lr, points=lidar,
                                           points_c='c', ax=ax, return_ax=True,
                                           marker_size=1)
        draw_points_boxes_plt(lr, boxes_gt=boxes, ax=ax)

        # draw bev-opv2v maps
        draw_img(data_dict['bevmap_dynamic'])

    def post_process(self, batch_dict):
        out_dict = {}
        if self.post_processes is not None:
            out_dict = self.post_processes(batch_dict)
        return out_dict

    def compose_coords_features(self, coords_symbols, feature_symbols, lidar):
        """
        Compose cooordinates and features according to the symbols,
        each valid symbol will be mapped to a self.get_feature_[symbol] function
        to get the corresponding feature in lidar. Valid symbols are
        - 'x'(coordinate),
        - 'y'(coordinate),
        - 'z'(coordinate),
        - 'i'(intensity),
        - 't'(theta in degree),
        - 'c'(cos(t)),
        - 's'(sin(t)).
        :param coords_symbols: list of symbols
        :param feature_symbols: list of symbols
        :param lidar: np.ndarray [N, 3+c], columns 1-3 are x, y, z,
               if intensity is availble, it should in the 4th column
        :return: (np.ndarray [N, len(coords_symbols)],
                np.ndarray [N, len(feature_symbols)]),
                composed coordinate and features
        """
        symbols = set(coords_symbols.split(','))
        symbols.update(set(feature_symbols.split(',')))
        data = {s: getattr(self, f'get_feature_{s.strip()}')(lidar) for s in symbols}
        coords = np.concatenate([data[s] for s in coords_symbols.split(',')], axis=1)
        features = np.concatenate([data[s] for s in feature_symbols.split(',')], axis=1)

        return coords, features

    # Feature retrieving functions, input lidar columns must be # [x,y,z,i,obj,cls]
    @staticmethod
    def get_feature_x(lidar):
        return lidar[:, 0].reshape(-1, 1)

    @staticmethod
    def get_feature_y(lidar):
        return lidar[:, 1].reshape(-1, 1)

    @staticmethod
    def get_feature_z(lidar):
        return lidar[:, 2].reshape(-1, 1)

    @staticmethod
    def get_feature_i(lidar):
        if lidar.shape[1] > 3:
            return lidar[:, 3].reshape(-1, 1)
        else:
            return np.ones_like(lidar[:, 0]).reshape(-1, 1)

    @staticmethod
    def get_feature_t(lidar):
        degs = np.rad2deg(np.arctan2(lidar[:, 1], lidar[:, 0]).reshape(-1, 1))
        degs = (degs + 360) % 360
        return degs

    @staticmethod
    def get_feature_d(lidar):
        return np.linalg.norm(lidar[:, :2], axis=1).reshape(-1, 1)

    @staticmethod
    def get_feature_c(lidar):
        return np.cos(np.arctan2(lidar[:, 1], lidar[:, 0])).reshape(-1, 1)

    @staticmethod
    def get_feature_s(lidar):
        return np.sin(np.arctan2(lidar[:, 1], lidar[:, 0])).reshape(-1, 1)

    @staticmethod
    def get_feature_cs(lidar):
        x_abs = 1 / (np.abs(lidar[:, 1] / (lidar[:, 0] +
                            (lidar[:, 0] == 0) * 1e-6)) + 1)
        y_abs = 1 - x_abs
        x = x_abs * np.sign(lidar[:, 0])
        y = y_abs * np.sign(lidar[:, 1])
        return np.stack([x, y], axis=1)

    @staticmethod
    def collate_batch(data_list):
        # data_list = list(itertools.chain(*data_list))
        batch_dict = {k: [data[k] for data in data_list] for k in data_list[0].keys()}
        ret = {}

        for key, val in batch_dict.items():
            if val[0] is None:
                ret[key] = None
                continue
            if key in ['target_boxes']:
                coors = []
                for i, coor in enumerate(val):
                    if isinstance(coor, np.ndarray) and len(coor) > 0:
                        coor = torch.from_numpy(coor).float()
                        coor_pad = F.pad(coor, (1, 0, 0, 0), mode="constant", value=i)
                        coors.append(coor_pad)
                if len(coors) > 0:
                    ret[key] = torch.cat(coors, dim=0)
                else:
                    ret[key] = torch.empty(0, 8)
            elif key in ['features', 'xyz', 'coords', 'target_bev_pts']:
                ret[key] = torch.from_numpy(np.concatenate(val, axis=0)).float()
            elif key in ['coords_idx', 'target_bev_pts_idx']:
                cnts = []
                indices = []
                for b, cur_indices in enumerate(val):
                    cur_indices += sum(cnts)
                    cnts.append(len(np.unique(cur_indices)))
                    indices.append(cur_indices)
                ret[key] = torch.from_numpy(np.concatenate(indices, axis=0)).float()
                ret['num_cav'] = cnts
            elif key in ['target_semantic']:
                ret[key] = torch.from_numpy(np.concatenate(val, axis=0)).long()
            elif key in ['bevmap_static', 'bevmap_dynamic']:
                if isinstance(val[0], np.ndarray):
                    ret[key] = torch.from_numpy(np.stack(val, axis=0)).float()
                else:
                    ret[key] = torch.stack(val, dim=0).float()
            else:
                ret[key] = val
        ret['coords'] = torch.cat([ret.pop('coords_idx').view(-1, 1), ret['coords']], dim=-1)
        if ret['target_bev_pts'] is not None:
            ret['target_bev_pts'] = torch.cat([ret.pop('target_bev_pts_idx').view(-1, 1),
                                               ret['target_bev_pts']], dim=-1)
        ret['batch_size'] = len(data_list)

        return ret

