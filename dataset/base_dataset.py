import logging, tqdm
import numpy as np
import torch
import torch.nn.functional as F
import dataset.processors as PP
from torch.utils.data import Dataset
from dataset.utils import project_points_by_matrix
from utils.pclib import rotate_points_along_z
from utils.box_utils import limit_period
from utils.vislib import draw_points_boxes_plt


class BaseDataset(Dataset):
    LABEL_COLORS = {}
    VALID_CLS = []

    def __init__(self, cfgs, mode):
        self.cfgs = cfgs
        self.mode = mode
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
        data_dict = self.crop_points_range(data_dict)
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

    def sample_bev_pts(self, data_dict):
        """
        Sample BEV points based on bev map and lidar points.
        :param data_dict: dict
        :return: data_dict: updated with sampled bev_points -
                np.ndarray [K, 3], columns are (x, y, bev_cls)
        """
        bevmap_static = data_dict['bevmap_static']
        bevmap_dynamic = data_dict['bevmap_dynamic']
        x_lim = self.cfgs['lidar_range'][3]
        y_lim = self.cfgs['lidar_range'][4]
        lidar = data_dict['lidar']
        lidar_idx = data_dict['lidar_idx']

        bev_points = []
        bev_points_idx = []

        for idx in sorted(np.unique(lidar_idx)):
            points = lidar[lidar_idx == idx][:, :2]
            # ramdom sampling
            res = 0.4
            points = np.unique(points // res, axis=0) * res
            points = points.reshape(-1, 2, 1) + np.random.normal(0, 3, (len(points), 2, 10))
            points = np.unique(points // res, axis=0) * res
            points = points.transpose(0, 2, 1).reshape(-1, 2)
            if not self.cfgs.get("discrete_bev_pts", False):
                points = points + np.random.normal(0, 1, (len(points), 2))

            h, w = bevmap_static.shape
            pixels_per_meter = 1 / self.cfgs['bev_res']
            xs = np.clip(np.floor((points[:, 0] + x_lim) * pixels_per_meter)
                         .astype(int), a_min=0, a_max=h - 1)
            ys = np.clip(np.floor((points[:, 1] + y_lim) * pixels_per_meter)
                         .astype(int), a_min=0, a_max=w - 1)
            bev_static = bevmap_static[xs, ys]
            bev_dynamic = bevmap_dynamic[xs, ys]

            labels = np.zeros_like(points[:, :2]) # neg static
            labels[bev_static > 0, 0] = 1  # pos dynamic

            # sample dynamic points
            labels[bev_dynamic > 0, 1] = 1
            bev_pts = np.concatenate([points, labels], axis=1)
            # sample static points
            neg_idx = np.where(bev_pts[:, -2] == 0)[0]
            if len(neg_idx) > 3000:
                neg_idx = np.random.choice(neg_idx, 3000)
            road_idx = np.where(bev_pts[:, -2] == 1)[0]
            if len(road_idx) > 3000:
                road_idx = np.random.choice(road_idx, 3000)

            veh_idx = np.where(bev_pts[:, -1] == 1)[0]
            selected = np.concatenate([neg_idx, road_idx, veh_idx], axis=0)
            np.random.shuffle(selected)
            bev_pts = bev_pts[selected]
            bev_points.append(bev_pts)
            bev_points_idx.append(np.ones_like(bev_pts[:, 0]) * idx)
        bev_points = np.concatenate(bev_points, axis=0)
        bev_points_idx = np.concatenate(bev_points_idx, axis=0)
        data_dict.update({
            'pts': bev_points,
            'bev_pts_idx': bev_points_idx,
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
            # make sure the bev pts for these cavs are also removed
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
        assert len(self.VALID_CLS) > 0, 'VALID_CLS not set!'
        cls = np.zeros_like(lidar[:, -1])
        for tgt_cls, src_cls_list in enumerate(self.VALID_CLS):
            for src_cls in src_cls_list:
                cls[lidar[:, -1] == src_cls] = tgt_cls + 1
        cls[lidar[:, -1] == -1] = -1 # free space points
        lidar[:, -1] = cls
        data_dict['lidar'] = lidar
        return data_dict

    def add_free_space_points(self, data_dict):
        # we set maximum height of free space points to z_min=1.5m
        # select points lower than z_min
        self.lidar_transform(data_dict)
        lidar = data_dict['lidar']
        lidar_idx = data_dict['lidar_idx']
        z_min = self.cfgs['free_space_h']
        m = lidar[:, 2] < z_min
        points = lidar[m][:, :3]
        points_idx = lidar_idx[m]
        # calculate
        d = self.get_feature_d(points)
        free_space_d = self.cfgs.get('free_space_d', 3)
        free_space_step = self.cfgs.get('free_space_step', 1)
        steps = int(free_space_d / free_space_step)
        delta_d = np.linspace(1, free_space_d,
                              steps).reshape(1, steps)
        tmp = (d - delta_d) / d # Nx5
        xyz_new = points[:, None, :] * tmp[:, :, None] # Nx3x3
        points_idx = np.tile(points_idx.reshape(-1, 1), (1, steps))
        m = tmp > 0
        xyz_new = xyz_new[m]
        points_idx = points_idx[m]
        m = xyz_new[..., 2] < z_min
        xyz_new = xyz_new[m]
        points_idx = points_idx[m]
        ixyz = np.concatenate([points_idx.reshape(-1, 1), xyz_new], axis=1)
        selected = np.unique(np.floor(ixyz).astype(int),  # resolution 1m
                             return_index=True, axis=0)[1]
        xyz_new = xyz_new[selected]
        points_idx = points_idx[selected]
        xyz_new = np.concatenate([
            xyz_new,
            - np.ones_like(xyz_new[:, :2])
        ], axis=-1)
        # pad lidar with intensity 1
        if lidar.shape[1] == 4: #xyzl
            lidar = np.concatenate([
                lidar[:, :3],
                np.ones_like(lidar[:, :1]),
                lidar[:, 3:]
            ], axis=-1)
        assert lidar.shape[1] == 5 #xyzil
        lidar = np.concatenate([lidar, xyz_new], axis=0)
        lidar_idx = np.concatenate([lidar_idx, points_idx], axis=0)
        data_dict['lidar'] = lidar
        data_dict['lidar_idx'] = lidar_idx
        self.lidar_transform(data_dict, ego2local=False)
        return data_dict

    def lidar_transform(self, data_dict, ego2local=True):
        lidar = data_dict['lidar']
        lidar_idx = data_dict['lidar_idx']
        if 'tf_matrices' in data_dict:
            tf_matrices = data_dict['tf_matrices']
            for i in np.unique(lidar_idx).astype(int):
                mat = np.linalg.inv(tf_matrices[i]) if ego2local else tf_matrices[i]
                lidar[lidar_idx==i, :3] = project_points_by_matrix(lidar[lidar_idx==i, :3], mat)

    def visualize_data(self, data_dict):
        boxes = data_dict['target_boxes']
        lidar = data_dict['xyz']
        lidar_idx = data_dict.get('coords_idx')
        bev_points = data_dict['target_bev_pts']
        labels_road = bev_points[:, -2]
        labels_vehicle = bev_points[:, -1]

        r = self.cfgs['lidar_range']
        ax = draw_points_boxes_plt(r, points=bev_points[labels_road == 0],
                                   points_c='gray', return_ax=True)
        ax = draw_points_boxes_plt(r, points=bev_points[labels_road == 1],
                                   points_c='blue', ax=ax, return_ax=True)
        ax = draw_points_boxes_plt(r, points=bev_points[labels_vehicle == 1],
                                   points_c='red', ax=ax, return_ax=True)
        if lidar_idx is not None:
            for i in np.unique(lidar_idx):
                ax = draw_points_boxes_plt(r, points=lidar[lidar_idx == i],
                                           points_c='c', ax=ax, return_ax=True,
                                           marker_size=0.5)
            else:
                ax = draw_points_boxes_plt(r, points=lidar,
                                           points_c='c', ax=ax, return_ax=True,
                                           marker_size=0.5)
        draw_points_boxes_plt(r, boxes_gt=boxes, ax=ax)

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
                ret[key] = torch.from_numpy(np.stack(val, axis=0)).float()
            else:
                ret[key] = val
        ret['coords'] = torch.cat([ret.pop('coords_idx').view(-1, 1), ret['coords']], dim=-1)
        if ret['target_bev_pts'] is not None:
            ret['target_bev_pts'] = torch.cat([ret.pop('target_bev_pts_idx').view(-1, 1),
                                               ret['target_bev_pts']], dim=-1)
        ret['batch_size'] = len(data_list)

        return ret

