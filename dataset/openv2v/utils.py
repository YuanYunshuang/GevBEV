# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib
import re
import yaml
import os

import open3d as o3d
import numpy as np
import torch
import torch.nn.functional as F


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def pcd_to_np(pcd_file):
    """
    Read  pcd and return numpy array.

    Parameters
    ----------
    pcd_file : str
        The pcd file that contains the point cloud.

    Returns
    -------
    pcd : o3d.PointCloud
        PointCloud object, used for visualization
    pcd_np : np.ndarray
        The lidar data in numpy format, shape:(n, 4)

    """
    pcd = o3d.io.read_point_cloud(pcd_file)

    xyz = np.asarray(pcd.points)
    # we save the intensity in the first channel
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
    pcd_np = np.hstack((xyz, intensity))

    return np.asarray(pcd_np, dtype=np.float32)


def mask_points_by_range(points, limit_range):
    """
    Remove the lidar points out of the boundary.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    limit_range : list
        [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """

    mask = (points[:, 0] > limit_range[0]) & (points[:, 0] < limit_range[3])\
           & (points[:, 1] > limit_range[1]) & (
                   points[:, 1] < limit_range[4]) \
           & (points[:, 2] > limit_range[2]) & (
                   points[:, 2] < limit_range[5])
    # mask = np.linalg.norm(points[:, :2], axis=-1) < limit_range[3] \
    #         & (points[:, 2] > limit_range[2]) & (
    #                points[:, 2] < limit_range[5])

    points = points[mask]

    return points


def mask_ego_points(points):
    """
    Remove the lidar points of the ego vehicle itself.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """
    mask = (points[:, 0] >= -1.95) & (points[:, 0] <= 2.95) \
           & (points[:, 1] >= -1.1) & (points[:, 1] <= 1.1)
    points = points[np.logical_not(mask)]

    return points


def shuffle_points(points):
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]

    return points


def lidar_project(lidar_data, extrinsic):
    """
    Given the extrinsic matrix, project lidar data to another space.

    Parameters
    ----------
    lidar_data : np.ndarray
        Lidar data, shape: (n, 4)

    extrinsic : np.ndarray
        Extrinsic matrix, shape: (4, 4)

    Returns
    -------
    projected_lidar : np.ndarray
        Projected lida data, shape: (n, 4)
    """

    lidar_xyz = lidar_data[:, :3].T
    # (3, n) -> (4, n), homogeneous transformation
    lidar_xyz = np.r_[lidar_xyz, [np.ones(lidar_xyz.shape[1])]]
    lidar_int = lidar_data[:, 3]

    # transform to ego vehicle space, (3, n)
    project_lidar_xyz = np.dot(extrinsic, lidar_xyz)[:3, :]
    # (n, 3)
    project_lidar_xyz = project_lidar_xyz.T
    # concatenate the intensity with xyz, (n, 4)
    projected_lidar = np.hstack((project_lidar_xyz,
                                 np.expand_dims(lidar_int, -1)))

    return projected_lidar


def projected_lidar_stack(projected_lidar_list):
    """
    Stack all projected lidar together.

    Parameters
    ----------
    projected_lidar_list : list
        The list containing all projected lidar.

    Returns
    -------
    stack_lidar : np.ndarray
        Stack all projected lidar data together.
    """
    stack_lidar = []
    for lidar_data in projected_lidar_list:
        stack_lidar.append(lidar_data)

    return np.vstack(stack_lidar)


def downsample_lidar(pcd_np, num):
    """
    Downsample the lidar points to a certain number.

    Parameters
    ----------
    pcd_np : np.ndarray
        The lidar points, (n, 4).

    num : int
        The downsample target number.

    Returns
    -------
    pcd_np : np.ndarray
        The downsampled lidar points.
    """
    assert pcd_np.shape[0] >= num

    selected_index = np.random.choice((pcd_np.shape[0]),
                                      num,
                                      replace=False)
    pcd_np = pcd_np[selected_index]

    return pcd_np


def downsample_lidar_minimum(pcd_np_list):
    """
    Given a list of pcd, find the minimum number and downsample all
    point clouds to the minimum number.

    Parameters
    ----------
    pcd_np_list : list
        A list of pcd numpy array(n, 4).
    Returns
    -------
    pcd_np_list : list
        Downsampled point clouds.
    """
    minimum = np.Inf

    for i in range(len(pcd_np_list)):
        num = pcd_np_list[i].shape[0]
        minimum = num if minimum > num else minimum

    for (i, pcd_np) in enumerate(pcd_np_list):
        pcd_np_list[i] = downsample_lidar(pcd_np, minimum)

    return pcd_np_list


def x_to_world_(pose):
    """
    The transformation matrix from x-coordinate system to carla world system

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = c_y * s_p * c_r + s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = s_y * s_p * c_r - c_y * s_r
    matrix[2, 0] = -s_p
    matrix[2, 1] = c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system
    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]
    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list
        The pose of x1 under world coordinates.
    x2 : list
        The pose of x2 under world coordinates.

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    x1_to_world = x_to_world(x1)
    x2_to_world = x_to_world(x2)
    world_to_x2 = np.linalg.inv(x2_to_world)

    transformation_matrix = np.dot(world_to_x2, x1_to_world)
    return transformation_matrix


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), radians, angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3].float(), rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def project_points_by_matrix_torch(points, transformation_matrix):
    """
    Project the points to another coordinate system based on the
    transformation matrix.

    Parameters
    ----------
    points : torch.Tensor
        3D points, (N, 3)
    transformation_matrix : torch.Tensor
        Transformation matrix, (4, 4)
    Returns
    -------
    projected_points : torch.Tensor
        The projected points, (N, 3)
    """
    points, is_numpy = \
        check_numpy_to_torch(points)
    transformation_matrix, _ = \
        check_numpy_to_torch(transformation_matrix)

    # convert to homogeneous coordinates via padding 1 at the last dimension.
    # (N, 4)
    points_homogeneous = F.pad(points, (0, 1), mode="constant", value=1)
    # (N, 4)
    projected_points = torch.einsum("ik, jk->ij", points_homogeneous,
                                    transformation_matrix)

    return projected_points[:, :3] if not is_numpy \
        else projected_points[:, :3].numpy()


def load_yaml(file, opt=None):
    """
    Load yaml file and return a dictionary.

    Parameters
    ----------
    file : string
        yaml file path.

    opt : argparser
         Argparser.
    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """
    if opt and opt.model_dir:
        file = os.path.join(opt.model_dir, 'config.yaml')

    stream = open(file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    param = yaml.load(stream, Loader=loader)
    if "yaml_parser" in param:
        param = eval(param["yaml_parser"])(param)

    return param


def create_bbx(extent):
    """
    Create bounding box with 8 corners under obstacle vehicle reference.

    Parameters
    ----------
    extent : list
        Width, height, length of the bbx.

    Returns
    -------
    bbx : np.array
        The bounding box with 8 corners, shape: (8, 3)
    """

    bbx = np.array([[extent[0], -extent[1], -extent[2]],
                    [extent[0], extent[1], -extent[2]],
                    [-extent[0], extent[1], -extent[2]],
                    [-extent[0], -extent[1], -extent[2]],
                    [extent[0], -extent[1], extent[2]],
                    [extent[0], extent[1], extent[2]],
                    [-extent[0], extent[1], extent[2]],
                    [-extent[0], -extent[1], extent[2]]])

    return bbx


def corner_to_center(corner3d, order='lwh'):
    """
    Convert 8 corners to x, y, z, dx, dy, dz, yaw.

    Parameters
    ----------
    corner3d : np.ndarray
        (N, 8, 3)

    order : str
        'lwh' or 'hwl'

    Returns
    -------
    box3d : np.ndarray
        (N, 7)
    """
    assert corner3d.ndim == 3
    batch_size = corner3d.shape[0]

    xyz = np.mean(corner3d[:, [0, 3, 5, 6], :], axis=1)
    h = abs(np.mean(corner3d[:, 4:, 2] - corner3d[:, :4, 2], axis=1,
                    keepdims=True))
    l = (np.sqrt(np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 5, [0, 1]] - corner3d[:, 6, [0, 1]]) ** 2,
                        axis=1, keepdims=True))) / 4

    w = (np.sqrt(
        np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2, axis=1,
               keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 5, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 6, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2,
                        axis=1, keepdims=True))) / 4

    theta = (np.arctan2(corner3d[:, 1, 1] - corner3d[:, 2, 1],
                        corner3d[:, 1, 0] - corner3d[:, 2, 0]) +
             np.arctan2(corner3d[:, 0, 1] - corner3d[:, 3, 1],
                        corner3d[:, 0, 0] - corner3d[:, 3, 0]) +
             np.arctan2(corner3d[:, 5, 1] - corner3d[:, 6, 1],
                        corner3d[:, 5, 0] - corner3d[:, 6, 0]) +
             np.arctan2(corner3d[:, 4, 1] - corner3d[:, 7, 1],
                        corner3d[:, 4, 0] - corner3d[:, 7, 0]))[:,
            np.newaxis] / 4

    if order == 'lwh':
        return np.concatenate([xyz, l, w, h, theta], axis=1).reshape(
            batch_size, 7)
    elif order == 'hwl':
        return np.concatenate([xyz, h, w, l, theta], axis=1).reshape(
            batch_size, 7)
    else:
        raise NotImplementedError


def mask_boxes_outside_range_numpy(boxes, limit_range, order,
                                   min_num_corners=8, return_mask=False):
    """
    Parameters
    ----------
    boxes: np.ndarray
        (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    limit_range: list
        [minx, miny, minz, maxx, maxy, maxz]

    min_num_corners: int
        The required minimum number of corners to be considered as in range.

    order : str
        'lwh' or 'hwl'

    return_mask : bool
        Whether return the mask.

    Returns
    -------
    boxes: np.ndarray
        The filtered boxes.
    """
    assert boxes.shape[1] == 8 or boxes.shape[1] == 7

    new_boxes = boxes.copy()
    if boxes.shape[1] == 7:
        new_boxes = boxes_to_corners_3d(new_boxes, order)

    mask = ((new_boxes >= limit_range[0:3]) &
            (new_boxes <= limit_range[3:6])).all(axis=2)
    mask = mask.sum(axis=1) >= min_num_corners  # (N)

    if return_mask:
        return boxes[mask], mask
    return boxes[mask]


def boxes_to_corners_3d(boxes3d, order):
    """
        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2
    Parameters
    __________
    boxes3d: np.ndarray or torch.Tensor
        (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center.

    order : str
        'lwh' or 'hwl'

    Returns:
        corners3d: np.ndarray or torch.Tensor
        (N, 8, 3), the 8 corners of the bounding box.

    """
    # ^ z
    # |
    # |
    # | . x
    # |/
    # +-------> y

    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    boxes3d_ = boxes3d

    if order == 'hwl':
        boxes3d_ = boxes3d[:, [0, 1, 2, 5, 4, 3, 6]]

    template = boxes3d.new_tensor((
        [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
        [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1],
    )) / 2

    corners3d = boxes3d_[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3),
                                      boxes3d_[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d_[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def project_world_objects(object_dict,
                          output_dict,
                          lidar_pose,
                          lidar_range,
                          order):
    """
    Project the objects under world coordinates into another coordinate
    based on the provided extrinsic.

    Parameters
    ----------
    object_dict : dict
        The dictionary contains all objects surrounding a certain cav.

    output_dict : dict
        key: object id, value: object bbx (xyzlwhyaw).

    lidar_pose : list
        (6, ), lidar pose under world coordinate, [x, y, z, roll, yaw, pitch].

    lidar_range : list
         [minx, miny, minz, maxx, maxy, maxz]

    order : str
        'lwh' or 'hwl'
    """
    for object_id, object_content in object_dict.items():
        location = object_content['location']
        rotation = object_content['angle']
        center = object_content['center']
        extent = object_content['extent']

        object_pose = [location[0] + center[0],
                       location[1] + center[1],
                       location[2] + center[2],
                       rotation[0], rotation[1], rotation[2]]
        object2lidar = x1_to_x2(object_pose, lidar_pose)

        # shape (3, 8)
        bbx = create_bbx(extent).T
        # bounding box under ego coordinate shape (4, 8)
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

        # project the 8 corners to lidar coordinate
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = corner_to_center(bbx_lidar, order=order)
        bbx_lidar = mask_boxes_outside_range_numpy(bbx_lidar,
                                                   lidar_range,
                                                   order)

        if bbx_lidar.shape[0] > 0:
            output_dict.update({object_id: bbx_lidar})


def generate_object_center(cav_contents,
                           filter_range,
                           reference_lidar_pose,
                           max_num=100):
    """
    Retrieve all objects in a format of (n, 7), where 7 represents
    x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

    Parameters
    ----------
    cav_contents : list
        List of dictionary, save all cavs' information.

    reference_lidar_pose : list
        The final target lidar pose with length 6.

    filter_range : list
         [minx, miny, minz, maxx, maxy, maxz]

    Returns
    -------
    object_np : np.ndarray
        Shape is (max_num, 7).
    mask : np.ndarray
        Shape is (max_num,).
    object_ids : list
        Length is number of bbx in current sample.
    """

    tmp_object_dict = {}
    for cav_content in cav_contents:
        tmp_object_dict.update(cav_content['params']['vehicles'])

    output_dict = {}

    project_world_objects(tmp_object_dict,
                          output_dict,
                          reference_lidar_pose,
                          filter_range,
                          'lwh')

    object_np = np.zeros((max_num, 7))
    mask = np.zeros(max_num)
    object_ids = []

    for i, (object_id, object_bbx) in enumerate(output_dict.items()):
        object_np[i] = object_bbx[0, :]
        mask[i] = 1
        object_ids.append(object_id)

    return object_np, mask, object_ids
