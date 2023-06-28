import numpy as np
import torch


def cart2cyl(input_xyz):
    rho = np.sqrt(input_xyz[..., 0] ** 2 + input_xyz[..., 1] ** 2)
    phi = np.arctan2(input_xyz[..., 1], input_xyz[..., 0])
    return np.concatenate((rho.reshape(-1, 1), phi.reshape(-1, 1), input_xyz[..., 2:]), axis=-1)


def cyl2cart(input_xyz_polar):
    x = input_xyz_polar[..., 0] * np.cos(input_xyz_polar[..., 1])
    y = input_xyz_polar[..., 0] * np.sin(input_xyz_polar[..., 1])
    return np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), input_xyz_polar[..., 2:]), axis=-1)


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (N, 3 + C or 2 + C)
        angle: float, angle along z-axis, angle increases x ==> y
    Returns:

    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    rot_matrix = np.array([
        [cosa,  sina, 0],
        [-sina, cosa, 0],
        [0, 0, 1]
    ]).astype(float)
    if points.shape[1]==2:
        points_rot = np.matmul(points, rot_matrix[:2, :2])
    elif points.shape[1]>2:
        points_rot = np.matmul(points[:, 0:3], rot_matrix)
        points_rot = np.concatenate((points_rot, points[:, 3:]), axis=-1)
    else:
        raise IOError('Input points should have the shape: (N, 3 + C or 2 + C).')
    return points_rot


def rotate_points_along_z_torch(points, angle):
    """
    Args:
        points: (N, 2 + C) or (B, 2 + C)
        angle: float or tensor of shape (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    if len(points.shape) == 2:
        points = points.unsqueeze(0)
    if isinstance(angle, float):
        angle = torch.tensor([angle], device=points.device)
    else:
        assert isinstance(angle, torch.Tensor)
        assert points.shape[0] == 1 or angle.shape[0] == points.shape[0]
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    rot_matrix = torch.stack([
        torch.stack([cosa,  sina], dim=-1),
        torch.stack([-sina, cosa], dim=-1)
    ], dim=1).float().to(points.device)
    if points.shape[0] == 1 and angle.shape[0] > 1:
        points = torch.tile(points, (len(rot_matrix), 1, 1))
    points_rot = torch.bmm(points[..., 0:2], rot_matrix)
    points_rot = torch.cat((points_rot, points[..., 2:]), dim=-1)
    return points_rot


def mask_values_in_range(values, min,  max):
    return np.logical_and(values>min, values<max)


def mask_points_in_box(points, pc_range):
    n_ranges = len(pc_range) // 2
    list_mask = [mask_values_in_range(points[:, i], pc_range[i],
                                       pc_range[i+n_ranges]) for i in range(n_ranges)]
    return np.array(list_mask).all(axis=0)


def mask_points_in_range(points: np.array, dist: float) -> np.array:
    """

    :rtype: np.array
    """
    return np.linalg.norm(points[:, :2], axis=1) < dist


def get_tf_matrix_torch(vectors, inv=False):
    device = vectors.device
    n, _ = vectors.shape
    xs = vectors[:, 0]
    ys = vectors[:, 1]
    angles = vectors[:, 2]
    cosa = torch.cos(angles)
    sina = torch.sin(angles)
    ones = torch.ones_like(angles)
    zeros = torch.zeros_like(angles)
    rot_matrix = torch.zeros((n, 3, 3), device=device, requires_grad=True)
    rot_matrix[:, 0, 0] = cosa
    rot_matrix[:, 0, 1] = -sina
    rot_matrix[:, 1, 0] = sina
    rot_matrix[:, 1, 1] = cosa
    shift_matrix = torch.zeros_like(rot_matrix, requires_grad=True)
    shift_matrix[:, 0, 1] = xs
    shift_matrix[:, 1, 0] = ys
    shift_matrix[:, [0, 1, 2], [0, 1, 2]] = 1.0
    if inv:
        mat = torch.einsum('...ij, ...jk', rot_matrix, shift_matrix)
    else:
        mat = torch.einsum('...ij, ...jk', shift_matrix, rot_matrix)
    return mat, rot_matrix, shift_matrix


def pose_err_global2relative_torch(poses, errs):
    """
    Calcaulate relative pose transformation based on the errorneous global positioning
    :param poses: Nx2 or Nx3, first row is ego pose, other rows are the coop poses
    :param errs: Nx3, first row is ego pose error and other rows for coop pose errors
    :return: (N-1)x3, relative localization errors between ego and coop vehicles
    """
    if poses.shape[-1]==2:
        poses = torch.cat([poses, torch.zeros_like(poses[:, 0:1])], dim=-1)
    poses_err = poses + errs

    R01, _, _ = get_tf_matrix_torch(-poses[:1], inv=True)
    R10_hat, _, _ = get_tf_matrix_torch(poses_err[:1])
    R20, _, _ = get_tf_matrix_torch(poses[1:])
    R02_hat, _, _ = get_tf_matrix_torch(-poses_err[1:], inv=True)

    delta_R21 = torch.einsum('...ij, ...jk', R01, R20)
    delta_R21 = torch.einsum('...ij, ...jk', delta_R21, R02_hat)
    delta_R21 = torch.einsum('...ij, ...jk', delta_R21, R10_hat)

    x = delta_R21[0, 2]
    y = delta_R21[1, 2]
    theta = torch.atan2(delta_R21[1, 0], delta_R21[0, 0])
    return torch.stack([x, y, theta], dim=-1)

