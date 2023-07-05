import numpy as np
import torch


def project_points_by_matrix(points, transformation_matrix, to_torch=False):
    """
    Project the points to another coordinate system based on the
    transformation matrix.
    """
    is_numpy = False
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).cuda()
        is_numpy = True
    if isinstance(transformation_matrix, np.ndarray):
        transformation_matrix = torch.from_numpy(transformation_matrix).to(points.device)
    # ensure float
    points = points.float()
    transformation_matrix = transformation_matrix.float()
    # convert to homogeneous coordinates via padding 1 at the last dimension.
    # (N, 4)
    points_homogeneous = torch.cat(
        [points[:, :3], torch.ones_like(points[:, :1])], dim=1)
    # (N, 4)
    projected_points = transformation_matrix @ points_homogeneous.T
    projected_points = torch.cat([projected_points[:3].T, points[:, 3:]], dim=-1)

    if is_numpy and not to_torch:
        projected_points = projected_points.cpu().numpy()
    return projected_points

