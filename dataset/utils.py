import numpy as np


def project_points_by_matrix(points, transformation_matrix):
    """
    Project the points to another coordinate system based on the
    transformation matrix.
    """

    # convert to homogeneous coordinates via padding 1 at the last dimension.
    # (N, 4)
    points_homogeneous = np.concatenate(
        [points, np.ones_like(points[:, :1])], axis=1)
    # (N, 4)
    projected_points = transformation_matrix @ points_homogeneous.T

    return projected_points[:3].T