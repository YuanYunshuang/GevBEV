import torch
from torch import nn
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiKernelGenerator import KernelGenerator
from torch.distributions.multivariate_normal import _batch_mahalanobis

pi = 3.141592653

def limit_period(val, offset=0.5, period=2 * pi):
    return val - torch.floor(val / period + offset) * period

@torch.no_grad()
def normalize_points(points, centroids, tensor_map):
    tensor_map = tensor_map if tensor_map.dtype == torch.int64 else tensor_map.long()
    norm_points = points - centroids[tensor_map]
    return norm_points


@torch.no_grad()
def normalize_centroids(down_points, coordinates, tensor_stride):
    norm_points = (down_points - coordinates[:, 1:]) / tensor_stride - 0.5
    return norm_points


@torch.no_grad()
def get_kernel_map_and_out_key(stensor, stensor_out=None,
                               kernel_size=3, stride=1, dilation=1,
                               kernel_type='cube', kernel_generator=None):
    """
    Generate kernel maps for the input stensor.
    The hybrid and custom kernel is not implemented in ME v0.5.x,
    this function uses a kernel mask to select the kernel maps for
    the customized kernel shapes.
    :param stensor: ME.SparseTensor, NxC
    :param kernel_type: 'cube'(default) | 'hybrid'
    :return: masked kernel maps
    """
    D = stensor.C.shape[-1] - 1
    if kernel_generator is None:
        kernel_generator = KernelGenerator(kernel_size=kernel_size,
                                           stride=stride,
                                           dilation=dilation,
                                           dimension=D)
    assert D == len(kernel_generator.kernel_stride)
    cm = stensor.coordinate_manager
    in_key = stensor.coordinate_key
    if stensor_out is None:
        out_key = cm.stride(in_key, kernel_generator.kernel_stride)
    else:
        out_key = stensor_out.coordinate_key
    region_type, region_offset, _ = kernel_generator.get_kernel(
        stensor.tensor_stride, False)

    kernel_map = cm.kernel_map(in_key,
                               out_key,
                               kernel_generator.kernel_stride,
                               kernel_generator.kernel_size,
                               kernel_generator.kernel_dilation,
                               region_type=region_type,
                               region_offset=region_offset)
    if kernel_type=='cube':
        kernel_volume = kernel_generator.kernel_volume
    elif kernel_type=='hybrid':
        assert dilation == 1, "currently, hybrid kernel only support dilation=1."
        xx = torch.tensor([-1, 0, 1]).int()
        xx_list = [xx for i in range(D)]
        kernels = torch.meshgrid([*xx_list], indexing='ij')
        kernels = torch.stack([t.flatten() for t in kernels], dim=1)
        kernel_mask = torch.zeros_like(kernels[:, 0]).bool()
        m = torch.logical_or(
            kernels[:, 0] == 0,
            torch.logical_and(kernels[:, 0]==-1, (kernels[:, 1:]==0).all(dim=1))
        )
        kernel_mask[m] = True
        kernel_mask_map = {ic.item(): ih for ih, ic in enumerate(torch.where(kernel_mask)[0])}
        kernel_map = {kernel_mask_map[k]: v for k, v in kernel_map.items() if kernel_mask[k]}
        kernel_volume = kernel_mask.sum().item()
    else:
        raise NotImplementedError

    return kernel_map, out_key, kernel_volume


@torch.no_grad()
def downsample_points(points, tensor_map, field_map, size):
    down_points = ME.MinkowskiSPMMAverageFunction().apply(
        tensor_map, field_map, size, points
    )
    _, counts = torch.unique(tensor_map, return_counts=True)
    return down_points, counts.unsqueeze_(1).type_as(down_points)


@torch.no_grad()
def stride_centroids(points, counts, rows, cols, size):
    stride_centroids = ME.MinkowskiSPMMFunction().apply(rows, cols, counts, size, points)
    ones = torch.ones(size[1], dtype=points.dtype, device=points.device)
    stride_counts = ME.MinkowskiSPMMFunction().apply(rows, cols, ones, size, counts)
    stride_counts.clamp_(min=1)
    return torch.true_divide(stride_centroids, stride_counts), stride_counts


def downsample_embeddings(embeddings, inverse_map, size, mode="avg"):
    assert len(embeddings) == size[1]
    assert mode in ["avg", "max"]
    if mode == "max":
        in_map = torch.arange(size[1], dtype=inverse_map.dtype, device=inverse_map.device)
        down_embeddings = ME.MinkowskiDirectMaxPoolingFunction().apply(
            in_map, inverse_map, embeddings, size[0]
        )
    else:
        cols = torch.arange(size[1], dtype=inverse_map.dtype, device=inverse_map.device)
        down_embeddings = ME.MinkowskiSPMMAverageFunction().apply(
            inverse_map, cols, size, embeddings
        )
    return down_embeddings


def minkconv_layer(in_dim, out_dim, kernel_size, stride, d, bn_momentum, tr=False):
    kernel = [kernel_size] * d
    if tr:
        conv = getattr(ME, 'MinkowskiConvolutionTranspose')
    else:
        conv = getattr(ME, 'MinkowskiConvolution')
    conv_layer = conv(
        in_channels=in_dim,
        out_channels=out_dim,
        kernel_size=kernel,
        stride=stride,
        dilation=1,
        dimension=d
    )
    return conv_layer


def minkconv_conv_block(in_dim, out_dim, kernel, stride, d, bn_momentum,
                        activation='LeakyReLU',
                        tr=False,
                        expand_coordinates=False,
                        norm_before=False):
    if isinstance(kernel, int):
        kernel = [kernel] * d
    if isinstance(stride, int):
        stride = [stride] * d
    if tr:
        conv = getattr(ME, 'MinkowskiConvolutionTranspose')
    else:
        conv = getattr(ME, 'MinkowskiConvolution')
    conv_layer = conv(
        in_channels=in_dim,
        out_channels=out_dim,
        kernel_size=kernel,
        stride=stride,
        dilation=1,
        dimension=d,
        expand_coordinates=expand_coordinates
    )
    activation_fn = getattr(ME, f'Minkowski{activation}')()
    norm_layer = ME.MinkowskiBatchNorm(out_dim, momentum=bn_momentum)
    if norm_before:
        layer = nn.Sequential(conv_layer, norm_layer, activation_fn)
    else:
        layer = nn.Sequential(conv_layer, activation_fn, norm_layer)
    return layer


def linear_last(in_channels, mid_channels, out_channels, bias=False):
    return nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=bias),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, out_channels)
        )


def linear_layers(in_out, activations=None):
    if activations is None:
        activations = ['ReLU'] * (len(in_out) - 1)
    elif isinstance(activations, str):
        activations = [activations] * (len(in_out) - 1)
    else:
        assert len(activations) == (len(in_out) - 1)
    layers = []
    for i in range(len(in_out) - 1):
        layers.append(nn.Linear(in_out[i], in_out[i+1], bias=False))
        layers.append(nn.BatchNorm1d(in_out[i+1]))
        layers.append(getattr(nn, activations[i])())
    return nn.Sequential(*layers)


def meshgrid(xmin, xmax, dim, n_steps=None, step=None):
    assert dim <= 3, f'dim <= 3, but dim={dim} is given.'
    if n_steps is not None:
        x = torch.linspace(xmin, xmax, n_steps)
    elif step is not None:
        x = torch.arange(xmin, xmax, step)
    else:
        raise NotImplementedError
    xs = (x, ) * dim
    indexing = 'ijk'
    indexing = indexing[:dim]
    coor = torch.stack(
        torch.meshgrid(*xs, indexing=indexing),
        dim=-1
    )
    return coor


def metric2indices(coor, voxel_size):
    """"Round towards floor"""
    indices = coor.clone()
    indices[:, 1:] = indices[:, 1:] / voxel_size
    return torch.floor(indices).long()


def indices2metric(indices, voxel_size):
    """Voxel indices to voxel center in meter"""
    coor = indices.clone().float()
    coor[:, 1:] = (coor[:, 1:] + 0.5) * voxel_size
    return coor


def pad_r(tensor, value=0):
    tensor_pad = torch.ones_like(tensor[..., :1]) * value
    return torch.cat([tensor, tensor_pad], dim=-1)


def pad_l(tensor, value=0):
    tensor_pad = torch.ones_like(tensor[..., :1]) * value
    return torch.cat([tensor_pad, tensor], dim=-1)


def fuse_batch_indices(coords, num_cav):
    """
    Fusing voxels of CAVs from the same frame
    :param stensor: ME sparse tensor
    :param num_cav: list of number of CAVs for each frame
    :return: fused coordinates and features of stensor
    """

    for i, c in enumerate(num_cav):
        idx_start = sum(num_cav[:i])
        mask = torch.logical_and(
            coords[:, 0] >= idx_start,
            coords[:, 0] < idx_start + c
        )
        coords[mask, 0] = i

    return coords


def weighted_mahalanobis_dists(reg_evi, reg_var, dists, var0):
    log_probs_list = []
    for i in range(reg_evi.shape[1]):
        vars = reg_var[:, i, :] + var0[i]
        covs = torch.diag_embed(vars.squeeze(), dim1=1)
        unbroadcasted_scale_tril = covs.unsqueeze(1)  # N 1 2 2

        # a.shape = (i, 1, n, n), b = (..., i, j, n),
        M = _batch_mahalanobis(unbroadcasted_scale_tril, dists)  # N M
        log_probs = -0.5 * M
        log_probs_list.append(log_probs)

    log_probs = torch.stack(log_probs_list, dim=-1)
    probs = log_probs.exp()  # N M 2
    cls_evi = reg_evi.view(-1, 1, 2)  # N 1 2
    probs_weighted = probs * cls_evi

    return probs_weighted


def draw_sample_prob(centers, reg, samples, res, distr_r, det_r, batch_size, var0):
    # from utils.vislib import draw_points_boxes_plt
    # vis_ctrs = centers[centers[:, 0]==0, 1:].cpu().numpy()
    # vis_sams = samples[samples[:, 0]==0, 1:].cpu().numpy()
    #
    # ax = draw_points_boxes_plt(50, vis_ctrs, points_c='r', return_ax=True)
    # draw_points_boxes_plt(50, vis_sams, points_c='b', ax=ax)
    reg_evi = reg[:, :2]
    reg_var = reg[:, 2:].view(-1, 2, 2)

    grid_size = int(det_r / res) * 2
    centers_map = torch.ones((batch_size, grid_size, grid_size),
                              device=reg.device).long() * -1
    ctridx = metric2indices(centers, res).T
    ctridx[1:] += int(grid_size / 2)
    centers_map[ctridx[0], ctridx[1], ctridx[2]] = torch.arange(ctridx.shape[1],
                                                                device=ctridx.device)

    steps = int(distr_r / res)
    offset = meshgrid(-steps, steps, 2, n_steps=steps * 2 + 1).to(samples.device) # s s 2
    samidx = metric2indices(samples, res).view(-1, 1, 3) \
             + pad_l(offset).view(1, -1, 3)  # n s*s 3
    samidx = samidx.view(-1, 3).T  # 3 n*s*s
    samidx[1:] = (samidx[1:] + (det_r / res))
    mask1 = torch.logical_and((samidx[1:] >= 0).all(dim=0),
                             (samidx[1:] < (det_r / res * 2)).all(dim=0))

    inds = samidx[:, mask1].long()
    ctr_idx_of_sam = centers_map[inds[0], inds[1], inds[2]]
    mask2 = ctr_idx_of_sam >= 0
    ctr_idx_of_sam = ctr_idx_of_sam[mask2]
    ns = offset.shape[0]**2
    new_samples = torch.tile(samples.unsqueeze(1),
                             (1, ns, 1)).view(-1, 3)  # n*s*s 3
    new_centers = centers[ctr_idx_of_sam]
    dists_sam2ctr = new_samples[mask1][mask2][:, 1:] - new_centers[:, 1:]

    probs_weighted = weighted_mahalanobis_dists(
        reg_evi[ctr_idx_of_sam],
        reg_var[ctr_idx_of_sam],
        dists_sam2ctr.unsqueeze(1),
        var0=var0
    ).squeeze()

    sample_evis = torch.zeros_like(samidx[:2].T)
    mask = mask1.clone()
    mask[mask1] = mask2
    sample_evis[mask] = probs_weighted
    sample_evis = sample_evis.view(-1, ns, 2).sum(dim=1)

    return sample_evis
