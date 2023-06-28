import functools
from model.submodules.utils import *
from model.losses.edl import edl_mse_loss


class MinkUnet(nn.Module):
    QMODE = ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
    def __init__(self, cfgs):
        super(MinkUnet, self).__init__()
        for name, value in cfgs.items():
            if name not in ["model", "__class__"]:
                setattr(self, name, value)
        self.d = getattr(self, 'd', 3)
        self.n_cls = cfgs.get('n_cls', 0)

        self.enc_mlp = linear_layers([self.in_dim, 32, 32])

        self.conv1 = minkconv_conv_block(32, 32, 3, 1, self.d, 0.1)
        self.conv2 = self.get_conv_block([32, 32, 32])
        self.conv3 = self.get_conv_block([32, 64, 64])
        self.conv4 = self.get_conv_block([64, 128, 128])

        self.trconv4 = self.get_conv_block([128, 64, 64], tr=True)
        self.trconv3 = self.get_conv_block([128, 64, 64], tr=True)
        self.trconv2 = self.get_conv_block([96, 64, 32], tr=True)

        self.out_layer = minkconv_conv_block(64, 32, 3, 1, 3, 0.1,
                                             'ReLU', norm_before=True)

        if self.n_cls > 0:
            self.cls_layer = linear_last(64, 32, self.n_cls)

            self.semseg_loss = functools.partial(
                edl_mse_loss,
                n_cls=self.n_cls,
                annealing_step=self.annealing_step)

        self.out = {}

    def get_conv_block(self, nc, k=3, tr=False):
        """
        create sparse convolution block
        :param nc: number of channels in each layer in [in_layer, mid_layer, out_layer]
        :param k: kernel size
        :param tr: transposed convolution
        :return: conv block
        """
        kernel = [k,] * self.d
        bnm = 0.1
        assert len(nc) == 3
        return nn.Sequential(
                minkconv_conv_block(nc[0], nc[1], kernel, 2, self.d, bnm, tr=tr),
                minkconv_conv_block(nc[1], nc[1], kernel, 1, self.d, bnm, tr=tr),
                minkconv_conv_block(nc[1], nc[2], kernel, 1, self.d, bnm, tr=tr),
            )

    def forward(self, batch_dict):
        batch_dict = self.prepare_input_data(batch_dict)
        x = batch_dict['in_data']
        x1, norm_points_p1, points_p1, count_p1, pos_embs = self.voxelize_with_centroids(x)

        # convs
        x1 = self.conv1(x1)
        x2 = self.conv2(x1)
        x4 = self.conv3(x2)
        x8 = self.conv4(x4)

        # transposed convs
        p4 = self.trconv4(x8)
        p2 = self.trconv3(ME.cat(x4, p4))
        p1 = self.trconv2(ME.cat(x2, p2))
        p1 = self.out_layer(ME.cat(x1, p1))
        p0 = self.devoxelize_with_centroids(p1, x, pos_embs)

        vars = locals()
        batch_dict['backbone'] = {k: vars[k] for k in self.cache}
        if self.n_cls > 0:
            self.out['evidence'] = self.cls_layer(p0).relu()

    def prepare_input_data(self, batch_dict):
        in_data = ME.TensorField(
            features=batch_dict.pop("features"),
            coordinates=batch_dict.pop("coords"),
            quantization_mode=self.QMODE
        )
        # ME rounds to the floor when casting coords to integer
        batch_dict["in_data"] = in_data
        return batch_dict

    def voxelize_with_centroids(self, x: ME.TensorField):
        cm = x.coordinate_manager
        features = x.F
        coords = x.C[:, 1:]

        out = x.sparse()
        size = torch.Size([len(out), len(x)])
        tensor_map, field_map = cm.field_to_sparse_map(x.coordinate_key, out.coordinate_key)
        coords_p1, count_p1 = downsample_points(coords, tensor_map, field_map, size)
        features_p1, _ = downsample_points(features, tensor_map, field_map, size)
        norm_features = normalize_points(features, features_p1, tensor_map)

        voxel_embs = self.enc_mlp(torch.cat([features, norm_features], dim=1))
        down_voxel_embs = downsample_embeddings(voxel_embs, tensor_map, size, mode="avg")
        out = ME.SparseTensor(down_voxel_embs,
                              coordinate_map_key=out.coordinate_key,
                              coordinate_manager=cm)

        norm_points_p1 = normalize_centroids(coords_p1, out.C, out.tensor_stride[0])
        return out, norm_points_p1, features_p1, count_p1, voxel_embs

    def devoxelize_with_centroids(self, out: ME.SparseTensor, x: ME.TensorField, h_embs):
        feats = torch.cat([out.slice(x).F, h_embs], dim=1)
        return feats

    def loss(self, batch_dict):
        if self.n_cls > 0:
            loss, loss_dict = self.semseg_loss(
                'unet', self.out['evidence'],
                batch_dict['target_semantic'],
                batch_dict['epoch']
            )
            return loss, loss_dict
        else:
            return 0, {}

