"""
PointPillars 使用硬/动态体素化
根据 MIT 许可证授权 [见 LICENSE]。
"""

import torch
from torch import nn
import numpy as np
import torch_scatter
import MinkowskiEngine as ME
import spconv
import spconv.pytorch

class PFNLayer(nn.Module):
    """
    Pillar 特征网络层。
    Pillar 特征网络可以由多个此类层组成，但 PointPillars 论文仅使用了一个 PFNLayer。
    此层的作用类似于 second.pytorch.voxelnet.VFELayer。

    :param in_channels: 输入通道数。
    :param out_channels: 输出通道数。
    :param norm_cfg: 归一化配置（此处未使用，仅为兼容性保留）。
    :param last_layer: 如果为 True，则不进行特征拼接。
    """
    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        out_channels = out_channels if self.last_vfe else out_channels // 2
        self.units = out_channels

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.PReLU(num_parameters=out_channels)
        self.alpha = nn.Parameter(torch.tensor(0.0))  # 用于混合池化的可学习参数

    def forward(self, inputs, unq_inv):
        # 线性变换、归一化和激活
        torch.backends.cudnn.benchmark = False
        x = self.linear(inputs)
        x = self.norm(x)
        x = self.relu(x)
        torch.backends.cudnn.benchmark = True
        # 混合池化：结合最大池化和平均池化
        feat_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        feat_mean = torch_scatter.scatter_mean(x, unq_inv, dim=0)
        alpha = torch.sigmoid(self.alpha)  # 确保 alpha 在 (0, 1) 范围内
        feat_hybrid = alpha * feat_max + (1 - alpha) * feat_mean

        if self.last_vfe:
            return feat_hybrid
        return torch.cat([x, feat_hybrid[unq_inv]], dim=1)

class PillarNet(nn.Module):
    """
    将点云动态转为 pillar 表示，并提取 pillar 特征。
    在初始化时预计算并缓存 voxel_size、pc_range 和 grid_size 以提高效率。

    :param num_input_features: 每个点的输入特征数。
    :param voxel_size: 体素大小 (x, y, z)。
    :param pc_range: 点云范围 (min_x, min_y, min_z, max_x, max_y, max_z)。
    """
    def __init__(self, num_input_features, voxel_size, pc_range):
        super().__init__()
        voxel_size_np = np.array(voxel_size, dtype=np.float32)
        pc_range_np = np.array(pc_range, dtype=np.float32)
        grid_size_np = np.round((pc_range_np[3:] - pc_range_np[:3]) / voxel_size_np).astype(np.int64)

        # 缓存常量为缓冲区，避免重复计算
        self.register_buffer('voxel_size', torch.tensor(voxel_size_np))
        self.register_buffer('pc_range', torch.tensor(pc_range_np))
        self.register_buffer('grid_size', torch.tensor(grid_size_np))

    def forward(self, points):
        """
        :param points: 形状为 (N, d) 的张量，格式为 [batch_id, x, y, z, feat1, ...]
        :return: 特征、pillar 坐标、逆索引、网格大小
        """
        with torch.no_grad():  # 关闭梯度计算
            device = points.device
            dtype = points.dtype

            voxel_size = self.voxel_size.to(device=device, dtype=dtype)
            pc_range = self.pc_range.to(device=device, dtype=dtype)
            grid_size = self.grid_size

            # 计算体素坐标并过滤超出范围的点
            points_coords = (points[:, 1:4] - pc_range[:3]) / voxel_size
            mask = (points_coords[:, 0] >= 0) & (points_coords[:, 0] < grid_size[0]) & \
                   (points_coords[:, 1] >= 0) & (points_coords[:, 1] < grid_size[1])
            points = points[mask]
            points_coords = points_coords[mask].long()
            batch_idx = points[:, 0:1].long()

            # 生成 pillar 索引 (batch_id, x_index, y_index)
            points_index = torch.cat([batch_idx, points_coords[:, :2]], dim=1)
            unq, unq_inv = torch.unique(points_index, return_inverse=True, dim=0)
            unq = unq.int()

            # 计算几何特征
            points_mean = torch_scatter.scatter_mean(points[:, 1:4], unq_inv, dim=0)
            f_cluster = points[:, 1:4] - points_mean[unq_inv]
            pillar_center = (points_coords[:, :2].to(dtype) * voxel_size[:2] +
                             voxel_size[:2] / 2 + pc_range[:2])
            f_center = points[:, 1:3] - pillar_center

        # 拼接特征
        features = torch.cat([points[:, 1:], f_cluster, f_center], dim=-1)
        return features, unq[:, [0, 2, 1]], unq_inv, grid_size[[1, 0]]

class PillarFeatureNet(nn.Module):
    """
    Pillar 特征网络，整合 PillarNet 和多个 PFNLayer，生成稀疏张量。

    :param num_input_features: 输入点特征数。
    :param num_filters: 每层 PFNLayer 的输出通道数列表。
    :param voxel_size: 体素大小。
    :param pc_range: 点云范围。
    :param norm_cfg: 归一化配置（未使用，保留兼容性）。
    """
    def __init__(self, num_input_features, num_filters, voxel_size, pc_range, norm_cfg=None):
        super().__init__()
        assert len(num_filters) > 0
        num_input_features += 5  # 增加 5 个几何特征

        # 构建 PFNLayer
        num_filters = [num_input_features] + list(num_filters)
        self.pfn_layers = nn.ModuleList([
            PFNLayer(num_filters[i], num_filters[i + 1], norm_cfg, i == len(num_filters) - 2)
            for i in range(len(num_filters) - 1)
        ])
        self.feature_output_dim = num_filters[-1]
        self.voxelization = PillarNet(num_input_features, voxel_size, pc_range)

        # 预计算网格大小
        self.register_buffer('grid_size', torch.tensor(np.round(
            (np.array(pc_range[3:]) - np.array(pc_range[:3])) / np.array(voxel_size)
        ).astype(np.int64)))

    def forward(self, points):
        # 体素化和初步特征提取
        features, coords, unq_inv, grid_size = self.voxelization(points)

        # 通过 PFNLayer 处理特征
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # 处理最后一个空间位置
        batch_size = coords[:, 0].max().item() + 1
        last_coord = torch.tensor([batch_size - 1] + [dim - 1 for dim in grid_size],
                                 dtype=torch.int32, device=coords.device)
        mask = (coords == last_coord).all(dim=1)

        if mask.any():
            idx = mask.nonzero(as_tuple=True)[0][0]
            if features[idx, -1] == 0:
                features[idx, -1] = 1e-10
        else:
            new_feature = torch.zeros(1, features.shape[1], device=features.device)
            new_feature[0, -1] = 1e-10
            features = torch.cat([features, new_feature], dim=0)
            coords = torch.cat([coords, last_coord.unsqueeze(0)], dim=0)

        # 构造稀疏张量
        return ME.SparseTensor(features=features,coordinates=coords,tensor_stride=1,device=features.device)

        # return spconv.pytorch.SparseConvTensor(features, coords, grid_size, batch_size).dense()