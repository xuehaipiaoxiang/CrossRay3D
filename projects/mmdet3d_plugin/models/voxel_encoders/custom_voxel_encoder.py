# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch_scatter
# from mmdet3d.models.builder import VOXEL_ENCODERS


# class PFNLayerV2(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  use_norm=True,
#                  last_layer=False):
#         super().__init__()
        
#         self.last_vfe = last_layer
#         self.use_norm = use_norm
#         if not self.last_vfe:
#             out_channels = out_channels // 2

#         if self.use_norm:
#             self.linear = nn.Linear(in_channels, out_channels, bias=False)
#             self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
#         else:
#             self.linear = nn.Linear(in_channels, out_channels, bias=True)
        
#         self.relu = nn.ReLU()

#     def forward(self, inputs, unq_inv):

#         x = self.linear(inputs)
#         x = self.norm(x) if self.use_norm else x
#         x = self.relu(x)
#         x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

#         if self.last_vfe:
#             return x_max
#         else:
#             x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
#             return x_concatenated


# @VOXEL_ENCODERS.register_module()
# class CustomDynamicPillarVFESimple2D(nn.Module):
#     def __init__(self, num_point_features, voxel_size, grid_size, point_cloud_range,
#                  use_norm, with_distance, use_absolute_xyz, num_filters,
#                   **kwargs):
#         super().__init__()

#         self.use_norm = use_norm
#         self.with_distance = with_distance
#         self.use_absolute_xyz = use_absolute_xyz
#         if self.use_absolute_xyz:
#             num_point_features += 3
#         if self.with_distance:
#             num_point_features += 1

#         self.num_filters = num_filters
#         assert len(self.num_filters) > 0
#         num_filters = [num_point_features] + list(self.num_filters)

#         pfn_layers = []
#         for i in range(len(num_filters) - 1):
#             in_filters = num_filters[i]
#             out_filters = num_filters[i + 1]
#             pfn_layers.append(
#                 PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
#             )
#         self.pfn_layers = nn.ModuleList(pfn_layers)

#         self.voxel_x = voxel_size[0]
#         self.voxel_y = voxel_size[1]
#         self.voxel_z = voxel_size[2]
#         self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
#         self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
#         self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

#         self.scale_xy = grid_size[0] * grid_size[1]
#         self.scale_y = grid_size[1]

#         self.grid_size = torch.tensor(grid_size[:2]).cuda()
#         self.voxel_size = torch.tensor(voxel_size).cuda()
#         self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

#     def get_output_feature_dim(self):
#         return self.num_filters[-1]

#     def forward(self, points, **kwargs):
#         pts_batch = []
#         batch_dict = {}
#         # points = batch_dict['points']  # (batch_idx, x, y, z, i, e)
#         for i, pts in enumerate(points):
#             pts_pad = F.pad(pts, (1, 0), mode='constant', value=i)
#             pts_batch.append(pts_pad)
#         points = torch.cat(pts_batch, dim=0)

#         points_coords = torch.floor(
#             (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / self.voxel_size[[0, 1]]).int()
#         mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]])).all(dim=1)
#         points = points[mask]
#         points_coords = points_coords[mask]
#         points_xyz = points[:, [1, 2, 3]].contiguous()

#         merge_coords = points[:, 0].int() * self.scale_xy + \
#                        points_coords[:, 0] * self.scale_y + \
#                        points_coords[:, 1]

#         unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

#         f_center = torch.zeros_like(points_xyz)
#         f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
#         f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
#         f_center[:, 2] = points_xyz[:, 2] - self.z_offset

#         features = [f_center]
#         if self.use_absolute_xyz:
#             features.append(points[:, 1:])
#         else:
#             features.append(points[:, 4:])



#         if self.with_distance:
#             points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
#             features.append(points_dist)
#         features = torch.cat(features, dim=-1)

#         for pfn in self.pfn_layers:
#             features = pfn(features, unq_inv)

#         unq_coords = unq_coords.int()
#         pillar_coords = torch.stack((unq_coords // self.scale_xy,
#                                      (unq_coords % self.scale_xy) // self.scale_y,
#                                      unq_coords % self.scale_y,
#                                      ), dim=1)
#         #[b, x, y] to [b, y, x]
#         pillar_coords = pillar_coords[:, [0, 2, 1]] 
#         batch_dict['pillar_features'] = features
#         batch_dict['pillar_coords'] = pillar_coords
#         return batch_dict
