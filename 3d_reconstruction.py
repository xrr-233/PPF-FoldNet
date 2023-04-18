import argparse

import numpy as np
import torch
import open3d as o3d

from geometric_registration.input_preparation import rgbd_to_point_cloud, cal_local_normal, \
    select_referenced_point, collect_local_neighbor, build_local_patch
from loss.chamfer_loss import ChamferLoss
from models.model_conv1d import PPFFoldNet

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=str, default='analysis-by-synthesis-office2-5a/seq-01/cloud_bin_0', help='relative path')
args = parser.parse_args()

case = args.case
num_patches = 1
num_points_per_patch = 1024
model = PPFFoldNet(1, 1024).cuda()
model.eval()

pretrain = 'pretrained/sun3d_best.pkl'
state_dict = torch.load(pretrain, map_location='cpu')
model.load_state_dict(state_dict)
print(f"Load model from {pretrain}")

root = 'data/3DMatch/rgbd_fragments'
pcd = rgbd_to_point_cloud(root, case)
o3d.visualization.draw_geometries([pcd])
cal_local_normal(pcd)
ref_pcd = select_referenced_point(pcd, num_patches)
neighbor = collect_local_neighbor(ref_pcd, pcd, num_points_per_patch=num_points_per_patch)
pcd_np = np.asarray(pcd.points)
local_patch = build_local_patch(ref_pcd, pcd, neighbor)
patch = torch.tensor(local_patch).cuda()
# output = model(patch)
codeword = model.encoder(patch)
print(codeword.shape)
output = model.decoder(codeword)
print(output.shape)

loss = ChamferLoss()
print(f'Chamfer loss: {loss(patch, output)}')
