import torch
import numpy as np
from net import TransPoseNet
from config import paths
from utils import normalize_and_concat
import os
import articulate as art
from preprocess import get_ori_acc

def get_amass_acc_ori():
    data = np.load("data/dataset_work/AMASS/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses.npz")
    ori, acc = get_ori_acc(torch.from_numpy(data['pose_global']), torch.from_numpy(data['reduce_vertex_global']), 60, 4)
    ori = ori[:, :6]
    acc = acc[:, :6]
    return acc, ori

def get_dip_acc_ori():
    data = np.load("data/dataset_work/DIP_IMU/s_04/04.npz")
    return torch.from_numpy(data['acc'][:, :6]), torch.from_numpy(data['ori'][:, :6])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = TransPoseNet().to(device)
checkpoint = torch.load("save_temp/checkpoint_pretrain_50.tar")
net.load_state_dict(checkpoint['state_dict'])

acc, ori = get_amass_acc_ori()

print(ori.shape, acc.shape)
x = normalize_and_concat(acc, ori).to(device)
x = x.unsqueeze(0)
print(x.shape)
pose, tran = net.forward_offline(x)     # offline
tran = torch.zeros_like(tran)
# pose, tran = [torch.stack(_) for _ in zip(*[net.forward_online(f) for f in x])]   # online
art.ParametricModel(paths.male_smpl_file).view_motion([pose], [tran])

