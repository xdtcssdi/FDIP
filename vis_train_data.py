import os
import numpy as np
from config import paths
import articulate as art
import torch

def vis_train():
    device = torch.device("cuda:0")
    sample_file = os.path.join(paths.amass_dir, "BioMotionLab_NTroje/rub001/0009_sitting1_poses.npz")
    data = np.load(sample_file)
    pose_global = data['pose_global']
    trans = data['tran']

    pose_global = torch.from_numpy(pose_global).float().to(device)
    tran = torch.from_numpy(trans).to(device)
    print(pose_global.shape, tran.shape)

    body_model = art.model.ParametricModel(paths.male_smpl_file, device=device) # 根据性别选择模型
    pose = body_model.inverse_kinematics_R(pose_global)


    body_model.view_motion([pose], [tran])


def vis_sample():
    device = torch.device("cuda:0")
    data = np.load("sample/163_30.npz")
    pose = torch.from_numpy(data['arr_0']).to(device)
    tran = torch.zeros((len(pose), 3)).to(device)
    body_model = art.model.ParametricModel(paths.male_smpl_file, device=device) # 根据性别选择模型

    body_model.view_motion([pose], [tran])

def vis_sample2():
    device = torch.device("cuda:0")
    data = np.load("sample/sample_valid_data_3150.pkl", allow_pickle=True)
    pose = torch.from_numpy(data['poses']).to(device)
    tran = torch.zeros((len(pose), 3)).to(device)
    body_model = art.model.ParametricModel(paths.male_smpl_file, device=device) # 根据性别选择模型

    body_model.view_motion([pose], [tran])
vis_sample()