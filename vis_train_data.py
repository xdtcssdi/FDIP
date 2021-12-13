import os
import numpy as np
from config import paths
import articulate as art
import torch

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
