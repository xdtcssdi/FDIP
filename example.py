r"""
    Test the system with an example IMU measurement sequence.
"""


import torch
from net import TransPoseNet, TransPoseNet3Stage
from config import paths
from utils import normalize_and_concat
import os
import articulate as art


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = TransPoseNet3Stage().to(device)
net.eval()
checkpoint = torch.load("save_temp_lstm_dropout/checkpoint_fineturning_327.tar")
net.load_state_dict(checkpoint['state_dict'])
acc = torch.load(os.path.join(paths.example_dir, 'acc.pt'))
ori = torch.load(os.path.join(paths.example_dir, 'ori.pt'))
x = normalize_and_concat(acc, ori).to(device)
x = x.unsqueeze(1)
pose, tran = net.forward_offline(x)     # offline
# pose, tran = [torch.stack(_) for _ in zip(*[net.forward_online(f) for f in x])]   # online
art.ParametricModel(paths.male_smpl_file).view_motion([pose], [tran])
