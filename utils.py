r"""
    Utils for the project.
"""


import torch
from config import acc_scale
import articulate as art
import yaml

# def normalize_and_concat(glb_acc, glb_ori):
#     glb_acc = glb_acc.view(-1, 6, 3)
#     glb_ori = glb_ori.view(-1, 6, 3, 3)
#     acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_ori[:, -1]) / acc_scale
#     ori = torch.cat((glb_ori[:, 5:].transpose(2, 3).matmul(glb_ori[:, :5]), glb_ori[:, 5:]), dim=1)
#     data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
#     return data


def normalize_and_concat(glb_acc, glb_ori, num_joints_in=6, isMatrix=True):
    glb_acc = glb_acc.view(-1, num_joints_in, 3)
    glb_ori = glb_ori.view(-1, num_joints_in, 3, 3)
    acc = torch.cat((glb_acc[:, :(num_joints_in-1)] - glb_acc[:, (num_joints_in-1):], glb_acc[:, (num_joints_in-1):]), dim=1).bmm(glb_ori[:, -1]) / acc_scale
    ori = torch.cat((glb_ori[:, (num_joints_in-1):].transpose(2, 3).matmul(glb_ori[:, :(num_joints_in-1)]), glb_ori[:, (num_joints_in-1):]), dim=1)
    if not isMatrix:
        ori = art.math.rotation_matrix_to_r6d(ori).view(-1, num_joints_in, 6)
    data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=-1)
    return data


def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    file.close()
    return dict