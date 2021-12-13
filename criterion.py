from einops import rearrange
import torch
import config
from config import joint_set, paths, device
import articulate as art
# from utils import global2local

body_model = art.ParametricModel(paths.smpl_file, device=device)
l2lLoss = torch.nn.MSELoss()
bceLoss = torch.nn.BCELoss()

def PoseS1Loss(leaf: torch.Tensor,  leaf_gt: torch.Tensor):
    return l2lLoss(leaf, leaf_gt)
    
def PoseS2Loss(all: torch.Tensor,  all_gt: torch.Tensor):
    return PoseS1Loss(all, all_gt)
def PoseS3Loss(all_6d: torch.Tensor,  all_6d_gt: torch.Tensor):
    return PoseS1Loss(all_6d, all_6d_gt)

def TransB1Loss(contact_prob: torch.Tensor, contact_prob_gt: torch.Tensor):
    contact_prob = torch.clamp(contact_prob,min=1e-4,max=1-1e-4) # 限制概率不会出现nan
    loss = bceLoss(contact_prob, contact_prob_gt)
    #term = - contact_prob_gt * torch.log(contact_prob) - (1 - contact_prob_gt) * torch.log(1 - contact_prob)
    return loss

def _reduced_glb_6d_to_full_local_mat(root_rotation, glb_reduced_pose):
    glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
    global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
    global_full_pose[:,joint_set.reduced] = glb_reduced_pose
    pose = global2local(global_full_pose).view(-1, 24, 3, 3)
    pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
    pose[:, 0] = root_rotation.reshape(-1, 3, 3)
    return pose
    
def JointLoss(r6d:torch.Tensor, joint_gt:torch.Tensor, root_rotation:torch.Tensor):
    """[summary]

    Args:
        r6d (torch.Tensor): [seq, batch_size, 15 *6]
        joint_gt ([type]): [seq, batch_size, 23, 3]
    """
    # print(r6d.shape, joint_gt.shape, root_rotation.shape)
    seq, batch_size, d = r6d.shape
    full_pose = _reduced_glb_6d_to_full_local_mat(root_rotation, r6d).view(seq * batch_size, 24, 3, 3)
    
    _, joint_global = body_model.forward_kinematics(full_pose)

    joint_norm = joint_global - joint_global[:, :1]
    joint_norm = joint_norm[:, config.joint_set.full].view(seq, batch_size, 69)

    return l2lLoss(joint_norm, joint_gt)

def _TransB2Loss(output: torch.Tensor, target: torch.Tensor, n: int):
    arr = torch.split(output - target, n, dim=0)
    diff_sum = torch.stack([torch.norm(item, p=2, dim=-1).sum(dim=0) for item in arr]).mean()
    return diff_sum

def TransB2Loss(output: torch.Tensor, target: torch.Tensor):
    result = _TransB2Loss(output, target, 1) + _TransB2Loss(output, target, 3) + _TransB2Loss(
        output, target, 9) + _TransB2Loss(output, target, 27)
    return result

def foot_accuracy(output: torch.Tensor, target: torch.Tensor):
    output = (output >= 0.5).int()
    target = target.int()
    _acc = ((output == target).sum(dim=-1) == 2).float()
    return _acc.mean().item() * 100