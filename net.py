import torch.nn
from torch.nn.functional import relu
from config import *
import articulate as art
from attention.model.attention.SelfAttention import ScaledDotProductAttention
import math
from sru import SRU, SRUCell
import numpy as np
from scipy import signal

class RNN(torch.nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional, dropout=dropout)
        self.linear1 = torch.nn.Linear(n_input, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, h=None):
        x, h = self.rnn(relu(self.linear1(x)), h)
        return self.linear2(x), h

class RNN_SRU(torch.nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=5, bidirectional=True, dropout=0.2):
        super(RNN_SRU, self).__init__()
        self.rnn = SRU(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional, dropout=dropout, layer_norm=True)
        self.linear1 = torch.nn.Linear(n_input, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, h=None):
        x, h = self.rnn(relu(self.linear1(x)), h)
        return self.linear2(x), h


class TransPoseNet(torch.nn.Module):
    r"""
    Whole pipeline for pose and translation estimation.
    """
    def __init__(self, num_past_frame=20, num_future_frame=5, hip_length=None, upper_leg_length=None,
                 lower_leg_length=None, prob_threshold=(0.5, 0.9), gravity_velocity=-0.98, device=torch.device("cuda:0")):
        r"""
        :param num_past_frame: Number of past frames for a biRNN window.
        :param num_future_frame: Number of future frames for a biRNN window.
        :param hip_length: Hip length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param upper_leg_length: Upper leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param lower_leg_length: Lower leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param prob_threshold: The probability threshold used to control the fusion of the two translation branches.
        :param gravity_velocity: The gravity velocity added to the Trans-B1 when the body is not on the ground.
        """
        super().__init__()
        n_imu = 6 * 3 + 6 * 9   # acceleration (vector3) and rotation matrix (matrix3x3) of 6 IMUs
        self.pose_s1 = RNN(n_imu,                         joint_set.n_leaf * 3,       256)
        self.pose_s2 = RNN(joint_set.n_leaf * 3 + n_imu,  joint_set.n_full * 3,       64)
        self.pose_s3 = RNN(joint_set.n_full * 3 + n_imu,  joint_set.n_reduced * 6,    128)
        self.tran_b1 = RNN(joint_set.n_leaf * 3 + n_imu,  2,                          64)
        self.tran_b2 = RNN(joint_set.n_full * 3 + n_imu,  3,                          256,    bidirectional=False)

        # lower body joint
        m = art.ParametricModel(paths.male_smpl_file, device=device)
        j, _ = m.get_zero_pose_joint_and_vertex()
        b = art.math.joint_position_to_bone_vector(j[joint_set.lower_body].unsqueeze(0),
                                                   joint_set.lower_body_parent).squeeze(0)
        bone_orientation, bone_length = art.math.normalize_tensor(b, return_norm=True)
        if hip_length is not None:
            bone_length[1:3] = torch.tensor(hip_length)
        if upper_leg_length is not None:
            bone_length[3:5] = torch.tensor(upper_leg_length)
        if lower_leg_length is not None:
            bone_length[5:7] = torch.tensor(lower_leg_length)
        b = bone_orientation * bone_length
        b[:3] = 0

        # constant
        self.global_to_local_pose = m.inverse_kinematics_R
        self.local_to_global_pose = m.forward_kinematics_R
        self.lower_body_bone = b
        self.num_past_frame = num_past_frame
        self.num_future_frame = num_future_frame
        self.num_total_frame = num_past_frame + num_future_frame + 1
        self.prob_threshold = prob_threshold
        self.gravity_velocity = torch.tensor([0, gravity_velocity, 0])
        self.feet_pos = j[10:12].clone()
        self.floor_y = j[10:12, 1].min().item()

        # variable
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_lfoot_pos, self.last_rfoot_pos = self.feet_pos
        self.last_root_pos = torch.zeros(3)
        self.reset()


    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:,joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose

    def _prob_to_weight(self, p):
        return (p.clamp(self.prob_threshold[0], self.prob_threshold[1]) - self.prob_threshold[0]) / \
               (self.prob_threshold[1] - self.prob_threshold[0])

    def reset(self):
        r"""
        Reset online forward states.
        """
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_lfoot_pos, self.last_rfoot_pos = self.feet_pos
        self.last_root_pos = torch.zeros(3)

    def forward(self, imu, leaf_joint_position_gt, full_joint_position_gt, rnn_state=None):
        leaf_joint_position = self.pose_s1.forward(imu)[0]
        full_joint_position = self.pose_s2.forward(torch.cat((leaf_joint_position_gt, imu), dim=-1))[0]
        global_reduced_pose = self.pose_s3.forward(torch.cat((full_joint_position_gt, imu), dim=-1))[0]
        # contact_probability = self.tran_b1.forward(torch.cat((leaf_joint_position_gt, imu), dim=-1))[0]
        # velocity, rnn_state = self.tran_b2.forward(torch.cat((full_joint_position_gt, imu), dim=-1), rnn_state)
        contact_probability, velocity = None, None
        return leaf_joint_position, full_joint_position, global_reduced_pose, contact_probability, velocity, rnn_state
        # return None, None, global_reduced_pose
    def predict(self, imu):
        leaf_joint_position = self.pose_s1.forward(imu)[0]
        full_joint_position = self.pose_s2.forward(torch.cat((leaf_joint_position, imu), dim=-1))[0]
        global_reduced_pose = self.pose_s3.forward(torch.cat((full_joint_position, imu), dim=-1))[0]
        # contact_probability = self.tran_b1.forward(torch.cat((leaf_joint_position, imu), dim=-1))[0]
        # velocity, rnn_state = self.tran_b2.forward(torch.cat((full_joint_position, imu), dim=-1))
        contact_probability, velocity, rnn_state = None, None, None

        return leaf_joint_position, full_joint_position, global_reduced_pose, contact_probability, velocity, rnn_state
        # return None, None, global_reduced_pose


    @torch.no_grad()
    def forward_offline(self, imu):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and velocity tensor in shape [num_frame, 3].
        """
        _, _, global_reduced_pose, contact_probability, velocity, _ = self.predict(imu)

        # calculate pose (local joint rotation matrices)
        root_rotation = imu[:, 0, -9:].view(-1, 3, 3)

        pose = self._reduced_glb_6d_to_full_local_mat(root_rotation.cpu(), global_reduced_pose.cpu())

        return pose, None
        
class TransPoseNet3Stage(torch.nn.Module):
    r"""
    Whole pipeline for pose and translation estimation.
    """
    def __init__(self, num_past_frame=20, num_future_frame=5, hip_length=None, upper_leg_length=None,
                 lower_leg_length=None, prob_threshold=(0.5, 0.9), gravity_velocity=-0.018):
        r"""
        :param num_past_frame: Number of past frames for a biRNN window.
        :param num_future_frame: Number of future frames for a biRNN window.
        :param hip_length: Hip length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param upper_leg_length: Upper leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param lower_leg_length: Lower leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param prob_threshold: The probability threshold used to control the fusion of the two translation branches.
        :param gravity_velocity: The gravity velocity added to the Trans-B1 when the body is not on the ground.
        """
        super().__init__()
        n_imu = 6 * 3 + 6 * 9   # acceleration (vector3) and rotation matrix (matrix3x3) of 6 IMUs
        self.pose_s1 = RNN(n_imu,                         joint_set.n_leaf * 3,       256)
        self.pose_s2 = RNN(joint_set.n_leaf * 3 + n_imu,  joint_set.n_full * 3,       64)
        self.pose_s3 = RNN(joint_set.n_full * 3 + n_imu,  joint_set.n_reduced * 6,    128)
        self.tran_b1 = RNN(joint_set.n_leaf * 3 + n_imu,  2,                          64)
        self.tran_b2 = RNN(joint_set.n_full * 3 + n_imu,  3,                          256,    bidirectional=False)

        # lower body joint
        m = art.ParametricModel(paths.male_smpl_file)
        j, _ = m.get_zero_pose_joint_and_vertex()
        b = art.math.joint_position_to_bone_vector(j[joint_set.lower_body].unsqueeze(0),
                                                   joint_set.lower_body_parent).squeeze(0)
        bone_orientation, bone_length = art.math.normalize_tensor(b, return_norm=True)
        if hip_length is not None:
            bone_length[1:3] = torch.tensor(hip_length)
        if upper_leg_length is not None:
            bone_length[3:5] = torch.tensor(upper_leg_length)
        if lower_leg_length is not None:
            bone_length[5:7] = torch.tensor(lower_leg_length)
        b = bone_orientation * bone_length
        b[:3] = 0

        # constant
        self.global_to_local_pose = m.inverse_kinematics_R
        self.lower_body_bone = b
        self.num_past_frame = num_past_frame
        self.num_future_frame = num_future_frame
        self.num_total_frame = num_past_frame + num_future_frame + 1
        self.prob_threshold = prob_threshold
        self.gravity_velocity = torch.tensor([0, gravity_velocity, 0])
        self.feet_pos = j[10:12].clone()
        self.floor_y = j[10:12, 1].min().item()

        # variable
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_lfoot_pos, self.last_rfoot_pos = self.feet_pos
        self.last_root_pos = torch.zeros(3)
        self.reset()

        # self.load_state_dict(torch.load(paths.weights_file))
        # self.eval()

    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose

    def _prob_to_weight(self, p):
        return (p.clamp(self.prob_threshold[0], self.prob_threshold[1]) - self.prob_threshold[0]) / \
               (self.prob_threshold[1] - self.prob_threshold[0])

    def reset(self):
        r"""
        Reset online forward states.
        """
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_lfoot_pos, self.last_rfoot_pos = self.feet_pos
        self.last_root_pos = torch.zeros(3)

    def forward(self, imu, rnn_state=None):
        leaf_joint_position = self.pose_s1.forward(imu)[0]
        full_joint_position = self.pose_s2.forward(torch.cat((leaf_joint_position, imu), dim=-1))[0]
        global_reduced_pose = self.pose_s3.forward(torch.cat((full_joint_position, imu), dim=-1))[0]
        contact_probability = self.tran_b1.forward(torch.cat((leaf_joint_position, imu), dim=-1))[0]
        velocity, rnn_state = self.tran_b2.forward(torch.cat((full_joint_position, imu), dim=-1), rnn_state)
        return leaf_joint_position, full_joint_position, global_reduced_pose, contact_probability, velocity, rnn_state
    
    # def forward_my(self, input, rnn_state=None, refine=False):
    #     imu, leaf_jtr, full_jtr = input

    #     if not refine:
    #         imu += torch.normal(mean=imu, std=0.04).to(imu.device)
    #     leaf_joint_position = self.pose_s1.forward(imu)[0]
    #     leaf_joint_position_gt_addG = leaf_jtr
    #     full_joint_position = self.pose_s2.forward(torch.cat((leaf_joint_position_gt_addG, imu), dim=-1))[0]
    #     full_joint_position_gt_addG = full_jtr
    #     global_reduced_pose = self.pose_s3.forward(torch.cat((full_joint_position_gt_addG, imu), dim=-1))[0]
        
    #     if not refine:
    #         x1 = torch.cat((leaf_jtr, imu), dim=-1)
    #         # x1 += torch.normal(mean=x1, std=0.04).to(imu.device)
    #         x2 = torch.cat((full_jtr, imu), dim=-1)
    #         # x2 += torch.normal(mean=x2, std=0.025).to(imu.device)

    #         contact_probability = self.tran_b1.forward(x1)[0]
    #         velocity, rnn_state = self.tran_b2.forward(x2, rnn_state)
    #     else:
    #         contact_probability = None
    #         velocity = None
    #     return leaf_joint_position, full_joint_position, global_reduced_pose, contact_probability, velocity, rnn_state

    def forward_my(self, input, rnn_state=None, refine=False):
        imu, leaf_jtr, full_jtr =  input
        if not refine:
            imu += torch.normal(mean=imu, std=0.04).to(imu.device)
        leaf_joint_position = self.pose_s1.forward(imu)[0]
        if not refine:
            leaf_joint_position_gt_addG = leaf_jtr + torch.normal(mean=0, std=0.04, size= leaf_jtr.shape).to(imu.device)
        else:
            leaf_joint_position_gt_addG = leaf_jtr
        full_joint_position = self.pose_s2.forward(torch.cat((leaf_joint_position_gt_addG, imu), dim=-1))[0]
        if not refine:
            full_joint_position_gt_addG = full_jtr + torch.normal(mean=0, std=0.025, size=full_jtr.shape).to(imu.device)
        else:
            full_joint_position_gt_addG = full_jtr
        global_reduced_pose = self.pose_s3.forward(torch.cat((full_joint_position_gt_addG, imu), dim=-1))[0]
        
        if not refine:
            x1 = torch.cat((leaf_jtr, imu), dim=-1)
            x1 += torch.normal(mean=x1, std=0.04).to(imu.device)
            x2 = torch.cat((full_jtr, imu), dim=-1)
            x2 += torch.normal(mean=x2, std=0.025).to(imu.device)

            contact_probability = self.tran_b1.forward(x1)[0]
            velocity, rnn_state = self.tran_b2.forward(x2, rnn_state)
        else:
            contact_probability = None
            velocity = None
        return leaf_joint_position, full_joint_position, global_reduced_pose, contact_probability, velocity, rnn_state

    @torch.no_grad()
    def forward_offline(self, imu):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and velocity tensor in shape [num_frame, 3].
        """
        _, _, global_reduced_pose, contact_probability, velocity, _ = self.forward(imu)

        # calculate pose (local joint rotation matrices)
        root_rotation = imu[:, 0, -9:].view(-1, 3, 3)
        velocity = velocity[:, 0]

        pose = self._reduced_glb_6d_to_full_local_mat(root_rotation.cpu(), global_reduced_pose.cpu())

        # calculate velocity (translation between two adjacent frames in 60fps in world space)
        j = art.math.forward_kinematics(pose[:, joint_set.lower_body],
                                        self.lower_body_bone.expand(pose.shape[0], -1, -1),
                                        joint_set.lower_body_parent)[1]

        tran_b1_vel = self.gravity_velocity + art.math.lerp(
            torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 7] - j[1:, 7])),
            torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 8] - j[1:, 8])),
            contact_probability.max(dim=-1).indices.view(-1, 1).cpu()
        )

        tran_b2_vel = root_rotation.bmm(velocity.unsqueeze(-1)).squeeze(-1).cpu() * vel_scale / 60   # to world space
        weight = self._prob_to_weight(contact_probability.cpu().max(dim=-1).values.sigmoid()).view(-1, 1)
        velocity = art.math.lerp(tran_b2_vel, tran_b1_vel, weight)

        # remove penetration
        current_root_y = 0
        for i in range(velocity.shape[0]):
            current_foot_y = current_root_y + j[i, 7:9, 1].min().item()
            if current_foot_y + velocity[i, 1].item() <= self.floor_y:
                velocity[i, 1] = self.floor_y - current_foot_y
            current_root_y += velocity[i, 1].item()

        return pose, self.velocity_to_root_position(velocity)

    @torch.no_grad()
    def forward_online(self, x):
        r"""
        Online forward.

        :param x: A tensor in shape [input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [24, 3, 3] and velocity tensor in shape [3].
        """
        imu = x.repeat(self.num_total_frame, 1, 1) if self.imu is None else torch.cat((self.imu[1:], x.view(1, 1, -1)))
        _, _, global_reduced_pose, contact_probability, velocity, self.rnn_state = self.forward(imu, self.rnn_state)
        contact_probability = contact_probability[self.num_past_frame].sigmoid().view(-1).cpu()

        # calculate pose (local joint rotation matrices)
        root_rotation = imu[self.num_past_frame, 0, -9:].view(3, 3).cpu()
        global_reduced_pose = global_reduced_pose[self.num_past_frame].cpu()
        pose = self._reduced_glb_6d_to_full_local_mat(root_rotation, global_reduced_pose).squeeze(0)

        # calculate velocity (translation between two adjacent frames in 60fps in world space)
        lfoot_pos, rfoot_pos = art.math.forward_kinematics(pose[joint_set.lower_body].unsqueeze(0),
                                                           self.lower_body_bone.unsqueeze(0),
                                                           joint_set.lower_body_parent)[1][0, 7:9]
        if contact_probability[0] > contact_probability[1]:
            tran_b1_vel = self.last_lfoot_pos - lfoot_pos + self.gravity_velocity
        else:
            tran_b1_vel = self.last_rfoot_pos - rfoot_pos + self.gravity_velocity
        tran_b2_vel = root_rotation.mm(velocity[self.num_past_frame].cpu().view(3, 1)).view(3) / 60 * vel_scale
        weight = self._prob_to_weight(contact_probability.max())
        velocity = art.math.lerp(tran_b2_vel, tran_b1_vel, weight)

        # remove penetration
        current_foot_y = self.current_root_y + min(lfoot_pos[1].item(), rfoot_pos[1].item())
        if current_foot_y + velocity[1].item() <= self.floor_y:
            velocity[1] = self.floor_y - current_foot_y

        self.current_root_y += velocity[1].item()
        self.last_lfoot_pos, self.last_rfoot_pos = lfoot_pos, rfoot_pos
        self.imu = imu
        self.last_root_pos += velocity
        return pose, self.last_root_pos.clone()

    @staticmethod
    def velocity_to_root_position(velocity):
        r"""
        Change velocity to root position. (not optimized)

        :param velocity: Velocity tensor in shape [num_frame, 3].
        :return: Translation tensor in shape [num_frame, 3] for root positions.
        """
        return torch.stack([velocity[:i+1].sum(dim=0) for i in range(velocity.shape[0])])


class TransPoseNet1Stage(torch.nn.Module):
    r"""
    Whole pipeline for pose and translation estimation.
    """
    def __init__(self, num_past_frame=20, num_future_frame=5, hip_length=None, upper_leg_length=None,
                 lower_leg_length=None, prob_threshold=(0.5, 0.9), gravity_velocity=-0.018):
        r"""
        :param num_past_frame: Number of past frames for a biRNN window.
        :param num_future_frame: Number of future frames for a biRNN window.
        :param hip_length: Hip length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param upper_leg_length: Upper leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param lower_leg_length: Lower leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param prob_threshold: The probability threshold used to control the fusion of the two translation branches.
        :param gravity_velocity: The gravity velocity added to the Trans-B1 when the body is not on the ground.
        """
        super().__init__()
        n_imu = 6 * 3 + 6 * 9   # acceleration (vector3) and rotation matrix (matrix3x3) of 6 IMUs
        self.pose_net = RNN(n_imu,                         joint_set.n_reduced * 6,       256)
        # self.tran_b1 = RNN(joint_set.n_leaf * 3 + n_imu,  2,                          64)
        # self.tran_b2 = RNN(joint_set.n_full * 3 + n_imu,  3,                          256,    bidirectional=False)

        # lower body joint
        self.m = art.ParametricModel(paths.male_smpl_file, device=torch.device("cuda:0"))
        j, _ = self.m.get_zero_pose_joint_and_vertex()
        b = art.math.joint_position_to_bone_vector(j[joint_set.lower_body].unsqueeze(0),
                                                   joint_set.lower_body_parent).squeeze(0)
        bone_orientation, bone_length = art.math.normalize_tensor(b, return_norm=True)
        if hip_length is not None:
            bone_length[1:3] = torch.tensor(hip_length)
        if upper_leg_length is not None:
            bone_length[3:5] = torch.tensor(upper_leg_length)
        if lower_leg_length is not None:
            bone_length[5:7] = torch.tensor(lower_leg_length)
        b = bone_orientation * bone_length
        b[:3] = 0

        # constant
        self.global_to_local_pose = self.m.inverse_kinematics_R
        self.lower_body_bone = b
        self.num_past_frame = num_past_frame
        self.num_future_frame = num_future_frame
        self.num_total_frame = num_past_frame + num_future_frame + 1
        self.prob_threshold = prob_threshold
        self.gravity_velocity = torch.tensor([0, gravity_velocity, 0]).cuda()
        self.feet_pos = j[10:12].clone()
        self.floor_y = j[10:12, 1].min().item()

        # variable
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_lfoot_pos, self.last_rfoot_pos = self.feet_pos
        self.last_root_pos = torch.zeros(3)
        self.reset()

        # self.load_state_dict(torch.load(paths.weights_file))
        # self.eval()

    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose

    def _prob_to_weight(self, p):
        return (p.clamp(self.prob_threshold[0], self.prob_threshold[1]) - self.prob_threshold[0]) / \
               (self.prob_threshold[1] - self.prob_threshold[0])

    def reset(self):
        r"""
        Reset online forward states.
        """
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_lfoot_pos, self.last_rfoot_pos = self.feet_pos
        self.last_root_pos = torch.zeros(3)

    def forward(self, imu, rnn_state=None):
        global_reduced_pose = self.pose_net.forward(imu)[0]

        # x1 = torch.cat((leaf_jtr, imu), dim=-1)
        # x2 = torch.cat((full_jtr, imu), dim=-1)

        # contact_probability = self.tran_b1.forward(x1)[0]
        # velocity, rnn_state = self.tran_b2.forward(x2, rnn_state)
        return global_reduced_pose, None, None, rnn_state
    
    # def forward_my(self, input, rnn_state=None, refine=False):
    #     imu, leaf_jtr, full_jtr =  input
    #     if not refine:
    #         global_reduced_pose = self.pose_net.forward(imu + torch.normal(mean=imu, std=0.04).to(imu.device))[0]
    #     else:
    #         global_reduced_pose = self.pose_net.forward(imu)[0]
            
    #     if not refine:
    #         x1 = torch.cat((leaf_jtr, imu), dim=-1)
    #         x1 += torch.normal(mean=x1, std=0.04).to(imu.device)
    #         x2 = torch.cat((full_jtr, imu), dim=-1)
    #         x2 += torch.normal(mean=x2, std=0.025).to(imu.device)

    #         contact_probability = self.tran_b1.forward(x1)[0]
    #         velocity, rnn_state = self.tran_b2.forward(x2, rnn_state)
    #     else:
    #         contact_probability = None
    #         velocity = None
    #     return global_reduced_pose, contact_probability, velocity, rnn_state

    def forward_my(self, input, rnn_state=None, refine=False):
        imu, leaf_jtr, full_jtr =  input
        if not refine: 
            imu += torch.normal(mean=imu, std=0.04).to(imu.device)
        global_reduced_pose = self.pose_net.forward(imu)[0]
        
        # if not refine:
        #     x1 = torch.cat((leaf_jtr, imu), dim=-1)
        #     x1 += torch.normal(mean=x1, std=0.04).to(imu.device)
        #     x2 = torch.cat((full_jtr, imu), dim=-1)
        #     x2 += torch.normal(mean=x2, std=0.025).to(imu.device)

        #     contact_probability = self.tran_b1.forward(x1)[0]
        #     velocity, rnn_state = self.tran_b2.forward(x2, rnn_state)
        # else:
        #     contact_probability = None
        #     velocity = None
        return global_reduced_pose, None, None, rnn_state

    @torch.no_grad()
    def forward_offline(self, imu):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and velocity tensor in shape [num_frame, 3].
        """
        global_reduced_pose, contact_probability, velocity, _ = self.forward(imu)

        # # calculate pose (local joint rotation matrices)
        root_rotation = imu[:, 0, -9:].view(-1, 3, 3)
        # velocity = velocity[:, 0]

        pose = self._reduced_glb_6d_to_full_local_mat(root_rotation, global_reduced_pose)

        # # calculate velocity (translation between two adjacent frames in 60fps in world space)
        # j = art.math.forward_kinematics(pose[:, joint_set.lower_body],
        #                                 self.lower_body_bone.expand(pose.shape[0], -1, -1),
        #                                 joint_set.lower_body_parent)[1]
        # print(j.device, contact_probability.device)
        # tran_b1_vel = self.gravity_velocity + art.math.lerp(
        #     torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 7] - j[1:, 7])),
        #     torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 8] - j[1:, 8])),
        #     contact_probability.max(dim=-1).indices.view(-1, 1)
        # )

        # tran_b2_vel = root_rotation.bmm(velocity.unsqueeze(-1)).squeeze(-1) * vel_scale / 60   # to world space
        # weight = self._prob_to_weight(contact_probability.max(dim=-1).values.sigmoid()).view(-1, 1)
        # velocity = art.math.lerp(tran_b2_vel, tran_b1_vel, weight)

        # # remove penetration
        # current_root_y = 0
        # for i in range(velocity.shape[0]):
        #     current_foot_y = current_root_y + j[i, 7:9, 1].min().item()
        #     if current_foot_y + velocity[i, 1].item() <= self.floor_y:
        #         velocity[i, 1] = self.floor_y - current_foot_y
        #     current_root_y += velocity[i, 1].item()

        return pose, None

    @torch.no_grad()
    def forward_online(self, x):
        r"""
        Online forward.

        :param x: A tensor in shape [input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [24, 3, 3] and velocity tensor in shape [3].
        """
        imu = x.repeat(self.num_total_frame, 1, 1) if self.imu is None else torch.cat((self.imu[1:], x.view(1, 1, -1)))
        _, _, global_reduced_pose, contact_probability, velocity, self.rnn_state = self.forward(imu, self.rnn_state)
        contact_probability = contact_probability[self.num_past_frame].sigmoid().view(-1).cpu()

        # calculate pose (local joint rotation matrices)
        root_rotation = imu[self.num_past_frame, 0, -9:].view(3, 3).cpu()
        global_reduced_pose = global_reduced_pose[self.num_past_frame].cpu()
        pose = self._reduced_glb_6d_to_full_local_mat(root_rotation, global_reduced_pose).squeeze(0)

        # calculate velocity (translation between two adjacent frames in 60fps in world space)
        lfoot_pos, rfoot_pos = art.math.forward_kinematics(pose[joint_set.lower_body].unsqueeze(0),
                                                           self.lower_body_bone.unsqueeze(0),
                                                           joint_set.lower_body_parent)[1][0, 7:9]
        if contact_probability[0] > contact_probability[1]:
            tran_b1_vel = self.last_lfoot_pos - lfoot_pos + self.gravity_velocity
        else:
            tran_b1_vel = self.last_rfoot_pos - rfoot_pos + self.gravity_velocity
        tran_b2_vel = root_rotation.mm(velocity[self.num_past_frame].cpu().view(3, 1)).view(3) / 60 * vel_scale
        weight = self._prob_to_weight(contact_probability.max())
        velocity = art.math.lerp(tran_b2_vel, tran_b1_vel, weight)

        # remove penetration
        current_foot_y = self.current_root_y + min(lfoot_pos[1].item(), rfoot_pos[1].item())
        if current_foot_y + velocity[1].item() <= self.floor_y:
            velocity[1] = self.floor_y - current_foot_y

        self.current_root_y += velocity[1].item()
        self.last_lfoot_pos, self.last_rfoot_pos = lfoot_pos, rfoot_pos
        self.imu = imu
        self.last_root_pos += velocity
        return pose, self.last_root_pos.clone()

    @staticmethod
    def velocity_to_root_position(velocity):
        r"""
        Change velocity to root position. (not optimized)

        :param velocity: Velocity tensor in shape [num_frame, 3].
        :return: Translation tensor in shape [num_frame, 3] for root positions.
        """
        return torch.stack([velocity[:i+1].sum(dim=0) for i in range(velocity.shape[0])])



class AttRNN(torch.nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super(AttRNN, self).__init__()
        self.encoder = torch.nn.LSTM(n_hidden, n_hidden, 1, bidirectional=bidirectional)
        self.decoder = torch.nn.LSTM(n_hidden * (2 if bidirectional else 1), n_hidden, 1, bidirectional=bidirectional)
        self.linear1 = torch.nn.Linear(n_input, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = torch.nn.Dropout(dropout)
        self.attention = ScaledDotProductAttention (d_model=n_hidden * (2 if bidirectional else 1), d_k=n_hidden * (2 if bidirectional else 1), d_v=n_hidden * (2 if bidirectional else 1), h=8)


    def forward(self, x, h=None):
        x = relu(self.linear1(x))
        x, h = self.encoder(x)
        x = x.transpose(0, 1)
        x = self.attention(x, x, x).transpose(0, 1)
        x, h = self.decoder(x)
        return self.linear2(x), h


class AttTransPoseNet1Stage(torch.nn.Module):
    r"""
    Whole pipeline for pose and translation estimation.
    """
    def __init__(self, num_past_frame=20, num_future_frame=5, hip_length=None, upper_leg_length=None,
                 lower_leg_length=None, prob_threshold=(0.5, 0.9), gravity_velocity=-0.018):
        r"""
        :param num_past_frame: Number of past frames for a biRNN window.
        :param num_future_frame: Number of future frames for a biRNN window.
        :param hip_length: Hip length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param upper_leg_length: Upper leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param lower_leg_length: Lower leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param prob_threshold: The probability threshold used to control the fusion of the two translation branches.
        :param gravity_velocity: The gravity velocity added to the Trans-B1 when the body is not on the ground.
        """
        super().__init__()
        n_imu = 6 * 3 + 6 * 9   # acceleration (vector3) and rotation matrix (matrix3x3) of 6 IMUs
        self.pose_net = AttRNN(n_imu,                         joint_set.n_reduced * 6,       256)
        # self.tran_b1 = RNN(joint_set.n_leaf * 3 + n_imu,  2,                          64)
        # self.tran_b2 = RNN(joint_set.n_full * 3 + n_imu,  3,                          256,    bidirectional=False)

        # lower body joint
        self.m = art.ParametricModel(paths.male_smpl_file, device=torch.device("cuda:0"))
        j, _ = self.m.get_zero_pose_joint_and_vertex()
        b = art.math.joint_position_to_bone_vector(j[joint_set.lower_body].unsqueeze(0),
                                                   joint_set.lower_body_parent).squeeze(0)
        bone_orientation, bone_length = art.math.normalize_tensor(b, return_norm=True)
        if hip_length is not None:
            bone_length[1:3] = torch.tensor(hip_length)
        if upper_leg_length is not None:
            bone_length[3:5] = torch.tensor(upper_leg_length)
        if lower_leg_length is not None:
            bone_length[5:7] = torch.tensor(lower_leg_length)
        b = bone_orientation * bone_length
        b[:3] = 0

        # constant
        self.global_to_local_pose = self.m.inverse_kinematics_R
        self.lower_body_bone = b
        self.num_past_frame = num_past_frame
        self.num_future_frame = num_future_frame
        self.num_total_frame = num_past_frame + num_future_frame + 1
        self.prob_threshold = prob_threshold
        self.gravity_velocity = torch.tensor([0, gravity_velocity, 0]).cuda()
        self.feet_pos = j[10:12].clone()
        self.floor_y = j[10:12, 1].min().item()

        # variable
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_lfoot_pos, self.last_rfoot_pos = self.feet_pos
        self.last_root_pos = torch.zeros(3)
        self.reset()

        # self.load_state_dict(torch.load(paths.weights_file))
        # self.eval()

    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose

    def _prob_to_weight(self, p):
        return (p.clamp(self.prob_threshold[0], self.prob_threshold[1]) - self.prob_threshold[0]) / \
               (self.prob_threshold[1] - self.prob_threshold[0])

    def reset(self):
        r"""
        Reset online forward states.
        """
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_lfoot_pos, self.last_rfoot_pos = self.feet_pos
        self.last_root_pos = torch.zeros(3)

    def forward(self, imu, rnn_state=None):
        global_reduced_pose = self.pose_net.forward(imu)[0]

        # x1 = torch.cat((leaf_jtr, imu), dim=-1)
        # x2 = torch.cat((full_jtr, imu), dim=-1)

        # contact_probability = self.tran_b1.forward(x1)[0]
        # velocity, rnn_state = self.tran_b2.forward(x2, rnn_state)
        return global_reduced_pose, None, None, rnn_state
    
    # def forward_my(self, input, rnn_state=None, refine=False):
    #     imu, leaf_jtr, full_jtr =  input
    #     if not refine:
    #         global_reduced_pose = self.pose_net.forward(imu + torch.normal(mean=imu, std=0.04).to(imu.device))[0]
    #     else:
    #         global_reduced_pose = self.pose_net.forward(imu)[0]
            
    #     if not refine:
    #         x1 = torch.cat((leaf_jtr, imu), dim=-1)
    #         x1 += torch.normal(mean=x1, std=0.04).to(imu.device)
    #         x2 = torch.cat((full_jtr, imu), dim=-1)
    #         x2 += torch.normal(mean=x2, std=0.025).to(imu.device)

    #         contact_probability = self.tran_b1.forward(x1)[0]
    #         velocity, rnn_state = self.tran_b2.forward(x2, rnn_state)
    #     else:
    #         contact_probability = None
    #         velocity = None
    #     return global_reduced_pose, contact_probability, velocity, rnn_state

    def forward_my(self, input, rnn_state=None, refine=False):
        imu, leaf_jtr, full_jtr =  input
        if not refine: 
            imu += torch.normal(mean=imu, std=0.04).to(imu.device)
        global_reduced_pose = self.pose_net.forward(imu)[0]
        
        # if not refine:
        #     x1 = torch.cat((leaf_jtr, imu), dim=-1)
        #     x1 += torch.normal(mean=x1, std=0.04).to(imu.device)
        #     x2 = torch.cat((full_jtr, imu), dim=-1)
        #     x2 += torch.normal(mean=x2, std=0.025).to(imu.device)

        #     contact_probability = self.tran_b1.forward(x1)[0]
        #     velocity, rnn_state = self.tran_b2.forward(x2, rnn_state)
        # else:
        #     contact_probability = None
        #     velocity = None
        return global_reduced_pose, None, None, rnn_state

    @torch.no_grad()
    def forward_offline(self, imu):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and velocity tensor in shape [num_frame, 3].
        """
        global_reduced_pose, contact_probability, velocity, _ = self.forward(imu)

        # # calculate pose (local joint rotation matrices)
        root_rotation = imu[:, 0, -9:].view(-1, 3, 3)
        # velocity = velocity[:, 0]

        pose = self._reduced_glb_6d_to_full_local_mat(root_rotation, global_reduced_pose)

        # # calculate velocity (translation between two adjacent frames in 60fps in world space)
        # j = art.math.forward_kinematics(pose[:, joint_set.lower_body],
        #                                 self.lower_body_bone.expand(pose.shape[0], -1, -1),
        #                                 joint_set.lower_body_parent)[1]
        # print(j.device, contact_probability.device)
        # tran_b1_vel = self.gravity_velocity + art.math.lerp(
        #     torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 7] - j[1:, 7])),
        #     torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 8] - j[1:, 8])),
        #     contact_probability.max(dim=-1).indices.view(-1, 1)
        # )

        # tran_b2_vel = root_rotation.bmm(velocity.unsqueeze(-1)).squeeze(-1) * vel_scale / 60   # to world space
        # weight = self._prob_to_weight(contact_probability.max(dim=-1).values.sigmoid()).view(-1, 1)
        # velocity = art.math.lerp(tran_b2_vel, tran_b1_vel, weight)

        # # remove penetration
        # current_root_y = 0
        # for i in range(velocity.shape[0]):
        #     current_foot_y = current_root_y + j[i, 7:9, 1].min().item()
        #     if current_foot_y + velocity[i, 1].item() <= self.floor_y:
        #         velocity[i, 1] = self.floor_y - current_foot_y
        #     current_root_y += velocity[i, 1].item()

        return pose, None

    @torch.no_grad()
    def forward_online(self, x):
        r"""
        Online forward.

        :param x: A tensor in shape [input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [24, 3, 3] and velocity tensor in shape [3].
        """
        imu = x.repeat(self.num_total_frame, 1, 1) if self.imu is None else torch.cat((self.imu[1:], x.view(1, 1, -1)))
        _, _, global_reduced_pose, contact_probability, velocity, self.rnn_state = self.forward(imu, self.rnn_state)
        contact_probability = contact_probability[self.num_past_frame].sigmoid().view(-1).cpu()

        # calculate pose (local joint rotation matrices)
        root_rotation = imu[self.num_past_frame, 0, -9:].view(3, 3).cpu()
        global_reduced_pose = global_reduced_pose[self.num_past_frame].cpu()
        pose = self._reduced_glb_6d_to_full_local_mat(root_rotation, global_reduced_pose).squeeze(0)

        # calculate velocity (translation between two adjacent frames in 60fps in world space)
        lfoot_pos, rfoot_pos = art.math.forward_kinematics(pose[joint_set.lower_body].unsqueeze(0),
                                                           self.lower_body_bone.unsqueeze(0),
                                                           joint_set.lower_body_parent)[1][0, 7:9]
        if contact_probability[0] > contact_probability[1]:
            tran_b1_vel = self.last_lfoot_pos - lfoot_pos + self.gravity_velocity
        else:
            tran_b1_vel = self.last_rfoot_pos - rfoot_pos + self.gravity_velocity
        tran_b2_vel = root_rotation.mm(velocity[self.num_past_frame].cpu().view(3, 1)).view(3) / 60 * vel_scale
        weight = self._prob_to_weight(contact_probability.max())
        velocity = art.math.lerp(tran_b2_vel, tran_b1_vel, weight)

        # remove penetration
        current_foot_y = self.current_root_y + min(lfoot_pos[1].item(), rfoot_pos[1].item())
        if current_foot_y + velocity[1].item() <= self.floor_y:
            velocity[1] = self.floor_y - current_foot_y

        self.current_root_y += velocity[1].item()
        self.last_lfoot_pos, self.last_rfoot_pos = lfoot_pos, rfoot_pos
        self.imu = imu
        self.last_root_pos += velocity
        return pose, self.last_root_pos.clone()

    @staticmethod
    def velocity_to_root_position(velocity):
        r"""
        Change velocity to root position. (not optimized)

        :param velocity: Velocity tensor in shape [num_frame, 3].
        :return: Translation tensor in shape [num_frame, 3] for root positions.
        """
        return torch.stack([velocity[:i+1].sum(dim=0) for i in range(velocity.shape[0])])

from torch import nn

class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()
        
        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'
        
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths
        
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        self.pad = [ filter_widths[0] // 2 ]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out*6, 1)
        

        self.m = art.ParametricModel(paths.male_smpl_file, device=torch.device("cuda:0"))
        
        # constant
        self.global_to_local_pose = self.m.inverse_kinematics_R


    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum
            
    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2*frames
    
    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames
        
    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
        
        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = self._forward_blocks(x)

        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 6)
        
        return x    

class TemporalModel(TemporalModelBase):

    
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        
        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        
        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            
            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))

        x = self.shrink(x)
        return x

    @torch.no_grad()
    def forward_offline(self, imu):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and velocity tensor in shape [num_frame, 3].
        """
        imu = imu.view(1, -1, 6, 12)
        global_reduced_pose = self.forward(imu)
        # # calculate pose (local joint rotation matrices)
        root_rotation = imu[0, 13:-13, -1, -9:].view(-1, 3, 3)
        # velocity = velocity[:, 0]

        pose = self._reduced_glb_6d_to_full_local_mat(root_rotation, global_reduced_pose.contiguous())

        return pose, None

    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self,num_layers=5,dropout=0.1, nhead=6, dim_feedforward=512):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder_src = PositionalEncoding(6*12)
        self.pos_encoder_tgt = PositionalEncoding(15*6)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(6*12,nhead,dim_feedforward,dropout)
        encoder_norm = nn.LayerNorm(6*12)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers,encoder_norm)

        self.inner = nn.Linear(72, 90)
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(15*6,nhead,dim_feedforward,dropout)
        decoder_norm = nn.LayerNorm(15*6)
        self.decoder = nn.TransformerDecoder(decoder_layer,num_layers,decoder_norm)
        self.nhead = nhead

        # lower body joint
        self.m = art.ParametricModel(paths.male_smpl_file, device=torch.device("cuda:0"))
        self.global_to_local_pose = self.m.inverse_kinematics_R

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src, tgt):
        src = self.pos_encoder_src(src)
        tgt = self.pos_encoder_tgt(tgt)
        memory = self.encoder(src)
        memory = self.inner(memory)
        output = self.decoder(tgt, memory)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @torch.no_grad()
    def forward_offline(self, imu):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and velocity tensor in shape [num_frame, 3].
        """
        global_reduced_pose = self.forward(imu)
        # # calculate pose (local joint rotation matrices)
        root_rotation = imu[:, 0, -9:].view(-1, 3, 3)
        # velocity = velocity[:, 0]

        pose = self._reduced_glb_6d_to_full_local_mat(root_rotation, global_reduced_pose.contiguous())
        return pose, None

    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose
        

class TransPoseNet1StageSRU(torch.nn.Module):
    r"""
    Whole pipeline for pose and translation estimation.
    """
    def __init__(self, num_past_frame=20, num_future_frame=5, hip_length=None, upper_leg_length=None,
                 lower_leg_length=None, prob_threshold=(0.5, 0.9), gravity_velocity=-0.018, isMatrix=True):
        r"""
        :param num_past_frame: Number of past frames for a biRNN window.
        :param num_future_frame: Number of future frames for a biRNN window.
        :param hip_length: Hip length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param upper_leg_length: Upper leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param lower_leg_length: Lower leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param prob_threshold: The probability threshold used to control the fusion of the two translation branches.
        :param gravity_velocity: The gravity velocity added to the Trans-B1 when the body is not on the ground.
        """
        super().__init__()
        d = 9 if isMatrix else 6
        n_imu = 6 * 3 + 6 * d   # acceleration (vector3) and rotation matrix (matrix3x3) of 6 IMUs
        self.pose_net = RNN_SRU(n_imu,                         joint_set.n_reduced * 6,       256)

        # lower body joint
        self.m = art.ParametricModel(paths.male_smpl_file, device=torch.device("cuda:0"))
        j, _ = self.m.get_zero_pose_joint_and_vertex()
        b = art.math.joint_position_to_bone_vector(j[joint_set.lower_body].unsqueeze(0),
                                                   joint_set.lower_body_parent).squeeze(0)
        bone_orientation, bone_length = art.math.normalize_tensor(b, return_norm=True)
        if hip_length is not None:
            bone_length[1:3] = torch.tensor(hip_length)
        if upper_leg_length is not None:
            bone_length[3:5] = torch.tensor(upper_leg_length)
        if lower_leg_length is not None:
            bone_length[5:7] = torch.tensor(lower_leg_length)
        b = bone_orientation * bone_length
        b[:3] = 0

        # constant
        self.global_to_local_pose = self.m.inverse_kinematics_R
        self.lower_body_bone = b
        self.num_past_frame = num_past_frame
        self.num_future_frame = num_future_frame
        self.num_total_frame = num_past_frame + num_future_frame + 1
        self.prob_threshold = prob_threshold
        self.gravity_velocity = torch.tensor([0, gravity_velocity, 0]).cuda()
        self.feet_pos = j[10:12].clone()
        self.floor_y = j[10:12, 1].min().item()

        # variable
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_lfoot_pos, self.last_rfoot_pos = self.feet_pos
        self.last_root_pos = torch.zeros(3)
        self.reset()
        
        self.b, self.a = signal.butter(8, 0.2, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
        # self.load_state_dict(torch.load(paths.weights_file))
        # self.eval()

    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose

    def _prob_to_weight(self, p):
        return (p.clamp(self.prob_threshold[0], self.prob_threshold[1]) - self.prob_threshold[0]) / \
               (self.prob_threshold[1] - self.prob_threshold[0])

    def reset(self):
        r"""
        Reset online forward states.
        """
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_lfoot_pos, self.last_rfoot_pos = self.feet_pos
        self.last_root_pos = torch.zeros(3)

    def forward(self, imu, rnn_state=None):
        global_reduced_pose = self.pose_net.forward(imu)[0]

        # x1 = torch.cat((leaf_jtr, imu), dim=-1)
        # x2 = torch.cat((full_jtr, imu), dim=-1)

        # contact_probability = self.tran_b1.forward(x1)[0]
        # velocity, rnn_state = self.tran_b2.forward(x2, rnn_state)
        return global_reduced_pose, None, None, rnn_state
    
    def forward_my(self, input, rnn_state=None, refine=False):
        imu, leaf_jtr, full_jtr =  input
        if not refine: 
            imu += torch.normal(mean=imu, std=0.04).to(imu.device)
        global_reduced_pose = self.pose_net.forward(imu)[0]
        
        return global_reduced_pose, None, None, rnn_state

    @torch.no_grad()
    def forward_offline(self, imu):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and velocity tensor in shape [num_frame, 3].
        """
        global_reduced_pose, contact_probability, velocity, _ = self.forward(imu)

        # # calculate pose (local joint rotation matrices)
        root_rotation = imu[:, 0, -9:].view(-1, 3, 3)
        # velocity = velocity[:, 0]

        pose = self._reduced_glb_6d_to_full_local_mat(root_rotation, global_reduced_pose)

        pose_6d = art.math.rotation_matrix_to_r6d(pose.clone()).reshape(-1, 24, 6).cpu()
        tmp = np.empty(pose_6d.shape)
        for c in range(6):
            for j in range(24):
                tmp[:, j, c] = signal.filtfilt(self.b, self.a, pose_6d[:, j, c].numpy().copy())  #data为要过滤的信号
        tmp = torch.from_numpy(tmp).float()
        pose = art.math.r6d_to_rotation_matrix(tmp).view(-1, 24, 3, 3)
        return pose, None