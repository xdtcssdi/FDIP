from torch.utils.data import DataLoader, Dataset
import torch
from config import paths, joint_set
import config
from utils import normalize_and_concat
import os

class OwnDatasets(Dataset):
    def __init__(self, filepath, use_joint=[0, 1, 2, 3, 4, 5]):
        super(OwnDatasets, self).__init__()
        data = torch.load(filepath)
        self.use_joint = use_joint
        self.pose = data['pose']
        self.tran = data['tran']
        self.ori = data['ori']
        self.acc = data['acc']
        self.point = data['jp']

    def __getitem__(self, idx):
        nn_pose = self.pose[idx].float()
        if self.tran[idx] is not None:
            tran = self.tran[idx].float()
        ori = self.ori[idx][:, self.use_joint].float()
        acc = self.acc[idx][:, self.use_joint].float()
        joint_pos = self.point[idx].float()
        root_ori = ori[:, -1] # 最后一组为胯部
        imu = normalize_and_concat(acc, ori)

        # 世界速度->本地速度
        if self.tran[idx] is not None:
            velocity = tran
            velocity_local = root_ori.transpose(1, 2).bmm(
                torch.cat((torch.zeros(1, 3), velocity[1:] - velocity[:-1])).unsqueeze(-1)).squeeze(-1) * 60 / config.vel_scale
        else:
            velocity_local = None
        # 支撑腿
        stable_threshold = 0.008
        diff = joint_pos - torch.cat((joint_pos[:1], joint_pos[:-1]))
        stable = (diff[:, [7, 8]].norm(dim=2) < stable_threshold).float()

        # 关节位置
        nn_jtr = joint_pos - joint_pos[:, :1]
        
        leaf_jtr = nn_jtr[:, joint_set.leaf]
        full_jtr = nn_jtr[:, joint_set.full]
        
        return imu, nn_pose.flatten(1),leaf_jtr.flatten(1), full_jtr.flatten(1), stable, velocity_local, root_ori

    def __len__(self):
        return len(self.ori)

if __name__ == "__main__":
    dataset = OwnDatasets(os.path.join(paths.dipimu_dir, "veri.pt"))
    for imu, nn_pose,leaf_jtr, full_jtr, stable, velocity_local, root_ori in dataset:
        print(imu.shape)
        print(nn_pose.shape)
        print(leaf_jtr.shape)
        print(full_jtr.shape)
        print(stable.shape)
        if velocity_local is not None:
            print(velocity_local.shape)
        print(root_ori.shape)
        break