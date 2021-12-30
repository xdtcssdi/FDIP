from torch.utils.data import DataLoader, Dataset, random_split
import torch
import numpy as np
import config
from utils import normalize_and_concat
import articulate as art
from einops import rearrange
class ActionDatasets(Dataset):
    def __init__(self, filepath):
        super(ActionDatasets, self).__init__()
        self.data = np.load(filepath, allow_pickle=True)

        self.poses = self.data['poses']
        self.s_foot = self.data['s_foot']
        self.root_velocity = self.data['trans']
        self.ori = self.data['ori']
        self.acc = self.data['acc']
        self.point = self.data['point']

    def __getitem__(self, idx):
        # 根节点位移
        global_velocity_local = torch.from_numpy(self.root_velocity[idx])

        # 脚触地概率
        contact_prob_gt = torch.from_numpy(self.s_foot[idx])

        # 6d旋转
        all_pose_6d_gt = torch.from_numpy(self.poses[idx]).view(-1, 15*6)
        # all_pose_6d_gt = art.math.r6d_to_rotation_matrix(all_pose_6d_gt)
        # all_pose_6d_gt = rearrange(all_pose_6d_gt, "(b e) a d -> b (e a d)", e=15)
        
        # SMPL加速度（全局）
        global_acc = torch.from_numpy(self.acc[idx])
        assert global_acc.shape[1:] == (6, 3)
        # 所有关节位置（全局）
        all_point = torch.from_numpy(self.point[idx])
        assert all_point.shape[1:] == (24, 3)
        
        # 传感器全局方向 叶节点方向 （全局）
        global_ori = torch.from_numpy(self.ori[idx])
        assert global_ori.shape[1:] == (6, 3,3)

        # 根节点方向
        root_ori = global_ori[:, -1].clone()
        assert root_ori.shape[1:] == (3, 3)

        x0 = normalize_and_concat(global_acc, global_ori)
        #assert x0.shape[-1] == 6 * (6)

        # 叶节点位置
        p_leaf_gt = all_point[:, config.joint_set.leaf, :].reshape(-1, 5 * 3)

        # 除胯的节点位置
        p_all_gt = all_point[:, config.joint_set.full].reshape(-1, 69)
        assert p_all_gt.shape[1:] == (69, )
    
        return x0.float(), all_pose_6d_gt.float(), p_leaf_gt.float(), p_all_gt.float(), contact_prob_gt.float(), global_velocity_local.float(), root_ori.float()

    def __len__(self):
        return len(self.ori)

def create_data_loader_split(batch_size, dataset_path):
    train_datasets = ActionDatasets(dataset_path)
    split_rate = 0.9  # 训练集占整个数据集的比例
    train_len = int(split_rate * len(train_datasets))
    valid_len = len(train_datasets) - train_len

    train_sets, valid_sets = random_split(train_datasets, [train_len, valid_len])

    train_loader = DataLoader(train_sets, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False,
                               num_workers=4)
    valid_loader = DataLoader(valid_sets, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False,
                            num_workers=4)

    print(f"训练集大小{len(train_sets)}， 验证集大小{len(valid_sets)}")
    return train_loader, valid_loader

def create_data_loader_all(batch_size, dataset_path):
    train_datasets = ActionDatasets(dataset_path)
    split_rate = 0.8  # 训练集占整个数据集的比例
    train_len = int(split_rate * len(train_datasets))
    valid_len = len(train_datasets) - train_len

    train_sets, valid_sets = random_split(train_datasets, [train_len, valid_len])

    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False,
                               num_workers=4)
    valid_loader = DataLoader(valid_sets, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False,
                            num_workers=4)

    print(f"训练集大小{len(train_sets)}， 验证集大小{len(valid_sets)}")
    return train_loader, valid_loader

if __name__ == "__main__":
    train_loader, test_loader, valid_loader = create_data_loader_all(2)
    data = next(iter(train_loader))
    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)
    print(data[3].shape)
    print(data[4].shape)
    print(data[5].shape)
    print(data[6].shape)

