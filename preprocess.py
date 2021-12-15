r"""
    Preprocess DIP-IMU and TotalCapture test dataset.

    Please refer to the `paths` in `config.py` and set the path of each dataset correctly.
"""


import torch
import os
import pickle
from config import paths, joint_set
from glob import glob
import numpy as np
from tqdm import tqdm
import articulate as art

def get_ori_acc(poses_global_rotation, vertexs, frame_rate, n):
    orientation = []  # 旋转矩阵
    acceleration = []  # 加速度
    ori = poses_global_rotation[:, joint_set.sensor][n:-n].cpu().numpy()
    vertexs = vertexs.cpu().numpy()
    time_interval = 1.0 / frame_rate
    total_number = len(poses_global_rotation)
    for idx in range(n, total_number - n):
        vertex_0 = vertexs[idx - n]  # 6 * 3
        vertex_1 = vertexs[idx]
        vertex_2 = vertexs[idx + n]
        # 1 加速度合成
        accel_tmp = (vertex_2 + vertex_0 - 2 * vertex_1) / \
            (n * n * time_interval * time_interval)
        acceleration.append(accel_tmp)

    acc = np.array(acceleration)
    
    return torch.from_numpy(ori), torch.from_numpy(acc)

def compute_imu_data(body_model, poses, trans, device):
    poses = torch.from_numpy(poses).to(device)
    if not trans is None:
        trans = torch.from_numpy(trans).to(device)

    poses = art.math.axis_angle_to_rotation_matrix(poses).view(-1, 24, 3, 3)
    pose_global, joint_global, vertex_global = body_model.forward_kinematics_batch(poses, tran=trans, calc_mesh=True)
    return pose_global, joint_global, vertex_global
    
def pre_process_amass():

    train_split = ["BioMotionLab_NTroje", "BMLhandball", "BMLmovi", "CMU", "MPI_mosh", "DanceDB", "Eyes_Japan_Dataset", "MPI_HDM05", "KIT"]
    test_split = ["ACCAD", "DFaust_67", "SFU", "EKUT", "HumanEva", "SSM_synced", "MPI_Limits"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for subject in tqdm(train_split + test_split):
        for path in tqdm(glob(os.path.join(paths.raw_amass_dir, subject, "**/*.npz"), recursive=True)):
            dirs, filename = os.path.split(path)
            dirs = dirs.replace("dataset_raw", "dataset_work")
            if filename == "shape.npz": continue # 过滤shape文件
            new_file = os.path.join(dirs, filename)
            #print(new_file)
            if(os.path.isfile(new_file)):continue

            data = np.load(path) # ['poses', 'gender', 'mocap_framerate', 'betas', 'trans']

            body_model = art.model.ParametricModel(paths.male_smpl_file \
                if data['gender'] == 'male' else paths.female_smpl_file, device=device) # 根据性别选择模型
            
            mocap_framerate = int(data['mocap_framerate'])
            mocap_framerate = 60 if mocap_framerate == 59 else mocap_framerate
            if mocap_framerate not in [60, 120]: # 只保留60，和120fps的数据
                continue
            
            n = 4 # 前后4帧计算加速度
            pose = data['poses'][::mocap_framerate//60, :24 * 3].astype(np.float32).reshape(-1, 24, 3) # 降采样
            tran = data['trans'][::mocap_framerate//60].astype(np.float32)

            pose_global, joint_global, vertex_global = compute_imu_data(body_model, pose, tran, device) # 计算姿态全局旋转，关节全局位置，顶点全局位置

            # 创建work数据集目录

            isExists = os.path.exists(dirs)
            if not isExists: os.makedirs(dirs) 

            # 保存全局旋转，全局关节位置，全局顶点位置到npz
            # ori, acc = get_ori_acc(pose_global, vertex_global, mocap_framerate, n) # 计算合成的旋转和加速度
            # pose = pose[n:-n]
            # tran = tran[n:-n]
            reduce_vertex_global = vertex_global[:, joint_set.VERTEX_IDS]
        
            np.savez(new_file, pose_global=pose_global.cpu().numpy(), tran=tran, \
                    joint_global=joint_global.cpu().numpy(), reduce_vertex_global=reduce_vertex_global.cpu().numpy())
            
    print('Preprocessed AMASS dataset is saved at', paths.amass_dir)

def del_dirty_data():
    match_file = os.path.join(paths.amass_dir, "BioMotionLab_NTroje", "**/*.npz")
    files = list(glob(match_file))
    filter_list = ['treadmill', 'motorcycle', 'walk', 'jog', 'knocking']
    def remove(x):
        for y in filter_list:
            if y in x:return True
        return False
    files = list(filter( remove,files))
    for file in files:
        os.remove(file)
    
def process_amass(seq_len = 300, train=True):
    train_split = ["BioMotionLab_NTroje", "BMLhandball", "BMLmovi", "CMU", "MPI_mosh", "DanceDB", "Eyes_Japan_Dataset", "MPI_HDM05", "KIT"]
    veri_split = ["ACCAD", "DFaust_67", "SFU", "EKUT", "HumanEva", "SSM_synced", "MPI_Limits"]
    accs_arr, oris_arr, poses_arr, trans_arr, jtr_arr = [], [], [], [], []
    
    total_seq_len = 0
    for subject in tqdm(train_split if train else veri_split):
        for path in tqdm(glob(os.path.join(paths.amass_dir, subject, "**/*.npz"), recursive=True)):
            data = np.load(path)
            pose_global = torch.from_numpy(data['pose_global'])
            tran = torch.from_numpy(data['tran'])
            joint_global = torch.from_numpy(data['joint_global'])
            reduce_vertex_global = torch.from_numpy(data['reduce_vertex_global'])

            # 提取15个姿态，并且转为6d旋转
            pose_mtx = torch.einsum("nij,nkjm->nkim", pose_global[:, 0].transpose(1, 2), pose_global)
            pose_6d = art.math.rotation_matrix_to_r6d(pose_mtx.contiguous()).reshape(-1, 24, 6)[:, joint_set.reduced]

            n = 4
            # 生成虚拟旋转和加速度
            ori, acc = get_ori_acc(pose_global, reduce_vertex_global, frame_rate=60, n=n)
            
            pose_6d = pose_6d[n:-n]
            tran = tran[n:-n]
            joint_global = joint_global[n:-n]
            
            # print(pose_6d.shape, tran.shape, joint_global.shape, ori.shape, acc.shape)

            # 分割为batch
            pose_6ds = torch.split(pose_6d, seq_len)
            trans = torch.split(tran, seq_len)
            joint_globals = torch.split(joint_global, seq_len)
            oris = torch.split(ori, seq_len)
            accs = torch.split(acc, seq_len)

            for p, t, j, ori, acc in zip(pose_6ds, trans, joint_globals, oris, accs):
                if len(p) != seq_len: continue
                total_seq_len += seq_len
                accs_arr.append(acc.clone().numpy())
                oris_arr.append(ori.clone().numpy())
                poses_arr.append(p.clone().numpy())
                trans_arr.append(t.clone().numpy())
                jtr_arr.append(j.clone().numpy())
                
    os.makedirs(paths.amass_dir, exist_ok=True)
    np.savez(os.path.join(paths.amass_dir, 'train' if train else "veri"), **{'acc': accs_arr, 'ori': oris_arr, 'pose': poses_arr, 'tran': trans_arr, 'jp':jtr_arr})
    print(total_seq_len // 3600, " Minutes")

def process_dipimu_test():
    imu_mask = [7, 8, 11, 12, 0, 2]
    test_split = ['s_09', 's_10']
    accs, oris, poses, trans = [], [], [], []

    for subject_name in test_split:
        for motion_name in os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name)):
            path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()

            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                trans.append(torch.zeros(pose.shape[0], 3))  # dip-imu does not contain translations
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    os.makedirs(paths.dipimu_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans}, os.path.join(paths.dipimu_dir, 'test.pt'))
    print('Preprocessed DIP-IMU dataset is saved at', paths.dipimu_dir)


def process_totalcapture():
    inches_to_meters = 0.0254
    file_name = 'gt_skel_gbl_pos.txt'

    accs, oris, poses, trans = [], [], [], []
    for file in sorted(os.listdir(paths.raw_totalcapture_dip_dir)):
        data = pickle.load(open(os.path.join(paths.raw_totalcapture_dip_dir, file), 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        acc = torch.from_numpy(data['acc']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)

        # acc/ori and gt pose do not match in the dataset
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]

        assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        accs.append(acc)    # N, 6, 3
        oris.append(ori)    # N, 6, 3, 3
        poses.append(pose)  # N, 24, 3

    for subject_name in ['S1', 'S2', 'S3', 'S4', 'S5']:
        for motion_name in sorted(os.listdir(os.path.join(paths.raw_totalcapture_official_dir, subject_name))):
            if subject_name == 'S5' and motion_name == 'acting3':
                continue   # no SMPL poses
            f = open(os.path.join(paths.raw_totalcapture_official_dir, subject_name, motion_name, file_name))
            line = f.readline().split('\t')
            index = torch.tensor([line.index(_) for _ in ['LeftFoot', 'RightFoot', 'Spine']])
            pos = []
            while line:
                line = f.readline()
                pos.append(torch.tensor([[float(_) for _ in p.split(' ')] for p in line.split('\t')[:-1]]))
            pos = torch.stack(pos[:-1])[:, index] * inches_to_meters
            pos[:, :, 0].neg_()
            pos[:, :, 2].neg_()
            trans.append(pos[:, 2] - pos[:1, 2])   # N, 3

    # match trans with poses
    for i in range(len(accs)):
        if accs[i].shape[0] < trans[i].shape[0]:
            trans[i] = trans[i][:accs[i].shape[0]]
        assert trans[i].shape[0] == accs[i].shape[0]

    os.makedirs(paths.totalcapture_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans},
               os.path.join(paths.totalcapture_dir, 'test.pt'))
    print('Preprocessed TotalCapture dataset is saved at', paths.totalcapture_dir)


if __name__ == '__main__':
    # del_dirty_data()
    # pre_process_amass()
    process_amass(train=True)
    # process_dipimu()
    # process_totalcapture()
