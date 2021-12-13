"""
测试DIP数据
结论：imu_ori 标定之后的数据与全局姿态类似
      imu_acc 标定之后的数据与全局加速度类似
"""
import os
from utils import matrix_to_axis_angle
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import pickle
import matplotlib.pyplot as plt
from config import paths
import articulate as art
import torch

manhattan_distance = lambda x, y: np.abs(x - y)

np.set_printoptions(threshold = np.inf) 

plt.figure(figsize=(10, 10))
imu_mask = [7, 8, 11, 12, 0, 2]
SMPL_IDX = [18, 19, 4, 5, 15, 0]
VERTEX_IDX =[1962, 5431, 1096, 4583, 412, 3021]
SMPL_SENSOR = ['L_Elbow', 'R_Elbow', 'L_Knee', 'R_Knee', 'Head', 'Pelvis']
filePath = os.path.join(paths.raw_dipimu_dir,"s_02/05.pkl")

data = pickle.load(open(filePath, 'rb'), encoding='latin1')
print(data.keys())

def testOri():
    imu = torch.FloatTensor(data['imu_ori'][:, [7, 8, 11, 12, 0, 2]])
    for _ in range(4):
        imu[1:].masked_scatter_(torch.isnan(imu[1:]), imu[:-1][torch.isnan(imu[1:])])
        imu[:-1].masked_scatter_(torch.isnan(imu[:-1]), imu[1:][torch.isnan(imu[:-1])])

    imu = matrix_to_axis_angle(imu)[70:]
    print("imu", imu.shape)

    # 本地
    pose_local = torch.from_numpy(data['gt']).view(-1, 24, 3)
    pose_local_reduce = pose_local[:-70, SMPL_IDX]
    # 全局
    global_pose = art.ParametricModel(paths.smpl_file).forward_kinematics(art.math.axis_angle_to_rotation_matrix(pose_local).view(-1, 24, 3, 3))[0]
    global_pose = art.math.rotation_matrix_to_axis_angle(global_pose).view(-1, 24, 3)
    global_pose_reduce = global_pose[:-70, SMPL_IDX]
    print("global_pose", global_pose_reduce.shape)
    print("local pose", pose_local_reduce.shape)
    length = 1000
    
    # imu = art.math.axis_angle_to_quaternion(imu).view(-1, 6, 4)
    # pose_local_reduce = art.math.axis_angle_to_quaternion(pose_local_reduce).view(-1, 6, 4)
    # global_pose_reduce = art.math.axis_angle_to_quaternion(global_pose_reduce).view(-1, 6, 4)

    imu = art.math.rotation_matrix_to_r6d(art.math.axis_angle_to_rotation_matrix(imu)).view(-1, 6, 6)
    pose_local_reduce = art.math.rotation_matrix_to_r6d(art.math.axis_angle_to_rotation_matrix(pose_local_reduce)).view(-1, 6, 6)
    global_pose_reduce = art.math.rotation_matrix_to_r6d(art.math.axis_angle_to_rotation_matrix(global_pose_reduce)).view(-1, 6, 6)


    plt.subplots_adjust(wspace =1, hspace =0.5)
    plt.subplots_adjust(wspace =1, hspace =0.5)
    for i in range(6):
        sub1 = plt.subplot(6, 3, i*3+1)
        sub2 = plt.subplot(6, 3, i*3+2)
        sub3 = plt.subplot(6, 3, i*3+3)

        sub1.set_title("imu %s"%(SMPL_SENSOR[i]))
        sub2.set_title("local pose %s"%(SMPL_SENSOR[i]))
        sub3.set_title("global pose %s"%(SMPL_SENSOR[i]))
        
        for ch in range(6):
            num = imu[list(range(length)), i, ch].numpy()
            left= num.mean()-3*num.std()
            right= num.mean()+3*num.std()
            new_num = num[(left<num)&(num<right)]
            sub1.plot(range(len(new_num)), new_num)
            num = pose_local_reduce[list(range(length)), i, ch].numpy()
            left= num.mean()-3*num.std()
            right= num.mean()+3*num.std()
            new_num = num[(left<num)&(num<right)]
            sub2.plot(range(len(new_num)), new_num)
            num = global_pose_reduce[list(range(length)), i, ch].numpy()
            left= num.mean()-3*num.std()
            right= num.mean()+3*num.std()
            new_num = num[(left<num)&(num<right)]
            sub3.plot(range(len(new_num)), new_num)

    plt.show()
import scipy.stats

def cal_error_acc(imu_acc, cal_acc):
    def KL_divergence(p,q):
        return scipy.stats.entropy(p, q)
    p=imu_acc[:, :, 0] * imu_acc[:, :, 0] + imu_acc[:, :, 1]*imu_acc[:, :, 1]+imu_acc[:, :, 2]*imu_acc[:, :, 2] 
    q=cal_acc[:, :, 0] * cal_acc[:, :, 0] + cal_acc[:, :, 1]*cal_acc[:, :, 1]+cal_acc[:, :, 2]*cal_acc[:, :, 2] 
    p = np.sqrt(p.numpy())[:, 2]
    q = np.sqrt(q.numpy())[:, 2]

    x = range(len(p)) 
    # plt.title('KL(P||Q) = %1.3f' % KL_divergence(p, q)) 
    plt.plot(x, p,'-') 
    plt.plot(x, q,'--', c='red')
    plt.show()

    # return KL_divergence(p, q)

def testAcc():
    imu_acc = torch.FloatTensor(data['imu_acc'][:, imu_mask])
    frames2del = np.unique(np.where(np.isnan(imu_acc) == True)[0])
    imu_acc = np.delete(imu_acc, frames2del, 0)
    imu_acc = imu_acc[70:]
    data['gt'] = np.delete(data['gt'], frames2del, 0)
    
     # 本地
    pose_local = torch.from_numpy(data['gt']).view(-1, 24, 3).float()
    # 全局
    vertexs = art.ParametricModel(paths.smpl_file).forward_kinematics(art.math.axis_angle_to_rotation_matrix(pose_local).view(-1, 24, 3, 3), calc_mesh=True)[-1]
    vertexs = vertexs[:, VERTEX_IDX].numpy()
    cal_acc = []
    time_interval = 1.0 / 60
    n = 1 # 3 -0.0146
    for idx in range(n, len(vertexs) - n):
        vertex_0 = vertexs[idx - n]  # 6 * 3
        vertex_1 = vertexs[idx]
        vertex_2 = vertexs[idx + n]
        # 1 加速度合成
        accel_tmp = (vertex_2 + vertex_0 - 2 * vertex_1) / \
            (n * n * time_interval * time_interval)
        cal_acc.append(accel_tmp)

    cal_acc = torch.FloatTensor(cal_acc)
    length = 700

    plt.subplots_adjust(wspace =1, hspace =0.5)
    plt.subplots_adjust(wspace =1, hspace =0.5)
    imu_acc = imu_acc[:-n*2]
    cal_acc = cal_acc[70-n*2:-n*2]

    # print("imu acc", imu_acc.shape)
    # print("cal acc", cal_acc.shape)
    # cal_error_acc(imu_acc[:length], cal_acc[:length])
    for i in range(6):
        sub1 = plt.subplot(6, 2, i*2+1)
        sub2 = plt.subplot(6, 2, i*2+2)

        sub1.set_title("imu acc %s"%(SMPL_SENSOR[i]))
        sub2.set_title("cal acc %s"%(SMPL_SENSOR[i]))
        for ch in range(3):
            num = imu_acc[list(range(length)), i, ch].numpy()
            left= num.mean()-3*num.std()
            right= num.mean()+3*num.std()
            new_num = num[(left<num)&(num<right)]
            sub1.plot(range(len(new_num)), new_num)
            num = cal_acc[list(range(length)), i, ch].numpy()
            left= num.mean()-3*num.std()
            right= num.mean()+3*num.std()
            new_num = num[(left<num)&(num<right)]
            sub2.plot(range(len(new_num)), new_num)

    plt.show()

def testContact():
   
    # 本地
    pose_local = torch.from_numpy(data['gt']).view(-1, 24, 3)
    joints = art.ParametricModel(paths.smpl_file).forward_kinematics(art.math.axis_angle_to_rotation_matrix(pose_local).view(-1, 24, 3, 3))[1]


    stable_threshold = 0.008  
    diff = joints - torch.cat((joints[:1], joints[:-1]))

    stable = (diff[:, [10, 11]].norm(dim=2) < stable_threshold).float().numpy()
    print(stable)



#testContact()
testOri()
# testAcc()