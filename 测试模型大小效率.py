import torch
import time
from net import TransPoseNet3Stage, TransPoseNet1Stage, TransPoseNet, AttTransPoseNet1Stage, TemporalModel, TransAm, TransPoseNet1StageSRU
device = torch.device("cuda:0")
isMatrix = False
net1 = TransPoseNet1StageSRU(isMatrix=isMatrix).to(device)
net1.train()
total = sum([param.nelement() for param in net1.parameters()]) #计算总参数量
print("Number of parameter: %.6f" % (total)) #输出

net2 = TransPoseNet3Stage().to(device)
net2.train()

total = sum([param.nelement() for param in net2.parameters()]) #计算总参数量
print("Number of parameter: %.6f" % (total)) #输出

x = torch.randn(5000, 1, 6*(3+6)).to(device)

s = time.time()
out = net1(x)[0]
print(time.time()-s, out.shape)
x = torch.randn(5000, 1, 6*(3+9)).to(device)

s = time.time()
out = net2.forward(x)[2]
print(time.time()-s, out.shape)
