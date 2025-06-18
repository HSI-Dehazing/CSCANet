import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from CSCANet import CSCANet
from load_data import train_data_loader, test_data_loader
import matplotlib.pyplot as plt
import scipy.io
from metrics import calculate_psnr, calculate_ssim
import math
import glob


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)  # cuda

# 数据的通道数
channel = 305 #
epoch_num = 301 #

# 学习率
lr = 0.0001
batch_size = 1 #

net = CSCANet(channel).to(device=device)
net.load_state_dict(torch.load("./"))  # 加载模型
# 损失函数
loss_func = nn.MSELoss() #


optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, [90,120,150,180,210,240,270], gamma=0.5)

cro = 256
cronum = 2
train_losses = []
test_losses = []
test_psnr = []
test_ssim = []
for epoch in range(epoch_num):
    net.train()
    tamp = 0
    j = 0
    for i, data in enumerate(train_data_loader):
	j = j+1
        if epoch == 0  
            print("step ", i)  # 一个epoch 第几个迭代次数(第几个batch)
        inputs, labels = data  # 输入数据和标签
        for flag in range(4):
            inputss = inputs[:,:,cro*(flag%cronum ):cro*((flag%cronum )+1),cro*(flag//cronum ):cro*((flag//cronum )+1)]
            labelss = labels[:, :, cro * (flag % cronum ):cro * ((flag % cronum ) + 1), cro * (flag // cronum ):cro * ((flag // cronum ) + 1)]
            inputss, labelss = inputss.to(device=device), labelss.to(device=device)
            outputs = net(inputss)  # 将输入放进网络，得到预测输出结果
            loss = loss_func(outputs, labelss)  # 预测输出和真实的labels,计算损失
            tamp = tamp + loss.item()
            optimizer.zero_grad()  # 反向传播前，将优化器的梯度置为0，否则梯度会累加
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

    train_losses.append(tamp/j)
    print("epoch", epoch + 1,"train_loss is", tamp/j)  # 打印一个epoch的平均loss
    scheduler_lr.step()

    if not os.path.exists("models"):
        os.mkdir("models")

    if epoch % 5 == 0:
        torch.save(net.state_dict(), r"./models/{}.pth".format(epoch))  # 每10epoch保存一个模型
        print('save successful')

        x = [k for k in range(epoch+1)]
        plt.figure(1)
        plt.plot(x, train_losses)
        plt.savefig('./1.png')

        test_loss = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        test_count = 0
        net.eval()
        ii = 0
        with torch.no_grad():  # 不需要计算梯度
            for data1 in test_data_loader:
                inputs1, labels1 = data1
                for flag1 in range(4):
                    inputss1 = inputs1[:, :, cro * (flag1 % cronum ):cro * ((flag1 % cronum ) + 1),cro * (flag1 // cronum ):cro * ((flag1 // cronum ) + 1)]
                    labelss1 = labels1[:, :, cro * (flag1 % cronum ):cro * ((flag1 % cronum ) + 1),cro * (flag1 // cronum ):cro * ((flag1 // cronum ) + 1)]
                    inputss1, labelss1 = inputss1.to(device=device), labelss1.to(device=device)
                    ii = ii + 1
                    outputs1 = net(inputss1)
                    loss1 = loss_func(outputs1, labelss1)  # 计算损失
                    test_loss += loss1.item() * inputs1.size(0)  # 累加测试集损失
                    test_count += inputs1.size(0)

                    # # 计算 PSNR 和 SSIM
                    psnr = calculate_psnr(outputs1, labelss1)
                    ssim = calculate_ssim(outputs1, labelss1)
                    #
                    # # 累加 PSNR 和 SSIM
                    psnr_sum += psnr * inputs1.size(0)
                    ssim_sum += ssim * inputs1.size(0)

                    # 保存输出
		    # outputs_numpy = outputs.squeeze().cpu().numpy()
                    # filename = 'outputs_{}.mat'.format(i)
                    # filename = filename_list[i-1][0:39]+'dehaze'+filename_list[i-1][43:]
                    # scipy.io.savemat(filename, {'outputs': outputs_numpy})

        # 计算平均损失、PSNR 和 SSIM
        avg_loss = test_loss / test_count
        avg_psnr = psnr_sum.cpu().numpy() / test_count
        avg_ssim = ssim_sum.cpu().numpy() / test_count

        print(f"Average Loss: {avg_loss}, Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}")
        test_losses.append(avg_loss)
        test_psnr.append(avg_psnr)
        test_ssim.append(avg_ssim)

        xx = [kk for kk in range(math.floor(epoch/5) + 1)]
        plt.figure(2)
        plt.plot(xx, test_losses)
        plt.savefig('./2.png')
        plt.figure(3)
        plt.plot(xx, test_psnr)
        plt.savefig('./3.png')
        plt.figure(4)
        plt.plot(xx, test_ssim)
        plt.savefig('./4.png')
###




