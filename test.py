import os
import torch
import torch.nn as nn
import scipy.io
from CSCANet import CSCANet
from metrics import calculate_psnr, calculate_ssim
from load_data import test_data_loader
import matplotlib.pyplot as plt
import glob


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)  # cuda

# 数据的通道数
channel = 305

# 加载训练好的模型
net = CSCANet().to(device=device)

net.load_state_dict(torch.load("./"))  # 加载模型
# 损失函数
loss_func = nn.MSELoss()


# 测试集损失
test_loss = 0.0
psnr_sum = 0.0
ssim_sum = 0.0
test_count = 0
net.eval()  # 切换到评估模式
i=0

with torch.no_grad():  # 不需要计算梯度
    for data in test_data_loader:
        inputs, labels  = data
        inputs, labels = inputs.to(device=device), labels.to(device=device)
        i = i+1
        outputs = net(inputs)  # 进行预测
        test_count += inputs.size(0)

        # # 计算 PSNR 和 SSIM
        psnr = calculate_psnr(outputs, labels)
        ssim = calculate_ssim(outputs, labels)

        # # 累加 PSNR 和 SSIM
        psnr_sum += psnr * inputs.size(0)
        ssim_sum += ssim * inputs.size(0)

        # 保存输出与打印指标
        print(f"Iteration {i}, Loss: {loss.item()}, PSNR: {psnr.item()}, SSIM: {ssim.item()}")
        # outputs_numpy = outputs.squeeze().cpu().numpy()
        # filename = 'outputs_{}.mat'.format(i)
        # filename = filename_list[i-1][0:39]+'dehaze'+filename_list[i-1][43:]
        # scipy.io.savemat(filename, {'outputs': outputs_numpy})

# 计算平均损失、PSNR 和 SSIM
avg_loss = test_loss / test_count
avg_psnr = psnr_sum / test_count
avg_ssim = ssim_sum / test_count
print(f"Average Loss: {avg_loss}, Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}")


