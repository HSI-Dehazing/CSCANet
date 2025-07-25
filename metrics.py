import cv2
import numpy as np
import torch
import torch.nn.functional as F
from math import exp
import numpy as np
# from niqe import niqe

def calculate_psnr(labels, outputs):
    assert labels.shape == outputs.shape, "Shapes of labels and outputs must be the same"

    # 转换数据类型为 float
    labels = labels.float()
    outputs = outputs.float()
    # 初始化 SSIM
    psnr_total = 0
    max_val = 1.0  # 假设像素值的范围是 [0, 1]

    # 遍历每个通道
    for i in range(labels.shape[1]):
        mse = torch.mean((labels[:, i, :, :] - outputs[:, i, :, :]) ** 2)
        psnr_channel = 20 * torch.log10(max_val / torch.sqrt(mse))
        psnr_channel = min(psnr_channel, 100)
        psnr_total += psnr_channel

    psnr_avg = psnr_total / labels.shape[1]

    return psnr_avg


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window



def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def calculate_ssim(labels, outputs):
    assert labels.shape == outputs.shape, "Shapes of labels and outputs must be the same"

    # 转换数据类型为 float
    labels = labels.float()
    outputs = outputs.float()

    # 初始化 SSIM
    ssim_total = 0

    # 遍历每个通道
    for i in range(labels.shape[1]):
        # 计算 SSIM
        # ssim_channel = 1 - F.mse_loss(outputs[:, i, :, :], labels[:, i, :, :])
        # ssim_channel = torchvision.metrics.SSIM(data_range = 1.0, win_size = 11, win_sigma = 1.5, k1 = 0.01, k2 = 0.03)
        ssim_channel = ssim(outputs[:, i, :, :], labels[:, i, :, :])
        # 将每个通道的 SSIM 累加
        ssim_total += ssim_channel

    # 对所有通道的 SSIM 求平均
    ssim_avg = ssim_total / labels.shape[1]

    return ssim_avg