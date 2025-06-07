import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import time
from torch.distributions.normal import Normal
import math
import matlab.engine
import gymnasium
from gymnasium import spaces
import matplotlib.pyplot as plt

cx = np.arange(0, 1000, 0.5)     

def generate_road_gradient(seed=None):
    """
    生成道路坡度
    :param seed: 随机种子，如果提供则生成固定的随机序列
    :return: (cx, cgrid) 水平距离数组和坡度数组
    """
    # 设置随机种子（如果提供了）
    if seed is not None:
        np.random.seed(seed)
    
    # 生成cx数组
    cx = np.arange(0, 1000, 0.5)
    n_points = len(cx)
    
    # 初始化坡度数组
    cgrid = np.zeros(n_points)
    
    # 设置初始坡度（在-15到15之间随机）
    cgrid[0] = np.random.uniform(-15, 15)
    
    for i in range(1, n_points):
        # 计算前一个坡度值
        prev_gradient = cgrid[i-1]
        
        # 计算下一个坡度的可能范围
        min_grad = max(-20, prev_gradient - 5)
        max_grad = min(20, prev_gradient + 5)
        # 在循环中添加平滑因子
        smoothing_factor = 0.5  # 值越接近1变化越剧烈
        cgrid[i] = smoothing_factor * np.random.uniform(min_grad, max_grad) + (1-smoothing_factor) * prev_gradient

    # 计算步长(相邻cx点的距离)
    step = cx[1] - cx[0]
    
    # 将坡度从度转换为弧度(因为np.tan使用弧度)
    #gradient_rad = np.arctan(cgrid/100)
    
    # 计算高度变化(dz = step * tan(坡度))
    dz = step * cgrid/100
    
    # 通过累加高度变化得到绝对高度
    cz = np.cumsum(dz)
    
    # 将高度基准调整到0开始
    cz = cz - cz[0]
    
    return cz, cgrid

# 使用之前生成的cx和cgrid
cz, cgrid = generate_road_gradient(seed=2)
# cz = generate_road_elevation(cx, cgrid)

def plot(cx, cz, cgrid):
    # 可视化结果
    plt.figure(figsize=(12, 6))


    # 绘制坡度图
    plt.subplot(2, 1, 1)
    plt.plot(cx, cgrid)
    plt.ylabel('grid (deg)')
    plt.grid(True)
    plt.legend()

    # 绘制高度图
    plt.subplot(2, 1, 2)
    plt.plot(cx, cz, color='orange')
    plt.xlabel('x(m)')
    plt.ylabel('z(m)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

