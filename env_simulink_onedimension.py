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

from road_generator import generate_road_gradient

my_engine = matlab.engine.start_matlab() # 在python中启动matlab
env_name = 'env_1D' # 文件名
my_engine.load_system(env_name)
dt = 0.1
torque_bounds = [0, 250]
k_bounds = [0,1]
Kp=10.0
Ki=0.2
Kd = 0.2

def scale_action(x, action_bounds):
    scaled_x = x * (action_bounds[1] - action_bounds[0]) / 2 + (action_bounds[1] + action_bounds[0]) / 2
    return scaled_x

class ENV_dyn(gymnasium.Env):

    """

    This class represents the state of the vehicle, including its position (x, y), orientation (yaw), and velocity (v).
    The __init__ method initializes the state and calculates the rear axle position (rear_x, rear_y) based on the given wheelbase (WB).
    The update method is used to update the state based on control inputs (a for acceleration and delta for steering angle).
    The calc_distance method calculates the distance between the rear axle of the vehicle and a given point (point_x, point_y). 

    """

    def __init__(self):
        self.time = 0
        self.spd = 0
        self.energy = 0
        self.cx = np.arange(0, 1000, 0.5) 
        self.len = len(self.cx)    
        self.cz = np.zeros(self.len)
        self.cgrid = np.zeros(self.len)
        self.target_spd = 11.1
        self.x = 0
        self.idx = 0
        self.integral=0
        self.action = np.array((0,0))
        self.acc = 0
        self.prev_energy = 0


        super(ENV_dyn, self).__init__()
        self.action_space = spaces.Box(low=-1, high = 1, shape = (2,),dtype = np.float32)
        # action_space 有两维，第一个是总转矩total，第二个是前轮转矩分配系数k，也就是说，前轮转矩为k*total,后轮转矩为(1-k)*total
        self.observation_space = spaces.Box(low= -500, high = 500, shape = (4,), dtype = np.float64)
        # observation_space 依次为：能耗、车速、当前点的坡度、后一点的坡度
        
        my_engine.set_param(env_name, 'SimulationCommand', 'stop', nargout=0)      # Stop possible simulink process
        simulation_time = 1000 # max simulation time (s) 
        my_engine.set_param(env_name, 'StopTime', str(simulation_time), nargout=0)
        my_engine.set_param(env_name + '/pause_time', 'value', str(dt), nargout=0) # 初始化第一个内部暂停时间为dt(单位为s)
        # my_engine.set_param(env_name, 'StopTime', str(simulation_time), nargout=0) # 设定仿真截止时间
        # Better not set stoptime, or the following training would reach the stop timeline but with no notification raised out, resulting in incorrect environment feedback.
        my_engine.set_param(env_name, 'SimulationCommand', 'start', nargout=0)
        # print("Start simulation.")


    def reset(self,seed = None):
        # 这里reset的seed代码对应路面的变化
        self.target_spd = 11.1
        self.integral = 0

        my_engine.set_param(env_name, 'SimulationCommand', 'stop', nargout=0)      # Stop possible simulink process
        # simulation_time = 1000 # simulation time (s) 
        my_engine.set_param(env_name + '/pause_time', 'value', str(dt), nargout=0) # 初始化第一个内部暂停时间为dt(单位为s)
        # my_engine.set_param(env_name, 'StopTime', str(simulation_time), nargout=0) # 设定仿真截止时间
        # Better not set stoptime, or the following training would reach the stop timeline but with no notification raised out, resulting in incorrect environment feedback.
        my_engine.set_param(env_name, 'SimulationCommand', 'start', nargout=0)
        # print("Restart the environment, seed = " + str(seed))
        self.time = 0
        self.spd = 0
        self.energy = 0
        self.x = 0
        self.idx = 0
        self.integral=0
        self.action = np.array((0,0))
        self.acc = 0
        self.prev_energy = 0

        # self.cz, self.cgrid = generate_road_gradient(seed=seed)
        self.cz, self.cgrid = generate_road_gradient(seed=seed)

        self.stepnum = 0
        self.observation = np.array([self.spd, self.energy, self.cgrid[0], self.cgrid[1]])
        info = {"state":self.observation}
        return self.observation, info
    
    def step(self, action):
        prev_spd = self.spd
        self.prev_energy = self.energy
        dis_spd = self.spd-self.target_spd

        toque_tot = Kp * dis_spd + Ki * self.integral -Kd*self.acc + self.cgrid[self.idx]
        toque_tot = max(min(toque_tot, 250), -250)
        k = scale_action(action, k_bounds)

        # self.time = np.array(my_engine.eval('timenow')).reshape(-1) #获取当前仿真时间
        # self.time = self.time[-1]
        self.time += dt

        my_engine.set_param(env_name+'/K_value', 'value', str(k), nargout=0)  
        my_engine.set_param(env_name+'/Beta', 'value', str(self.cgrid[self.idx]), nargout=0)

        my_engine.set_param(env_name+'/pause_time', 'value', str(self.time), nargout=0)  #在开始仿真前，先设置在当前时间再过dt后暂停仿真
        my_engine.set_param(env_name, 'SimulationCommand', 'continue', nargout=0) #开始仿真，由于上一行代码的限制，dt时间后仿真暂停
        
        self.spd = np.array(my_engine.eval('vehiclespd')).reshape(-1)[-1]
        self.energy = np.array(my_engine.eval('energy')).reshape(-1)[-1]
        self.x = np.array(my_engine.eval('xlocation')).reshape(-1)[-1]

        engine_spd = np.array(my_engine.eval('enginespd')).reshape(-1)[-1]

        self.idx = np.argmin(np.abs(self.cx - self.x))
        torque_tot = np.array(my_engine.eval('simout')).reshape(-1)[-1]

        self.action =np.array((torque_tot, k))

        dis_spd = self.spd-self.target_spd

        self.integral += dis_spd

        self.acc = self.spd - prev_spd

        reward = -1*abs(dis_spd) - (self.energy-self.prev_energy)*0.1
        # reward += self.time*0.1
        # 为了防止模型直接自杀，给模型一个和运行时间正相关的奖励
        # (自杀相对来说是个局部最优的选择？因为这样的话,总损失可能比每一步损失很少但运行很长的步长的总损失更少）

        truncated = False
        done = False

        if abs(dis_spd) > 10:
            reward -= 1000
            truncated = True
            done = True

        '''if engine_spd > 4900:
            truncated = True
            done = True
            '''

        if self.time >= 200 or self.idx >= self.len-2:
            print("Warning: reached the max simulation time!")
            my_engine.set_param(env_name, 'SimulationCommand', 'stop', nargout=0)
            done = True
            self.observation = np.array([self.spd, self.energy, self.cgrid[self.len - 2], self.cgrid[self.len - 1]])
        else:
            self.observation = np.array([self.spd, self.energy, self.cgrid[self.idx], self.cgrid[self.idx+1]])

        info = {} # It is to occupy the value info in order to use the stable baseline3

        return self.observation, reward, done, truncated, info
    
    def render(self):
        return False
    
    def close(self):
        return False