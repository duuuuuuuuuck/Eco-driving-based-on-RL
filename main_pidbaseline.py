from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from torch import device
import matplotlib.pyplot as plt
from tqdm import tqdm
from env_baseline import ENV_bsl
import pandas as pd

"""
import time
import psutil
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
import numpy as np
import math
from torch import device
import matplotlib.pyplot as plt
from tqdm import tqdm
from env_baseline import ENV_bsl
"""
test_seed = 12

env = ENV_bsl()
env.reset(seed=1)
x_pid = []
z_pid = []
vx_pid = []
t_pid = []
energy_pid = []
toque_pid = []
episode_steps = 0

episode_reward = 0

lastIndex = 0
state, _ = env.reset(seed =test_seed)
done = False
while not done:
    episode_steps+=1
    next_state, reward, done, _, _= env.step()

    # print(f"p: {p}")
    # print(f"steering_angle: {steering_angle}")
    episode_reward += reward

    state = next_state
    x_pid.append(env.x)
    z_pid.append(env.cz[env.idx])
    vx_pid.append(env.spd)
    t_pid.append(env.time)
    energy_pid.append(env.energy)
    toque_pid.append(env.action)
    
csv_path = "model/baseline/data.csv"
# csv_path = "model/" + MODEL_NAME+'/' + MODEL_NAME + "data.csv"
# 构造 DataFrame

# 构建 DataFrame
df = pd.DataFrame({
    "step": range(len(t_pid)),
    "x": x_pid,
    "z": z_pid,
    "vx": vx_pid,
    "time": t_pid,
    "energy": energy_pid,
    "torque": toque_pid
})

# 保存为 CSV
df.to_csv(csv_path, index=False)

print(f"测试数据已保存至：{csv_path}")


plt.plot(np.array(t_pid),energy_pid,label="RL model path", color="red")
# plt.plot(env.cx,env.cz,label="Ground Rruth", color="black")
# plt.grid()
plt.xlabel('T(s)')
plt.ylabel('energy_pid(kJ)')  
plt.legend()
plt.show()

plt.clf()
plt.plot(np.array(t_pid), vx_pid, color = "green")
plt.xlabel('T(s)')
plt.ylabel('Vx(m·s$^{-1}$)')
plt.grid(True)  
plt.legend()
plt.show()

# 绘制坡度图
# 可视化结果
plt.clf()
plt.figure(figsize=(12, 9))
# 绘制速度图
plt.subplot(3, 1, 1)
plt.plot(x_pid, vx_pid, color = "green")
plt.xlabel('X(m)')
plt.ylabel('Vx(m·s$^{-1}$)')
plt.grid(True)  
plt.legend()
# 绘制力矩图
plt.subplot(3, 1, 2)
plt.plot(x_pid, toque_pid, color = "green")
plt.xlabel('X(m)')
plt.ylabel('Toque(N·m)')
plt.grid(True)  
plt.legend()
#绘制高度图
plt.subplot(3, 1, 3)
plt.plot(x_pid, z_pid, color = "green")
plt.xlabel('X(m)')
plt.ylabel('Z(m)')
plt.grid(True)  
plt.legend()

plt.tight_layout()
plt.show()

plt.clf()
plt.plot(x_pid,z_pid,marker = "o", markersize = 2, label = "The vehicle path", color = "red")
plt.plot(env.cx, env.cz,"--", label = "The real road", color = "blue")
plt.xlabel('X(m)')
plt.ylabel('Z(m)')  
plt.legend()
plt.show()

plt.clf()
plt.figure(figsize=(12, 6))
# 绘制力矩图
plt.subplot(2, 1, 1)
plt.plot(x_pid, toque_pid, color = "green")
plt.xlabel('X(m)')
plt.ylabel('Toque(N·m)')
plt.grid(True)  
plt.legend()
#绘制高度图
plt.subplot(2, 1, 2)
plt.plot(x_pid, z_pid, color = "green")
plt.xlabel('X(m)')
plt.ylabel('Z(m)')
plt.grid(True)  
plt.legend()