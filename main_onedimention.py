import torch
from agent import SAC
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
from env_simulink_onedimension import ENV_dyn

MODEL_NAME = "SAC_oned" # This is the name of path that the model would be saved. 
TRAIN = False # When TRAIN == True, the script train the model, else the script test the model.

TRAIN_PREVIOUS_MODEL = True # When TRAIN_PREVIOUS_MODEL == True, the script load and continous training the previous model, else training a new model.


if not os.path.exists(MODEL_NAME): 
    os.mkdir(MODEL_NAME)

n_states = 4
n_actions = 1
action_bounds_0 = [-250, 250] # This is the action bounds for steer angle


MAX_EPISODES = 10000
memory_size = 1e+6
batch_size = 256
gamma = 0.9  # gamma is used to calculate the Q-value
alpha = 0.3 
# alpha, ranging from 0 to 1,  can be considered as the "temperature" in this entorphy-optimizing based RL
# Large alpha means the agent explore more, while small alpha means the agent tends to use the previous policy
lr = 8e-4 # Typically range from 1e-3 to 3e-4
reward_scale = 0.1

num_test_episode = 1

to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024
global_running_reward = 0

WB = 2.91          # WB: The wheelbase of the vehicle, which is the distance between the front and rear axles.                                                 [Pure Pursuit]
rear_dis = 1.46    # rear_dis: The distance between car CG and rear axles.                                                                                     [Pure Pursuit]
dt = 0.1           # dt: The time step used in the simulation, representing the interval at which updates are calculated. # 0.1                                [PID Speed Control]
k = 0.1            # k: This is the look forward gain, which scales the look-ahead distance based on the vehicle's velocity.                                   [Pure Pursuit]
Lfc = 2.5          # Lfc: The look-ahead distance, which determines how far ahead the vehicle should consider when choosing the target point on the path.      [Pure Pursuit]
Kp = 1             # Kp: The speed proportional gain, used in the proportional control law to adjust the acceleration.                                         [PID Speed Control]


def scale_action(x, action_bounds):
    scaled_x = x * (action_bounds[1] - action_bounds[0]) / 2 + (action_bounds[1] + action_bounds[0]) / 2
    return scaled_x


def log(episode, start_time, episode_reward, value_loss, q_loss, policy_loss, memory_length):
    global global_running_reward
    if episode == 0:
        global_running_reward = episode_reward
    else:
        global_running_reward = 0.99 * global_running_reward + 0.01 * episode_reward


    if episode % 100 == 0:
        print(f"EP:{episode}| "
              f"EP_r:{episode_reward:3.3f}| "
              f"EP_running_reward:{global_running_reward:3.3f}| "
              f"Value_Loss:{value_loss:3.3f}| "
              f"Q-Value_Loss:{q_loss:3.3f}| "
              f"Policy_Loss:{policy_loss:3.3f}| "
              f"Memory_length:{memory_length}| "
              f"Duration:{time.time() - start_time:3.3f}| "
              f'Time:{datetime.datetime.now().strftime("%H:%M:%S")}')
        


    with SummaryWriter(MODEL_NAME + "/logs/") as writer:
        writer.add_scalar("Value Loss", value_loss, episode)
        writer.add_scalar("Q-Value Loss", q_loss, episode)
        writer.add_scalar("Policy Loss", policy_loss, episode)
        writer.add_scalar("Episode running reward", global_running_reward, episode)
        writer.add_scalar("Episode reward", episode_reward, episode)


if __name__ == "__main__":
    print(f"Number of states:{n_states}\n"
          f"Number of actions:{n_actions}\n")

    agent = SAC(env_name=MODEL_NAME,
                n_states=n_states,
                n_actions=n_actions,
                memory_size=memory_size,
                batch_size=batch_size,
                gamma=gamma,
                alpha=alpha,
                lr=lr,
                action_bounds=action_bounds_0,
                reward_scale=reward_scale,
                )
    target_speed = 10

    env = ENV_dyn()

if TRAIN:
    if TRAIN_PREVIOUS_MODEL:
        agent.load_weights()
        print("Continue to train the previous model.")
    else:
        print("Train a new model.")
    for episode in tqdm(range(MAX_EPISODES), desc="Training", unit="episode", leave=True):
        state, _ = env.reset(seed = 1)
        episode_reward = 0
        done = 0
        start_time = time.time()
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _= env.step(action)
            agent.store(state, reward, done, action, next_state)
            value_loss, q_loss, policy_loss = agent.train()
            episode_reward += reward
            state = next_state
        if episode % 250 == 0:
            agent.save_weights()
            print("The network has been saved.")
            
        log(episode, start_time, episode_reward, value_loss, q_loss, policy_loss, len(agent.memory))
    agent.save_weights()
    print("The newtwork has been saved.")

else:
    print("Start test.")
    agent.load_weights()
    agent.set_to_eval_mode()
    device = device("cuda" if torch.cuda.is_available() else "cpu")
    state, _ = env.reset(seed = 1)

    x_location_single = []
    z_location_single = []
    vx_single = []
    t = []
    energy = []
    episode_steps = 0

    episode_reward = 0

    lastIndex = 0
    
    done = False
    while not done:
        episode_steps+=1
        action = agent.choose_action(state)
        next_state, reward, done, _, _= env.step(action)

        # print(f"p: {p}")
        # print(f"steering_angle: {steering_angle}")
        episode_reward += reward

        state = next_state
        x_location_single.append(env.x)
        z_location_single.append(env.cz[env.idx])
        vx_single.append(env.spd)
        t.append(env.time)
        energy.append(env.energy)
        

    plt.plot(np.array(t),energy,label="RL model path", color="red")
    # plt.plot(env.cx,env.cz,label="Ground Rruth", color="black")
    # plt.grid()
    plt.xlabel('T(s)')
    plt.ylabel('Energy(kJ)')  
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(np.array(t), vx_single, color = "green")
    plt.xlabel('T(s)')
    plt.ylabel('Vx/m·s$^{-1}$')  
    plt.legend()
    plt.show()
    
    plt.clf()
    plt.plot(np.array(t), x_location_single, color = "green")
    plt.xlabel('T(s)')
    plt.ylabel('Vx/m·s$^{-1}$')  
    plt.legend()
    plt.show()
