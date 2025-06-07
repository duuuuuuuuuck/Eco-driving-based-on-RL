import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import os
import datetime
from torch import device
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from env_simulink import ENV

MODEL_NAME = "DDPG"

TRAIN = False # When TRAIN == True, the script train the model, else the script test the model.

TRAIN_PREVIOUS_MODEL = True # When TRAIN_PREVIOUS_MODEL == True, the script load and continous training the previous model, else training a new model.

test_seed = 1

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
        


    with SummaryWriter("record/" + MODEL_NAME + "/logs/") as writer:
        writer.add_scalar("Value Loss", value_loss, episode)
        writer.add_scalar("Q-Value Loss", q_loss, episode)
        writer.add_scalar("Policy Loss", policy_loss, episode)
        writer.add_scalar("Episode running reward", global_running_reward, episode)
        writer.add_scalar("Episode reward", episode_reward, episode)


env = ENV()

env_id = "road"
n_training_envs = 1
n_eval_envs = 5

# Create log dir where evaluation results will be saved
eval_log_dir = "./eval_logs/" + MODEL_NAME + "/"
os.makedirs(eval_log_dir, exist_ok=True)
'''
# Initialize a vectorized training environment with default parameters
train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0)

# Separate evaluation env, with different parameters passed via env_kwargs
# Eval environments can be vectorized to speed up evaluation.
eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=0,
                        env_kwargs={'g':0.7})

# Create callback that evaluates agent for 5 episodes every 500 training environment steps.
# When using multiple training environments, agent will be evaluated every
# eval_freq calls to train_env.step(), thus it will be evaluated every
# (eval_freq * n_envs) training steps. See EvalCallback doc for more information.
eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
                              log_path=eval_log_dir, eval_freq=max(500 // n_training_envs, 1),
                              n_eval_episodes=5, deterministic=True,
                              render=False)

'''
if TRAIN:
    if TRAIN_PREVIOUS_MODEL:
        print("Continue to train the previous model.")
        model = DDPG.load("model/" + MODEL_NAME+'/' + MODEL_NAME)
        model.load_replay_buffer("model/"+ MODEL_NAME+'/' +MODEL_NAME+"_buffer")

        # 保存模型是包含了环境的，但是自定义环境可能出问题，所以重新设置环境
        model.set_env(env, force_reset=True)

    else: 
        print("Train a new model.")
        model = DDPG("MlpPolicy", env, verbose=1)
    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    model.learn(total_timesteps=10000, log_interval=10, progress_bar=True)
    model.save("model/" + MODEL_NAME+'/' + MODEL_NAME)
    model.save_replay_buffer("model/"+ MODEL_NAME+'/' +MODEL_NAME+"_buffer")
    print("The model has been saved.")
    del model # remove to demonstrate saving and loading



else:
    print("Start test.")
    model = DDPG.load("model/" + MODEL_NAME+'/' + MODEL_NAME)

    done = False
    x_location_single = []
    z_location_single = []
    vx_single = []
    t = []
    energy = []
    episode_steps = 0

    episode_reward = 0

    lastIndex = 0

    state, _ = env.reset(seed = test_seed)
    done = False

    while not done:
        episode_steps+=1
        action, _states = model.predict(state)

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
    plt.ylabel('Vx(m·s$^{-1}$)')
    plt.grid(True)  
    plt.legend()

    # 绘制坡度图
    # 可视化结果
    plt.figure(figsize=(12, 6))
    # 绘制速度图
    plt.subplot(2, 1, 1)
    plt.plot(x_location_single, vx_single, color = "green")
    plt.xlabel('X(m)')
    plt.ylabel('Vx(m·s$^{-1}$)')
    plt.grid(True)  
    plt.legend()
    #绘制高度图
    plt.subplot(2, 1, 2)
    plt.plot(x_location_single, z_location_single, color = "green")
    plt.xlabel('X(m)')
    plt.ylabel('Z(m)')
    plt.grid(True)  
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.clf()
    plt.plot(x_location_single,z_location_single,marker = "o", markersize = 2, label = "The vehicle path", color = "red")
    plt.plot(env.cx, env.cz,"--", label = "The real road", color = "blue")
    plt.xlabel('X(m)')
    plt.ylabel('Z(m)')  
    plt.legend()
    plt.show()
