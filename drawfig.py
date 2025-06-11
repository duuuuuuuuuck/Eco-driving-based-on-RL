import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from road_generator import generate_road_gradient

# Set the font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})  # Applies to tick labels, legend, etc.
flag = 2

if flag ==1:
    # Algorithms to compare
    algorithms = ['DDPG', 'PPO', 'SAC', 'TD3']
    colors = ['red', 'blue', 'green', 'orange']

    testseed = 1

    # === Final Path Plot ===
    # Mockup: Replace with your real road ground truth path
    env_cx = np.arange(0, 1000, 0.5) 
    env_cz, env_cgrid = generate_road_gradient(seed=testseed)


    # === Plot Training Rewards ===
    plt.figure(figsize=(10, 6))
    for algo, color in zip(algorithms, colors):
        path = f"./model/{algo}_test/episode_rewards.csv"
        df = pd.read_csv(path)
        df['avg_reward_per_step'] = df['reward'] / df['steps'].replace(0, float('nan'))
        plt.plot(df['episode'], df['avg_reward_per_step'], label=algo, color=color)
    plt.xlabel('Episode', fontsize = 24)
    plt.ylabel('Average Reward per Step', fontsize = 24)
    plt.title('Training Progress: Average Reward per Step')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Plot Testing Results ===
    for algo, color in zip(algorithms, colors):
        path = f"./model/{algo}_test/data.csv"
        df = pd.read_csv(path)

        t = df['time']
        energy = df['energy']
        vx_single = df['vx']
        x_location_single = df['x']
        z_location_single = df['z']
        torque = df['torque']

        # Energy over time
        plt.plot(np.array(t), energy, label=f"{algo} path", color=color)
    plt.xlabel('T(s)')
    plt.ylabel('Energy(kJ)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Clear plot
    plt.clf()

    # Vx over time
    for algo, color in zip(algorithms, colors):
        df = pd.read_csv(f"./model/{algo}_test/data.csv")
        plt.plot(np.array(df['time']), df['vx'], label=algo, color=color)
    plt.xlabel('T(s)')
    plt.ylabel('Vx(m·s$^{-1}$)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Create one figure with 3 subplots
    plt.figure(figsize=(12, 10))

    # Subplot 1: Vx vs X
    plt.subplot(3, 1, 1)
    for algo, color in zip(algorithms, colors):
        df = pd.read_csv(f"./model/{algo}_test/data.csv")
        plt.plot(df['x'], df['vx'], label=algo, color=color)
    plt.xlabel('X(m)')
    plt.ylabel('Vx(m·s$^{-1}$)')
    plt.grid(True)
    plt.legend(fontsize = 14)
    plt.title('Velocity Comparison')

    # Subplot 2: Torque vs X
    plt.subplot(3, 1, 2)
    for algo, color in zip(algorithms, colors):
        df = pd.read_csv(f"./model/{algo}_test/data.csv")
        plt.plot(df['x'], df['torque'], label=algo, color=color)
    plt.xlabel('X(m)')
    plt.ylabel('Torque(N·m)')
    plt.grid(True)
    plt.legend(fontsize = 14)
    plt.title('Torque Comparison')

    # Subplot 3: Z vs X (Elevation)
    plt.subplot(3, 1, 3)
    for algo, color in zip(algorithms, colors):
        df = pd.read_csv(f"./model/{algo}_test/data.csv")
        plt.plot(df['x'], df['z'], label=algo, color=color)
    plt.xlabel('X(m)')
    plt.ylabel('Z(m)')
    plt.grid(True)
    plt.legend(fontsize = 14)
    plt.title('Elevation Comparison')

    # Final layout and show
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for algo, color in zip(algorithms, colors):
        df = pd.read_csv(f"./model/{algo}_test/data.csv")
        x_location_single = df['x']
        z_location_single = df['z']
        plt.plot(x_location_single, z_location_single, marker="o", markersize=2, label=f"{algo} path", color=color)
        plt.scatter(x_location_single.iloc[-1], z_location_single.iloc[-1], color=color, s=40, edgecolor='k', zorder=5, label=f"{algo} final point")

        # Fill area below the curve down to y=0
        # plt.fill_between(x_location_single, z_location_single, y2=0, color=color, alpha=0.2)

    # plt.figure(figsize=(10, 6))
    plt.plot(env_cx, env_cz, "--", label="Ground Truth", color="black")
    plt.xlabel('X(m)')
    plt.ylabel('Z(m)')
    plt.xlim(0, 1000)  # Set X-axis limits
    plt.legend(fontsize = 14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if flag == 2:
    # === Plot Testing Results with DDPG and baseline===

    # Algorithms to compare
    algorithms = ['DDPG', 'baseline']
    colors = ['red', 'black']
    for algo, color in zip(algorithms, colors):
        path = f"./model/{algo}/data.csv"
        df = pd.read_csv(path)

        t = df['time']
        energy = df['energy']
        vx_single = df['vx']
        x_location_single = df['x']
        z_location_single = df['z']
        torque = df['torque']

        # Energy over time
        plt.plot(np.array(x_location_single), energy, label=f"{algo} path", color=color)
    plt.xlabel('X(m)')
    plt.ylabel('Energy(kJ)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Clear plot
    plt.clf()

    # Vx over time
    for algo, color in zip(algorithms, colors):
        df = pd.read_csv(f"./model/{algo}/data.csv")
        plt.plot(np.array(df['time']), df['vx'], label=algo, color=color)
    plt.xlabel('T(s)')
    plt.ylabel('Vx(m·s$^{-1}$)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Create one figure with 3 subplots
    plt.figure(figsize=(12, 8))

    # Subplot 1: Vx vs X
    plt.subplot(3, 1, 1)
    for algo, color in zip(algorithms, colors):
        df = pd.read_csv(f"./model/{algo}/data.csv")
        plt.plot(df['x'], df['vx'], label=algo, color=color)
    plt.xlabel('X(m)')
    plt.ylabel('Vx(m·s$^{-1}$)')
    plt.grid(True)
    plt.legend()
    plt.title('Velocity Comparison')

    # Subplot 2: Torque vs X
    plt.subplot(3, 1, 2)
    for algo, color in zip(algorithms, colors):
        df = pd.read_csv(f"./model/{algo}/data.csv")
        plt.plot(df['x'], df['torque'], label=algo, color=color)
    plt.xlabel('X(m)')
    plt.ylabel('Torque(N·m)')
    plt.grid(True)
    plt.legend()
    plt.title('Torque Comparison')

    # Subplot 3: Z vs X (Elevation)
    plt.subplot(3, 1, 3)
    for algo, color in zip(algorithms, colors):
        df = pd.read_csv(f"./model/{algo}/data.csv")
        plt.plot(df['x'], df['z'], label=algo, color=color)
    plt.xlabel('X(m)')
    plt.ylabel('Z(m)')
    plt.grid(True)
    plt.legend()
    plt.title('Elevation Comparison')

    # Final layout and show
    plt.tight_layout()
    plt.show()

    for algo, color in zip(algorithms, colors):
        df = pd.read_csv(f"./model/{algo}/data.csv")
        x_location_single = df['x']
        z_location_single = df['z']
        plt.plot(x_location_single, z_location_single, marker="o", markersize=2, label=f"{algo} path", color=color)

    # plt.plot(env_cx, env_cz, "--", label="Ground Truth", color="black")
    plt.xlabel('X(m)')
    plt.ylabel('Z(m)')
    plt.xlim(0, 1000)  # Set X-axis limits
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
