from ruamel.yaml import YAML
from io import StringIO
from Navigation_RL.bin import rsg_NavRL
from Navigation_RL.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import Navigation_RL.algo.ppo.module as ppo_module
import os
import math
import time
import torch.nn as nn
import numpy as np
import torch
import argparse

# directories
home_path = os.path.dirname(os.path.realpath(__file__)) + "/.."

# config
yaml = YAML(typ='unsafe', pure=True)
yaml.default_flow_style = False
cfg = yaml.load(open(home_path + "/rsc/cfg.yaml", 'r'))

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, required=True)
parser.add_argument('-n', '--num_episodes', help='number of episodes to test', type=int, default=10)
parser.add_argument('-p', '--policy', help='policy checkpoint number', type=int, default=5000)
args = parser.parse_args()
weight_path = args.weight
num_episodes = args.num_episodes
policy_num = args.policy

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create environment from the configuration file
stream = StringIO()
yaml.dump(cfg['environment'], stream)
cfg_str = stream.getvalue()
env = VecEnv(rsg_NavRL.RaisimGymEnv(home_path + "/rsc", cfg_str))

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])

# Load policy
print(f"Loading policy from {weight_path} (checkpoint {policy_num})")
actor_checkpoint = torch.load(weight_path + f'/full_{policy_num}.pt', map_location=device)

module = ppo_module.MLP(cfg['architecture']['policy_net'],
                        nn.LeakyReLU,
                        ob_dim,
                        act_dim)

actor = ppo_module.Actor(module,
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),
                         device=device)

actor.architecture.load_state_dict(actor_checkpoint['actor_architecture_state_dict'])
actor.distribution.load_state_dict(actor_checkpoint['actor_distribution_state_dict'])

if os.path.exists(weight_path + f'/mean{policy_num}.csv'):
    env.load_scaling(weight_path, policy_num)

print("Policy loaded successfully")

# Test
env.turn_on_visualization()
time.sleep(1)

total_rewards = []
total_dones = 0

for episode in range(num_episodes):
    env.map_change()
    env.reset()
    # Set a forward command after reset
    env.set_command(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    episode_reward = 0
    episode_dones = 0
    
    for step in range(n_steps):
        frame_start = time.time()
        obs = env.observe(False)
        action = actor.noiseless_action(obs)
        reward, dones = env.step(action.cpu().detach().numpy())
        episode_reward += np.sum(reward)
        episode_dones += np.sum(dones)
        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)
    
    total_rewards.append(episode_reward / env.num_envs)
    total_dones += episode_dones
    print(f"Episode {episode+1}/{num_episodes}: reward={episode_reward/env.num_envs:.4f}, dones={episode_dones}")

print('----------------------------------------------------')
print(f"Average reward: {np.mean(total_rewards):.4f} +/- {np.std(total_rewards):.4f}")
print(f"Total dones: {total_dones}")
print('----------------------------------------------------')

env.turn_off_visualization()
