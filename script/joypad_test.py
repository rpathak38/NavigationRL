from ruamel.yaml import YAML
from io import StringIO
from Navigation_RL.bin import rsg_NavRL
from Navigation_RL.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import Navigation_RL.algo.ppo.module as ppo_module
import os
import math
import torch.nn as nn
import time
import numpy as np
import torch
import argparse
import pygame

pygame.init()
pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
for joystick in joysticks:
    print(joystick.get_name())

# directories
home_path = os.path.dirname(os.path.realpath(__file__)) + "/.."

# config
yaml = YAML(typ='unsafe', pure=True)
yaml.default_flow_style = False
cfg = yaml.load(open(home_path + "/rsc/cfg.yaml", 'r'))

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
args = parser.parse_args()
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create environment from the configuration file (single env for joypad test)
cfg['environment']['num_envs'] = 1
cfg['environment']['render'] = True
cfg['environment']['num_threads'] = 1
cfg['environment']['curriculum']['terrain_initial_factor'] = 1.0  # max difficulty for testing
stream = StringIO()
yaml.dump(cfg['environment'], stream)
cfg_str = stream.getvalue()
env = VecEnv(rsg_NavRL.RaisimGymEnv(home_path + "/rsc", cfg_str))
env.seed(int(time.time()))

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
nav_ob_dim = env.num_nav_obs

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])

# Load policy
assert weight_path != '', 'weight path error'
policy = 2500
loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim).to(device)
loaded_graph.load_state_dict(
    torch.load(weight_path + "/full_" + str(policy) + '.pt', map_location=device)['actor_architecture_state_dict'])
env.load_scaling(weight_path, policy)


env.nav_reset()
command = [0, 0, 0]
nav_command = [0.0, 0.0, 0.0]
nav_counter = 0
nav_auto = False  # toggle with button 2
for step in range(n_steps * 100000):
    for event in pygame.event.get():
        if event.type == pygame.JOYBUTTONDOWN:
            if event.button == 1:
                env.map_change()
                env.nav_reset()
                print("env reset")
            if event.button == 2:
                nav_auto = not nav_auto
                print(f"nav auto: {nav_auto}")

    # Navigation policy (every 10 steps)
    if nav_counter % 10 == 0:
        nav_obs = env.nav_observe(False)
        # with torch.no_grad():
        #     nav_action = loaded_nav_graph.architecture(torch.from_numpy(nav_obs).to(device))
        # nav_command = nav_action.cpu().detach().numpy().flatten().tolist()
        #
        # if nav_auto:
        #     command = nav_command
        # else:
        if len(joysticks) > 0:
            command[0] = - joysticks[0].get_axis(1)  # left stick Y = forward/back
            command[1] = - joysticks[0].get_axis(0)  # left stick X = left/right
            command[2] = - joysticks[0].get_axis(3)  # right stick X = yaw

    env.set_command(np.float32(command).transpose())

    obs = env.observe(False)
    network_action = loaded_graph.architecture(torch.from_numpy(obs).to(device))
    _, dones = env.step(network_action.cpu().detach().numpy())
    nav_counter += 1
    time.sleep(cfg['environment']['control_dt'])
