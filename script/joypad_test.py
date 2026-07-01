from ruamel.yaml import YAML
from io import StringIO
from Navigation_RL.bin import rsg_NavRL
from Navigation_RL.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import Navigation_RL.algo.ppo.module as loco_ppo_module
import Navigation_RL.algo.nav_ppo.module as nav_ppo_module
import Navigation_RL.algo.nav_ppo.ppo as NavPPO
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
parser.add_argument('-w', '--weight', help='loco weight path (default: rsc/policy)', type=str, default='')
parser.add_argument('-p', '--policy', help='loco policy iteration', type=int, default=5000)
parser.add_argument('--nav_weight', help='nav weight path (optional)', type=str, default='')
parser.add_argument('--nav_policy', help='nav policy iteration', type=int, default=0)
args = parser.parse_args()

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
nav_act_dim = 3
nav_cmd_history_k = cfg['environment']['nav_cmd_history_k']
nav_cmd_history_features = nav_cmd_history_k * 3

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])

# Load loco policy
loco_weight_path = args.weight if args.weight else home_path + "/rsc/policy"
loco_policy_iter = args.policy
loco_loaded_graph = loco_ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim).to(device)
loco_loaded_graph.load_state_dict(
    torch.load(loco_weight_path + "/full_" + str(loco_policy_iter) + '.pt', map_location=device)['actor_architecture_state_dict'])
env.load_scaling(loco_weight_path, loco_policy_iter)
print(f"Loaded loco policy from {loco_weight_path}/full_{loco_policy_iter}.pt")

# Load nav policy (optional)
nav_policy_loaded = False
if args.nav_weight:
    nav_weight_path = args.nav_weight
    nav_policy_iter = args.nav_policy
    nav_loaded_graph = nav_ppo_module.NavActor(grid_size=20, n_goal_features=4,
                                                n_cmd_history_features=nav_cmd_history_features,
                                                action_dim=nav_act_dim,
                                                fc_shape=cfg['architecture']['nav_policy_net'],
                                                action_scale=1.0).to(device)
    nav_loaded_graph.load_state_dict(
        torch.load(nav_weight_path + "/nav_full_" + str(nav_policy_iter) + '.pt', map_location=device)['actor_architecture_state_dict'])
    nav_policy_loaded = True
    print(f"Loaded nav policy from {nav_weight_path}/nav_full_{nav_policy_iter}.pt")
else:
    print("No nav policy loaded — using joystick for nav commands")

env.nav_reset()
command = [0, 0, 0]
nav_counter = 0
nav_auto = False  # toggle with button 2

print(f"\nControls:")
print(f"  Button 1: map change + nav reset")
print(f"  Button 2: toggle nav auto ({'enabled' if nav_policy_loaded else 'disabled — no nav policy loaded'})")
print(f"  Left stick: vx/vy command")
print(f"  Right stick X: wz command")
print()

for step in range(n_steps * 100000):
    for event in pygame.event.get():
        if event.type == pygame.JOYBUTTONDOWN:
            if event.button == 1:
                env.map_change()
                env.nav_reset()
                for _ in range(10):
                    env.step(np.zeros((1, 12), dtype=np.float32))
                nav_counter = 10
                print("env reset")
            if event.button == 2:
                if nav_policy_loaded:
                    nav_auto = not nav_auto
                    print(f"nav auto: {nav_auto}")
                else:
                    print("nav auto: disabled — no nav policy loaded")

    # Locomotion policy (every step, always runs first for settling)
    obs = env.observe(False)
    network_action = loco_loaded_graph.architecture(torch.from_numpy(obs).to(device))
    env.step(network_action.cpu().detach().numpy())

    # Navigation policy (every 10 steps)
    if nav_counter % 10 == 0:
        nav_obs = env.nav_observe(False)

        # Get nav command from policy or joystick
        if nav_auto and nav_policy_loaded:
            with torch.no_grad():
                nav_action = nav_loaded_graph(torch.from_numpy(nav_obs).to(device))
            command = nav_action.cpu().numpy().flatten().tolist()
        elif len(joysticks) > 0:
            command[0] = - joysticks[0].get_axis(1)  # left stick Y = forward/back
            command[1] = - joysticks[0].get_axis(0)  # left stick X = left/right
            command[2] = - joysticks[0].get_axis(3)  # right stick X = yaw

        # Nav step
        nav_reward, nav_done = env.nav_step(np.float32(command).reshape(1, -1))
        nav_rewards = env.get_nav_rewards()

        # Print nav info
        goal_dist = nav_obs[0, 402]  # goal distance is at index 402
        print(f"[{nav_counter:4d}] cmd=[{command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f}] "
              f"goal_dist={goal_dist:.3f} "
              f"reward=[incr={nav_rewards[0,0]:+.4f}, goal={nav_rewards[0,1]:+.4f}, death={nav_rewards[0,2]:+.4f}, total={nav_rewards[0,3]:+.4f}] "
              f"done={nav_done[0]}")

        if nav_done.any():
            env.nav_reset()
            for _ in range(10):
                env.step(np.zeros((1, 12), dtype=np.float32))
            nav_counter = 10
            print(f"  >>> nav done! reward={nav_reward}")

    nav_counter += 1
    time.sleep(cfg['environment']['control_dt'])
