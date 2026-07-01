from ruamel.yaml import YAML
from io import StringIO
from Navigation_RL.bin import rsg_NavRL
from Navigation_RL.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from Navigation_RL.helper.raisim_gym_helper import ConfigurationSaver
import Navigation_RL.algo.ppo.module as loco_ppo_module
import Navigation_RL.algo.nav_ppo.module as nav_ppo_module
import Navigation_RL.algo.nav_ppo.ppo as NavPPO
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
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
parser.add_argument('-p', '--policy', help='loco policy iteration', type=int, default=2500)
parser.add_argument('-n', '--num_iters', help='number of training iterations', type=int, default=4000)
parser.add_argument('-v', '--visualize', help='keep visualization on during training', action='store_true')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight
num_iterations = args.num_iters
keep_viz = args.visualize

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
nav_ob_dim = env.num_nav_obs
nav_act_dim = 3
nav_cmd_history_k = cfg['environment']['nav_cmd_history_k']
nav_cmd_history_features = nav_cmd_history_k * 3

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
nav_n_steps = n_steps // 10  # nav policy runs at 1/10 loco frequency
log_intervals = 1

# Load loco policy (frozen)
loco_weight_path = weight_path if weight_path else home_path + "/rsc/policy"
loco_policy_iter = args.policy
loco_loaded_graph = loco_ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim).to(device)
loco_loaded_graph.load_state_dict(
    torch.load(loco_weight_path + "/full_" + str(loco_policy_iter) + '.pt', map_location=device)['actor_architecture_state_dict'])
env.load_scaling(loco_weight_path, loco_policy_iter)
print(f"Loaded loco policy from {loco_weight_path}/full_{loco_policy_iter}.pt")

# Nav policy
init_std = 1.0
lr = 5e-4
entropy_coef = 0.01

nav_actor = nav_ppo_module.Actor(nav_ppo_module.NavActor(grid_size=20, n_goal_features=4,
                                                          n_cmd_history_features=nav_cmd_history_features,
                                                          action_dim=nav_act_dim,
                                                          fc_shape=cfg['architecture']['nav_policy_net'],
                                                          action_scale=1.0),
                                 nav_ppo_module.MultivariateGaussianDiagonalCovariance(nav_act_dim, init_std),
                                 device=device)

nav_critic = nav_ppo_module.Critic(nav_ppo_module.NavCritic(grid_size=20, n_goal_features=4,
                                                             n_cmd_history_features=nav_cmd_history_features,
                                                             fc_shape=cfg['architecture']['nav_value_net']),
                                   device=device)

saver = ConfigurationSaver(log_dir=home_path + "/data/NavRL",
                           save_items=[home_path + "/rsc/cfg.yaml", home_path + "/script/navRL.py"])

nav_ppo = NavPPO.PPO(actor=nav_actor,
                     critic=nav_critic,
                     num_envs=cfg['environment']['num_envs'],
                     num_transitions_per_env=nav_n_steps,
                     num_learning_epochs=10,
                     num_mini_batches=4,
                     clip_param=0.2,
                     gamma=0.99,
                     lam=0.95,
                     entropy_coef=entropy_coef,
                     learning_rate=lr,
                     max_grad_norm=0.5,
                     device=device,
                     log_dir=saver.data_dir,
                     mini_batch_sampling='shuffle',
                     log_intervals=log_intervals)

for update in range(num_iterations + 1):
    start = time.time()
    nav_reward_sum = 0
    incremental_dist_sum = 0
    goal_reaching_sum = 0
    death_penalty_sum = 0
    done_sum = 0
    if not keep_viz:
        env.turn_off_visualization()

    env.nav_reset()
    nav_counter = 0

    # collect nav transitions (loco runs in background)
    for step in range(n_steps):
        # loco step (always first — natural settling)
        obs = env.observe(False)
        loco_action = loco_loaded_graph.architecture(torch.from_numpy(obs).to(device))
        env.step(loco_action.cpu().detach().numpy())

        # nav step (every 10 loco steps)
        if nav_counter % 10 == 0:
            nav_obs = env.nav_observe(False)
            nav_action = nav_ppo.observe(nav_obs)
            nav_reward, nav_done = env.nav_step(nav_action)
            nav_ppo.step(value_obs=nav_obs, rews=nav_reward, dones=nav_done)
            done_sum += sum(nav_done)
            nav_reward_sum += sum(nav_reward)
            
            # accumulate reward components for logging
            nav_rewards = env.get_nav_rewards()  # (num_envs, 4) [incremental, goal, death, total]
            incremental_dist_sum += sum(nav_rewards[:, 0])
            goal_reaching_sum += sum(nav_rewards[:, 1])
            death_penalty_sum += sum(nav_rewards[:, 2])

        nav_counter += 1
        time.sleep(cfg['environment']['control_dt'])
    
    # update nav policy
    last_nav_obs = env.nav_observe(False)
    nav_ppo.update(actor_obs=last_nav_obs,
                   value_obs=last_nav_obs,
                   log_this_iteration=update % log_intervals == 0,
                   update=update,
                   info={
                       'incremental_dist': incremental_dist_sum / env.num_envs,
                       'goal_reaching': goal_reaching_sum / env.num_envs,
                       'death_penalty': death_penalty_sum / env.num_envs,
                       'total_reward': nav_reward_sum / env.num_envs,
                       'dones': done_sum / env.num_envs
                   })

    end = time.time()
    nav_actor.distribution.enforce_minimum_std((torch.ones(nav_act_dim) * 0.1).to(device))
    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("avg nav reward: ", '{:0.10f}'.format(nav_reward_sum / env.num_envs)))
    print('{:<40} {:>6}'.format("incremental dist: ", '{:0.10f}'.format(incremental_dist_sum / env.num_envs)))
    print('{:<40} {:>6}'.format("goal reaching: ", '{:0.10f}'.format(goal_reaching_sum / env.num_envs)))
    print('{:<40} {:>6}'.format("death penalty: ", '{:0.10f}'.format(death_penalty_sum / env.num_envs)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(done_sum / env.num_envs)))
    print('{:<40} {:>6}'.format("time elapsed: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(n_steps / (end - start))))
    print('std: ')
    print(nav_actor.distribution.std.cpu().detach().numpy())
    print('----------------------------------------------------\n')
