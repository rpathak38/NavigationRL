from ruamel.yaml import YAML
from io import StringIO
from Navigation_RL.bin import rsg_NavRL
from Navigation_RL.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from Navigation_RL.helper.raisim_gym_helper import ConfigurationSaver
import Navigation_RL.algo.ppo.module as ppo_module
import Navigation_RL.algo.ppo.ppo as PPO
import os
import math
import time
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse

# directories
home_path = os.path.dirname(os.path.realpath(__file__)) + "/.."

# config
yaml = YAML(typ='unsafe', pure=True)
yaml.default_flow_style = False
cfg = yaml.load(open(home_path + "/rsc/cfg.yaml", 'r'))

reward_list_temp = cfg['environment']['rewardCoeff']
reward_list = []
for key,_ in reward_list_temp.items():
    reward_list.append(key)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
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
value_ob_dim = cfg['environment']['value_ob_dim']
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs
log_intervals = 1

# save the configuration and other files
saver = ConfigurationSaver(log_dir=home_path + "/data/Navigation_RL",
                           save_items=[home_path + "/rsc/cfg.yaml", home_path + "/src/env/Environment.hpp", home_path + "/script/RL.py",
                                       home_path + "/Navigation_RL/algo/ppo/ppo.py"])

# Set hyperparameters based on mode
if weight_path and os.path.exists(weight_path):
    lr = 1.0e-5
    entropy_coef = 0.75e-4
    init_std = 1.0
else:
    lr = 5.0e-4
    entropy_coef = 0.01
    init_std = 1.0

module = ppo_module.MLP(cfg['architecture']['policy_net'],
                        nn.LeakyReLU,
                        ob_dim,
                        act_dim)

actor = ppo_module.Actor(module,
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, init_std),
                         device=device)

critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'],
                                          nn.LeakyReLU,
                                          value_ob_dim,
                                          1),
                           device=device)

# Load pretrained weights if provided
if weight_path and os.path.exists(weight_path):
    print(f"Loading pretrained weights from {weight_path}")
    actor_checkpoint = torch.load(weight_path + '/full_2500.pt', map_location=device)
    actor.architecture.load_state_dict(actor_checkpoint['actor_architecture_state_dict'])
    if os.path.exists(weight_path + '/mean2500.csv'):
        env.load_scaling(weight_path, 2500)
        env.obs_rms.count += 10 * n_steps * 100  # approximate scaling count
    print("Pretrained weights loaded successfully")
    print(f"Fine-tuning for {num_iterations} iterations...")
else:
    print("Training from scratch")
    print(f"Training for {num_iterations} iterations...")

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
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
              mini_batch_sampling='in_order',
              log_intervals=log_intervals
              )

for update in range(num_iterations + 1):
    start = time.time()
    reward_ll_sum = 0
    real_reward_ll_sum = 0
    info = np.zeros(shape=(env.num_rewards, n_steps, env.num_envs))
    various_rewards_sum = np.zeros(shape=(env.num_rewards, 1))
    done_sum = 0
    average_dones = 0.
    if not keep_viz:
        env.turn_off_visualization()

    if update != 0 and update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        # Save with _finetuned suffix if fine-tuning, otherwise use original naming
        if weight_path:
            save_name = f"full_{update}_finetuned.pt"
        else:
            save_name = f"full_{update}.pt"
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir + "/" + save_name)
        env.save_scaling(saver.data_dir, str(update))
        # we create another graph just to demonstrate the save/load method
        loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
        loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])
        loaded_graph.to(device)

        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')
        time.sleep(1)
        for iteration in range(5):
            env.map_change()
            env.reset()
            for step in range(n_steps):
                frame_start = time.time()
                obs = env.observe(False)
                action_ll = loaded_graph.architecture(torch.from_numpy(obs).to(device))
                env.step(action_ll.cpu().detach().numpy())
                frame_end = time.time()
                wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                if wait_time > 0.:
                    time.sleep(wait_time)
        env.stop_video_recording()
        if not keep_viz:
            env.turn_off_visualization()

    env.reset()
    # actual training
    for step in range(n_steps):
        obs = env.observe()
        value_obs = env.value_observe()
        action = ppo.observe(obs)
        real_reward, dones = env.step(action)
        various_rewards_mean, various_rewards_stdev = env.getrewards()
        if(update % log_intervals == 0):
            for i in range(env.num_rewards):
                info[i, step, :] = various_rewards_mean[:, i].transpose()
        ppo.step(value_obs=value_obs, rews=real_reward, dones=dones)
        done_sum = done_sum + sum(dones)
        reward = env.get_logging_reward()
        reward_ll_sum = reward_ll_sum + sum(reward)
        real_reward_ll_sum = real_reward_ll_sum + sum(real_reward)
        for i in range(env.num_rewards):
            various_rewards_sum[i, 0] = various_rewards_sum[i, 0] + sum(various_rewards_mean[:, i])


    average_ll_performance = reward_ll_sum / total_steps
    average_real_ll_performance = real_reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    average_various_rewards = various_rewards_sum / total_steps

    # take st step to get value obs
    obs = env.observe()
    value_obs = env.value_observe()
    ppo.update(actor_obs=obs,
               value_obs=value_obs,
               log_this_iteration=update % log_intervals == 0,
               update=update,
               info=[reward_list, info, average_ll_performance, average_real_ll_performance, average_dones])

    if update % cfg['environment']['curriculum']['iteration_per_update'] == 0:
        env.curriculum_callback()

    end = time.time()
    actor.distribution.enforce_minimum_std((torch.ones(act_dim)*0.1).to(device))
    print('----------------------------------------------------')
    for i in range(env.num_rewards):
        print('{name:<30} {mean_name:<7} {mean_value:<14.8f} {stdev_name:<7} {stdev_value:>10.8f} '.format(name=reward_list[i], mean_name="mean", mean_value=average_various_rewards[i,0],
                                                                                                            stdev_name="stdev", stdev_value=various_rewards_stdev[i]))
    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('std: ')
    print(actor.distribution.std.cpu().detach().numpy())
    print('----------------------------------------------------\n')
