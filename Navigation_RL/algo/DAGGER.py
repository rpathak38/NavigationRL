import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

class Dagger:
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, action_shape, device,
                 architecture, learning_rate, num_learning_epochs, num_mini_batches,
                 log_dir):
        self.device = device

        # Core
        self.obs = torch.zeros(num_transitions_per_env, num_envs, *obs_shape).to(self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *action_shape).to(self.device)
        self.obs_shape = obs_shape
        self.action_shape = action_shape


        # For SL
        self.architecture = architecture
        self.architecture.to(device)
        self.optimizer = optim.Adam([*self.architecture.parameters()], lr=learning_rate)
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.device = device

        self.step = 0

        # Log
        self.net_save_dir = log_dir
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.num_update = 0

    def add_transitions(self, obs, actions):
        self.obs[self.step].copy_(torch.from_numpy(obs).to(self.device))
        self.actions[self.step].copy_(torch.from_numpy(actions).to(self.device))
        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator_shuffle(self, num_mini_batches, update):
        batch_size = self.num_envs * self.num_transitions_per_env * (update + 1)
        mini_batch_size = batch_size // num_mini_batches

        for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):
            obs_batch = self.obs.view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            yield obs_batch, actions_batch

    def mini_batch_generator_inorder(self, num_mini_batches, update):
        batch_size = self.num_envs * self.num_transitions_per_env * (update + 1)
        mini_batch_size = batch_size // num_mini_batches

        for batch_id in range(num_mini_batches):
            yield self.obs.view(-1, *self.obs.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                  self.actions.view(-1, self.actions.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]

    def update(self, update):
        self._train_step(update)
        self.obs.resize_as_(torch.zeros((update + 2) * self.num_transitions_per_env, self.num_envs, *self.obs_shape))
        self.actions.resize_as_(torch.zeros((update + 2) * self.num_transitions_per_env, self.num_envs, *self.action_shape))
        self.num_mini_batches += 1

    def _train_step(self,update):
        for epoch in range(self.num_learning_epochs):
            mean_loss = 0
            for obs_batch, actions_batch in self.mini_batch_generator_shuffle(self.num_mini_batches, update):

                loss = (self.architecture.architecture(obs_batch) - actions_batch).pow(2).mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                mean_loss += loss.item()

            mean_loss /= self.num_mini_batches
            self.writer.add_scalar('Loss/', mean_loss, self.num_update + epoch)

        self.num_update += epoch