import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class buffer:
    def __init__(self, num_envs, num_transitions_per_env, noise_num, ob_dim, act_dim, device):
        self.device = device

        # Core
        self.obs = torch.zeros((noise_num + 1) * num_transitions_per_env,  num_envs, ob_dim).to(self.device)
        self.actions = torch.zeros((noise_num + 1) * num_transitions_per_env, num_envs, act_dim).to(self.device)

        self.num_transitions_per_env = (noise_num + 1) * num_transitions_per_env
        self.num_envs = num_envs
        self.device = device

        self.step = 0

    def add_transitions(self, obs, actions):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.obs[self.step].copy_(torch.from_numpy(obs).to(self.device))
        self.actions[self.step].copy_(torch.from_numpy(actions).to(self.device))
        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator_shuffle(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):
            obs_batch = self.obs.view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            yield obs_batch, actions_batch

    def mini_batch_generator_inorder(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for batch_id in range(num_mini_batches):
            yield self.obs.view(-1, *self.obs.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                  self.actions.view(-1, self.actions.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]
