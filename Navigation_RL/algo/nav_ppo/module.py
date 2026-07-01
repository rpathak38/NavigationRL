import torch.nn as nn
import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal


class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device

    def sample(self, obs):
        logits = self.architecture.architecture(obs).squeeze(dim=0)
        actions, log_prob = self.distribution.sample(logits)
        return actions, log_prob

    def evaluate(self, obs, actions):
        action_mean = self.architecture.architecture(obs).view(-1, *self.action_shape)
        return self.distribution.evaluate(obs, action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture.architecture(torch.from_numpy(obs).to(self.device))

    def deterministic_parameters(self):
        return self.architecture.parameters()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape


class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        return self.architecture.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture.architecture(obs).view(-1, 1)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape

class SharedConv(nn.Module):
    def __init__(self, grid_size=20, n_goal_features=4, n_cmd_history_features=15):
        super(SharedConv, self).__init__()
        self.grid_size = grid_size
        self.n_goal_features = n_goal_features
        self.n_cmd_history_features = n_cmd_history_features

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # (8, 10, 10)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # (16, 5, 5)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),  # (32, 2, 2)
            nn.ReLU(),
        )
        self.cnn_output_dim = 32 * 2 * 2  # 128
        self.latent_dim = self.cnn_output_dim + n_goal_features + n_cmd_history_features  # 147

    def forward(self, x):
        # x shape: (batch, 419)
        # Split into grid (400), goal features (4), and command history (15)
        grid_size = self.grid_size * self.grid_size
        grid = x[:, :grid_size]  # (batch, 400)
        goal = x[:, grid_size:grid_size + self.n_goal_features]  # (batch, 4)
        cmd_history = x[:, grid_size + self.n_goal_features:]  # (batch, 15)

        # Reshape grid to (batch, 1, 20, 20) for CNN
        grid = grid.view(-1, 1, self.grid_size, self.grid_size)

        # CNN features
        cnn_features = self.cnn(grid)  # (batch, 32, 2, 2)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)  # (batch, 128)

        # Concatenate with goal features and command history
        latent = torch.cat([cnn_features, goal, cmd_history], dim=1)  # (batch, 147)

        return latent


class NavActor(nn.Module):
    def __init__(self, grid_size=20, n_goal_features=4, n_cmd_history_features=15, action_dim=3, fc_shape=[128, 64], action_scale=1.0):
        super(NavActor, self).__init__()
        self.shared = SharedConv(grid_size, n_goal_features, n_cmd_history_features)
        self.action_scale = action_scale

        modules = []
        in_dim = self.shared.latent_dim
        for out_dim in fc_shape:
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.LeakyReLU())
            in_dim = out_dim
        modules.append(nn.Linear(in_dim, action_dim))

        self.fc = nn.Sequential(*modules)
        self.input_shape = [grid_size * grid_size + n_goal_features + n_cmd_history_features]
        self.output_shape = [action_dim]

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            elif isinstance(module, nn.Conv2d):
                torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))

    @property
    def architecture(self):
        return self

    def forward(self, x):
        latent = self.shared(x)
        return torch.tanh(self.fc(latent)) * self.action_scale


class NavCritic(nn.Module):
    def __init__(self, grid_size=20, n_goal_features=4, n_cmd_history_features=15, fc_shape=[128, 64]):
        super(NavCritic, self).__init__()
        self.shared = SharedConv(grid_size, n_goal_features, n_cmd_history_features)

        modules = []
        in_dim = self.shared.latent_dim
        for out_dim in fc_shape:
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.LeakyReLU())
            in_dim = out_dim
        modules.append(nn.Linear(in_dim, 1))

        self.fc = nn.Sequential(*modules)
        self.input_shape = [grid_size * grid_size + n_goal_features + n_cmd_history_features]
        self.output_shape = [1]

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            elif isinstance(module, nn.Conv2d):
                torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))

    @property
    def architecture(self):
        return self

    def forward(self, x):
        latent = self.shared(x)
        return self.fc(latent)


class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class GRU_MLP_Actor(nn.Module):
    def __init__(self, measurement_dim, hidden_dim, mlp_shape, mlp2_shape, output_size, batch_size, device='cuda'):
        super(GRU_MLP_Actor, self).__init__()
        self.device = device
        self.measurement_dim = measurement_dim
        self.estimation_dim = 10
        self.hidden_dim = hidden_dim
        self.mlp_shape = mlp_shape
        self.mlp2_shape = mlp2_shape
        self.batch_size = batch_size
        self.GRU = nn.GRU(input_size=self.measurement_dim, hidden_size=self.hidden_dim, num_layers=1).to(self.device)
        self.MLP = MLP(shape=self.mlp_shape, actionvation_fn=nn.LeakyReLU, input_size=self.hidden_dim, output_size=self.estimation_dim).to(self.device)
        self.MLP2 = MLP(shape=self.mlp2_shape, actionvation_fn=nn.LeakyReLU, input_size=self.hidden_dim + self.estimation_dim + 9, output_size=output_size).to(self.device)
        self.h = torch.zeros([1, self.batch_size, self.hidden_dim], dtype=torch.float32).to(self.device)
        self.init_weights(self.GRU)
        self.input_shape = [measurement_dim + 9]
        self.output_shape = [output_size]

    def forward(self, input):
        latent, self.h = self.GRU(input[:, :, :self.measurement_dim], self.h)
        estimation_pred = self.MLP.architecture(latent)
        action = self.MLP2.architecture(torch.cat((latent, estimation_pred, input[:, :, self.measurement_dim:]), dim=2))

        return action

    def init_hidden(self):
        self.h = torch.zeros([1, self.batch_size, self.hidden_dim], dtype=torch.float32).to(self.device)

    def init_by_done(self, arg_dones):
        self.h[:, arg_dones, :] = torch.zeros([1, arg_dones.size, self.hidden_dim], dtype=torch.float32).to(self.device)

    @staticmethod
    def init_weights(rnn):
        for layer in range(len(rnn.all_weights)):
            torch.nn.init.kaiming_normal_(rnn.all_weights[layer][0])
            torch.nn.init.kaiming_normal_(rnn.all_weights[layer][1])


class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None

    def sample(self, logits):
        self.distribution = Normal(logits, self.std.reshape(self.dim))

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=1).detach()

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std