import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from dataclasses import asdict, dataclass, field
from typing import List

from cellitaire.environment.agents.memory import PPOMemory
from torch.distributions.categorical import Categorical


class ActorNetwork(nn.Module):
    def __init__(
            self,
            n_actions,
            input_dims,
            alpha,
            layer_1_dim=1024,
            hidden_layer_dims=[1024],
            chkpt_dir='tmp/ppo'
    ):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        module_list = []
        module_list.append(nn.Linear(*input_dims, layer_1_dim))
        module_list.append(nn.LayerNorm(layer_1_dim))
        module_list.append(nn.ReLU())
        previous_dim = layer_1_dim

        for dim in hidden_layer_dims:
            module_list.append(nn.Linear(previous_dim, dim))
            module_list.append(nn.LayerNorm(dim))
            module_list.append(nn.ReLU())
            previous_dim = dim

        module_list.append(nn.Linear(hidden_layer_dims[-1], *n_actions))
        module_list.append(nn.Softmax(dim=-1))
        self.actor = nn.Sequential(*module_list)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(
            self,
            input_dims,
            alpha,
            layer_1_dim=2048,
            hidden_layer_dims=[2048],
            chkpt_dir='tmp/ppo'
    ):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

        module_list = []
        module_list.append(nn.Linear(*input_dims, layer_1_dim))
        module_list.append(nn.LayerNorm(layer_1_dim))
        module_list.append(nn.ReLU())
        previous_dim = layer_1_dim

        for dim in hidden_layer_dims:
            module_list.append(nn.Linear(previous_dim, dim))
            module_list.append(nn.LayerNorm(dim))
            module_list.append(nn.ReLU())
            previous_dim = dim

        module_list.append(nn.Linear(hidden_layer_dims[-1], 1))
        module_list.append(nn.Softmax(dim=-1))

        self.critic = nn.Sequential(*module_list)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(
            torch.load(
                self.checkpoint_file,
                map_location='cpu'))


@dataclass
class AgentConfig:
    gamma: float
    alpha: float
    gae_lambda: float
    policy_clip: float
    batch_size: int
    n_epochs: int
    layer_1_dim_actor: int = 1024
    hidden_layer_dims_actor: List[int] = field(default_factory=lambda: [1024])
    layer_1_dim_critic: int = 2048
    hidden_layer_dims_critic: List[int] = field(default_factory=lambda: [2048])


class Agent:
    def __init__(self, n_actions, input_dims, config: AgentConfig = None, config_dir='tmp/ppo'):
        self.n_actions = n_actions
        self.input_dims = input_dims

        self.config = config
        self.config_dir = config_dir
        self.config_file = None

        if self.config_dir is not None:
            self.config_file = os.path.join(
                self.config_dir, 'agent_config.json')

        if self.config_file is not None and self.config is None:
            self.load_config()

        assert self.config is not None, "Need to specify config or config path"

        self.initialize_from_config()

    def initialize_from_config(self):
        self.layer_1_dim_actor = self.config.layer_1_dim_actor
        self.hidden_layer_dims_actor = self.config.hidden_layer_dims_actor
        self.layer_1_dim_critic = self.config.layer_1_dim_critic
        self.hidden_layer_dims_critic = self.config.hidden_layer_dims_critic
        self.gamma = self.config.gamma
        self.alpha = self.config.alpha
        self.gamma = self.config.gamma
        self.gae_lambda = self.config.gae_lambda
        self.policy_clip = self.config.policy_clip
        self.batch_size = self.config.batch_size
        self.n_epochs = self.config.n_epochs

        self.actor = ActorNetwork(
            self.n_actions,
            self.input_dims,
            self.alpha,
            layer_1_dim=self.layer_1_dim_actor,
            hidden_layer_dims=self.hidden_layer_dims_actor,
            chkpt_dir=self.config_dir
        )
        self.critic = CriticNetwork(
            self.input_dims,
            self.alpha,
            layer_1_dim=self.layer_1_dim_critic,
            hidden_layer_dims=self.hidden_layer_dims_critic,
            chkpt_dir=self.config_dir)
        self.memory = PPOMemory(self.batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        # print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        # print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def load_config(self):
        with open(self.config_file, 'r') as f:
            c = json.load(f)
        self.config = AgentConfig(**c)

    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=4)

    def choose_action(self, observation):
        state = torch.tensor(
            np.array(
                [observation]),
            dtype=torch.float).to(
            self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def choose_legal_action(self, observation, legal_actions):
        with torch.no_grad():
            state = torch.tensor(
                np.array(
                    [observation]),
                dtype=torch.float).to(
                self.actor.device)

            dist = self.actor(state)
            value = self.critic(state)

            mask = torch.zeros(self.n_actions).to(self.actor.device)
            mask[legal_actions] = 1
            masked_dist = dist.probs * mask
            masked_dist = Categorical(masked_dist)
            action = masked_dist.sample()

            probs = torch.squeeze(dist.log_prob(action)).item()
            action = torch.squeeze(action).item()
            value = torch.squeeze(value).item()

            return action, probs, value

    def learn(self):
        _, _, _, vals_arr, reward_arr, dones_arr, _ = self.memory.generate_batches()

        rewards = torch.tensor(
            reward_arr,
            dtype=torch.float32,
            device=self.actor.device)
        values = torch.tensor(
            vals_arr,
            dtype=torch.float32,
            device=self.actor.device)
        dones = torch.tensor(
            dones_arr,
            dtype=torch.float32,
            device=self.actor.device)

        advantage = compute_advantage(
            values, dones, rewards, self.gamma, self.gae_lambda)

        losses = []
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, _, _, _, batches = self.memory.generate_batches()

            for batch in batches:
                states = torch.tensor(
                    state_arr[batch],
                    dtype=torch.float).to(
                    self.actor.device)
                old_probs = torch.tensor(
                    old_prob_arr[batch]).to(
                    self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs,
                                        weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                # nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
                # nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                losses.append(total_loss.item())
        # print(f'Average loss {np.mean(losses)}')
        self.memory.clear_memory()


def compute_advantage(values, dones, rewards, gamma, gae_lambda):
    T = len(rewards)
    advantage = torch.zeros_like(rewards, device=values.device)

    # Compute the temporal difference errors (delta)
    # Note: Assuming dones is 0 or 1 so that (1-dones) is a mask.
    delta = rewards[:-1] + gamma * values[1:] * (1 - dones[:-1]) - values[:-1]

    last_advantage = 0
    # Loop backwards (from T-2 down to 0)
    for t in reversed(range(T - 1)):
        non_terminal = 1 - dones[t]
        last_advantage = delta[t] + gamma * \
            gae_lambda * non_terminal * last_advantage
        advantage[t] = last_advantage

    return advantage
