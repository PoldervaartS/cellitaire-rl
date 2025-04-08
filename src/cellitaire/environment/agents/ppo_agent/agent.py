import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from cellitaire.environment.agents.memory import PPOMemory
from torch.distributions.categorical import Categorical


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=1024, fc2_dims=1024, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

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
    def __init__(self, input_dims, alpha, fc1_dims=2048, fc2_dims=2048,
                 chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

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


class Agent:
    def __init__(self, n_actions, input_dims, fc1_actor=1024, fc2_actor=1024, fc1_critic=2048, fc2_critic=2048, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, checkpoint_dir='tmp/ppo'):
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.fc1_actor = fc1_actor
        self.fc2_actor = fc2_actor
        self.fc1_critic = fc1_critic
        self.fc2_critic = fc2_critic
        self.gamma = gamma
        self.alpha = alpha
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_actions = n_actions
        self.checkpoint_dir = checkpoint_dir

        self.actor = ActorNetwork(
            n_actions,
            input_dims,
            alpha,
            fc1_dims=fc1_actor,
            fc2_dims=fc2_actor,
            chkpt_dir=checkpoint_dir)
        self.critic = CriticNetwork(
            input_dims,
            alpha,
            fc1_dims=fc1_critic,
            fc2_dims=fc2_critic,
            chkpt_dir=checkpoint_dir)
        self.memory = PPOMemory(batch_size)

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
