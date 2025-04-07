import torch.multiprocessing as mp
import copy
import time
import numpy as np
import os
import torch


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.gamma = gamma
        self.alpha = alpha
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self, memory):
        _, _, _, vals_arr, reward_arr, dones_arr, _ = memory.generate_batches()
        
        rewards = torch.tensor(reward_arr, dtype=torch.float32, device=self.actor.device)
        values = torch.tensor(vals_arr, dtype=torch.float32, device=self.actor.device)
        dones = torch.tensor(dones_arr, dtype=torch.float32, device=self.actor.device)

        advantage = compute_advantage(values, dones, rewards, self.gamma, self.gae_lambda)
        
        losses = np.array([])
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, _, _, _, batches = memory.generate_batches() 
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                losses = np.append(losses, total_loss.item())
        print(f'Average Loss: {losses.mean()}')
def compute_advantage(values, dones, rewards, gamma, gae_lambda):
    T = len(rewards)
    advantage = torch.zeros_like(rewards, device=values.device)
    
    delta = rewards[:-1] + gamma * values[1:] * (1 - dones[:-1]) - values[:-1]
    
    last_advantage = 0
    for t in reversed(range(T - 1)):
        non_terminal = 1 - dones[t]
        last_advantage = delta[t] + gamma * gae_lambda * non_terminal * last_advantage
        advantage[t] = last_advantage
    
    return advantage

def get_agent_copy(agent: Agent):
    agent_copy = Agent(
        n_actions=agent.n_actions, 
        input_dims=agent.input_dims, 
        gamma=agent.gamma,
        alpha=agent.alpha, 
        gae_lambda=agent.gae_lambda,
        policy_clip=agent.policy_clip, 
        batch_size=agent.batch_size, 
        n_epochs=agent.n_epochs,
        checkpoint_dir=agent.checkpoint_dir
    )
    
    agent_copy.actor.device = torch.device('cpu')
    agent_copy.actor.to('cpu')
    agent_copy.critic.device = torch.device('cpu')
    agent_copy.critic.to('cpu')
    
    return agent_copy

class ExperienceCollector(mp.get_context().Process):
    def __init__(self, agent, steps_to_post, batch_size, env, memory_queue, model_io_lock, normalize_reward=False, stop_event=None):
        super().__init__()
        self.agent = get_agent_copy(agent)
        self.steps_to_post = steps_to_post
        self.batch_size = batch_size
        self.env = copy.deepcopy(env)
        self.memory_queue = memory_queue
        self.model_io_lock = model_io_lock
        self.normalize_reward = normalize_reward
        self.memory = PPOMemory(batch_size)
        self.stop_event = stop_event or mp.Event()


    def post_experiences(self):
        self.memory_queue.put(self.memory)
        self.memory = PPOMemory(self.batch_size)

    def remember(self, observation, action, prob, val, reward, done):
        self.memory.store_memory(observation, action, prob, val, reward, done)

    def run(self):
        with self.model_io_lock:
            self.agent.load_models()
        n_steps = 0
        while not self.stop_event.is_set():
            observation = self.env.reset()[0]
            done = False
            truncated = False
            while not done and not truncated:
                if self.stop_event.is_set():
                    break
                n_steps += 1
                action, prob, val = self.agent.choose_action(observation)
                observation_, reward, done, truncated, info = self.env.step(action)
                self.remember(observation, action, prob, val, reward, done)
                if n_steps % self.steps_to_post == 0:
                    self.post_experiences()
                    with self.model_io_lock:
                        self.agent.load_models()
                observation = observation_

    def kill(self):
        self.stop_event.set()


class AgentTrainer:
    def __init__(self, agent, env, checkpoint_dir, steps_to_post=10000, batch_size=2000, collector_processes=8,
                 normalize_reward=False, max_learn_steps=-1, performance_games_to_sim=100, steps_between_performance_checks=50):
        self.agent = agent
        try:
            self.agent.load_models()
        except Exception as e:
            print('New run, godspeed lad')
            os.mkdir(checkpoint_dir)
            self.agent.save_models()

        self.env = env
        self.steps_to_post = steps_to_post
        self.batch_size = batch_size
        self.collector_processes = collector_processes
        self.normalize_reward = normalize_reward
        self.memory_queue = mp.Queue()
        self.monitor_signal_queue = mp.Queue()
        self.max_learn_steps = max_learn_steps
        self.performance_games_to_sim = performance_games_to_sim
        self.steps_between_performance_checks = steps_between_performance_checks
        self.model_io_lock = mp.Lock()

    def start_training(self):
        collectors = []
        collector_stop_event = mp.Event()
        new_collector = ExperienceCollector(self.agent, self.steps_to_post, self.batch_size, self.env,
                                    self.memory_queue, self.model_io_lock, self.normalize_reward, stop_event=collector_stop_event)
        new_collector.start()
        collectors.append(new_collector)
        time.sleep(8)

        learn_steps = 0
        running = True
        last_batch = time.time()
        try:
            while running:
                new_memory = self.memory_queue.get()
                print('-------------------')
                print(f'Time since last batch {time.time() - last_batch}')
                last_batch = time.time()
                print('learning')
                self.agent.learn(new_memory)
                print('learned')
                learn_steps += 1
                with self.model_io_lock:
                    self.agent.save_models()
                if learn_steps % (self.collector_processes * 2) == 0:
                    self.monitor_signal_queue.put(learn_steps)
                if len(collectors) < self.collector_processes:
                    time.sleep(8)
                    new_collector = ExperienceCollector(self.agent, self.steps_to_post, self.batch_size, self.env,
                                    self.memory_queue, self.model_io_lock, self.normalize_reward, stop_event=collector_stop_event)
                    new_collector.start()
                    collectors.append(new_collector)
                    
                if self.max_learn_steps != -1 and learn_steps >= self.max_learn_steps:
                    running = False
        finally:
            # Signal all processes to terminate and wait for them to finish
            collector_stop_event.set()
            for collector in collectors:
                collector.join()
