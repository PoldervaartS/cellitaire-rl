import multiprocessing
import copy
import time
import numpy as np
import os
import torch

from cellitaire.environment.agents.PPOMemory import PPOMemory
from cellitaire.environment.agents.PPOAgent import Agent

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
        checkpoint_dir=agent.checkpoint_dir,
        fc1_actor=agent.fc1_actor,
        fc2_actor=agent.fc2_actor,
        fc1_critic=agent.fc1_critic,
        fc2_critic=agent.fc2_critic
    )
    
    agent_copy.actor.device = torch.device('cpu')
    agent_copy.actor.to('cpu')
    agent_copy.critic.device = torch.device('cpu')
    agent_copy.critic.to('cpu')
    
    return agent_copy

class ExperienceCollector(multiprocessing.get_context().Process):
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
        # Use a shared event to signal stopping the process
        self.stop_event = stop_event or multiprocessing.Event()


    def post_experiences(self):
        self.memory_queue.put(self.memory)
        self.memory = PPOMemory(self.batch_size)

    def remember(self, observation, action, prob, val, reward, done):
        self.memory.store_memory(observation, action, prob, val, reward, done)

    def run(self):
        n_steps = 0
        while not self.stop_event.is_set():
            with self.model_io_lock:
                self.agent.load_models()
            self.env.reset()
            observation = self.env.get_state()
            done = False
            truncated = False
            while not done and not truncated:
                if self.stop_event.is_set():
                    break
                n_steps += 1
                action, prob, val = self.agent.choose_legal_action_mostly(observation, self.env.get_legal_actions_as_int())
                observation_, reward, done, truncated, info = self.env.step(action)
                if self.normalize_reward:
                    reward = reward / self.env.reward.max_reward
                self.remember(observation, action, prob, val, reward, done)
                if n_steps % self.steps_to_post == 0:
                    self.post_experiences()
                observation = observation_

    def kill(self):
        self.stop_event.set()


class AgentPerformanceMonitor(multiprocessing.get_context().Process):
    def __init__(self, agent, env, signal_queue, model_io_lock, episodes_to_sim=100, normalize_reward=False, stop_event=None):
        super().__init__()
        self.agent = get_agent_copy(agent)
        self.agent.max_moves = 1200
        self.env = copy.deepcopy(env)
        self.signal_queue = signal_queue
        self.model_io_lock = model_io_lock
        self.episodes_to_sim = episodes_to_sim
        self.normalize_reward = normalize_reward
        self.stop_event = stop_event or multiprocessing.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Use a timeout to avoid blocking indefinitely
                learn_steps = self.signal_queue.get(timeout=1)
            except:
                continue
            if self.stop_event.is_set():
                break
            print(f'Model has had {learn_steps} Learning Steps. Simming {self.episodes_to_sim} games...')
            episodes_simmed = 0
            scores = []
            cards_saved = []
            while episodes_simmed < self.episodes_to_sim:
                with self.model_io_lock:
                    self.agent.load_models()
                self.env.reset()
                observation = self.env.get_state()
                done = False
                truncated = False
                score = 0
                while not done:
                    action, _, _ = self.agent.choose_legal_action_mostly(observation, self.env.get_legal_actions_as_int())
                    observation_, reward, done, truncated, _ = self.env.step(action)
                    if self.normalize_reward:
                        reward = reward / self.env.reward.max_reward
                    score += reward
                    observation = observation_
                scores.append(score)
                cards_saved.append(self.env.game.foundation.total_cards())
                episodes_simmed += 1
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            avg_cs = np.mean(cards_saved)
            max_cs = np.max(cards_saved)
            print(f'Simmed {self.episodes_to_sim} games | Average Score {avg_score:>6.1f} | Best Score {max_score:>6.1f} | Average Cards Saved {avg_cs:>2.0f} | Max Cards Saved {max_cs:>2.0f}')

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
        self.memory_queue = multiprocessing.Queue()
        self.monitor_signal_queue = multiprocessing.Queue()
        self.max_learn_steps = max_learn_steps
        self.performance_games_to_sim = performance_games_to_sim
        self.steps_between_performance_checks = steps_between_performance_checks
        self.model_io_lock = multiprocessing.Lock()

    def start_training(self):
        collectors = []
        collector_stop_event = multiprocessing.Event()
        new_collector = ExperienceCollector(self.agent, self.steps_to_post, self.batch_size, self.env,
                                    self.memory_queue, self.model_io_lock, self.normalize_reward, stop_event=collector_stop_event)
        new_collector.start()
        collectors.append(new_collector)
        time.sleep(8)

        monitor_stop_event = multiprocessing.Event()
        performance_monitor = AgentPerformanceMonitor(self.agent, self.env, self.monitor_signal_queue, model_io_lock=self.model_io_lock,
                                                        episodes_to_sim=self.performance_games_to_sim,
                                                        normalize_reward=self.normalize_reward,
                                                        stop_event=monitor_stop_event)
        performance_monitor.start()

        learn_steps = 0
        running = True
        last_batch = time.time()
        try:
            while running:
                new_memory = self.memory_queue.get()
                #print(f'Time since last batch {time.time() - last_batch}')
                last_batch = time.time()
                self.agent.learn(new_memory)
                learn_steps += 1
                with self.model_io_lock:
                    self.agent.save_models()
                if learn_steps % (self.collector_processes * 3) == 0:
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
            monitor_stop_event.set()
            for collector in collectors:
                collector.join()
            performance_monitor.join()
            self.monitor_signal_queue.put(0)