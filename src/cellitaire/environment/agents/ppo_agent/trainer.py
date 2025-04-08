import time
import numpy as np
import os
import torch

from cellitaire.environment.agents.memory import PPOMemory
from cellitaire.environment.agents.ppo_agent.agent import Agent


class AgentTrainer:
    def __init__(
        self,
        agent,
        env,
        checkpoint_dir,
        steps_per_learn_step=10000,
        batch_size=2000,
        normalize_reward=False,
        max_learn_steps=-1
    ):
        self.agent = agent
        try:
            self.agent.load_models()
        except Exception as e:
            print('New run, godspeed lad')
            os.mkdir(checkpoint_dir)
            self.agent.save_models()

        self.env = env
        self.steps_per_learn_step = steps_per_learn_step
        self.batch_size = batch_size
        self.normalize_reward = normalize_reward
        self.max_learn_steps = max_learn_steps

    def start_training(self):
        while running:
            new_memory = self.memory_queue.get()
            # print(f'Time since last batch {time.time() - last_batch}')
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
