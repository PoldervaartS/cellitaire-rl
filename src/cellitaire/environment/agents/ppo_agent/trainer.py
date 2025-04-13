from dataclasses import asdict, dataclass
import json
import os
import numpy as np

from cellitaire.environment.agents.ppo_agent.agent import Agent


@dataclass
class PerformanceCheckpoint:
    learning_steps: int
    max_average_score: float
    max_average_cards_saved: float


@dataclass
class AgentTrainerConfig:
    steps_per_learn_step: int = 10000
    batch_size: int = 2000
    normalize_reward: bool = False
    episodes_between_performance_checks: int = 100
    max_learn_steps: int = -1


class AgentTrainer:
    def __init__(
        self,
        agent: Agent,
        env,
        config_dir,
        config: AgentTrainerConfig = None
    ):
        self.agent = agent
        self.env = env
        self.config_dir = config_dir
        try:
            self.agent.load_models()
        except Exception as e:
            print('New run, godspeed lad')
            if not os.path.isdir(self.config_dir):
                os.mkdir(self.config_dir)
            self.agent.save_models()

        self.config = config
        self.config_file = None

        self.config_file = os.path.join(
            self.config_dir, 'trainer_config.json')

        if self.config_file is not None and self.config is None:
            self.load_config()

        assert self.config is not None, "Need to specify config or config path"

        self.initialize_from_config()
        self.save_config()

        self.performance_checkpoint_file = os.path.join(
            self.config_dir, "performance_checkpoint.json")
        try:
            self.load_performance_checkpoint()
        except Exception as e:
            self.performance_checkpoint = PerformanceCheckpoint(
                learning_steps=0, max_average_score=-1.0, max_average_cards_saved=-1)
            self.save_performance_checkpoint()

    def initialize_from_config(self):
        self.steps_per_learn_step = self.config.steps_per_learn_step
        self.batch_size = self.config.batch_size
        self.normalize_reward = self.config.normalize_reward
        self.max_learn_steps = self.config.max_learn_steps
        self.episodes_between_performance_checks = self.config.episodes_between_performance_checks

    def load_config(self):
        with open(self.config_file, 'r') as f:
            c = json.load(f)
        self.config = AgentTrainerConfig(**c)

    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=4)

    def load_performance_checkpoint(self):
        with open(self.performance_checkpoint_file, 'r') as f:
            c = json.load(f)
        self.performance_checkpoint = PerformanceCheckpoint(**c)

    def save_performance_checkpoint(self):
        with open(self.performance_checkpoint_file, 'w') as f:
            json.dump(asdict(self.performance_checkpoint), f, indent=4)

    def update_performance_checkpoint(self, episode_number, learning_iters, scores, cards_saved):
        recent_max_score = np.max(scores[-100:])
        recent_avg_score = np.mean(scores[-100:])

        recent_max_cards_saved = np.max(cards_saved[-100:])
        recent_avg_cards_saved = np.mean(cards_saved[-100:])

        if recent_avg_score > self.performance_checkpoint.max_average_score:
            self.performance_checkpoint.max_average_score = recent_avg_score
            self.performance_checkpoint.max_average_cards_saved = recent_avg_cards_saved
            self.performance_checkpoint.learning_steps = learning_iters
            self.save_performance_checkpoint()
            self.agent.save_models()

        print(f'Current Episode {episode_number:>7}: R Avg Score {recent_avg_score:>6.1f} | R Max Score {recent_max_score:>4.0f} | R Avg CS {recent_avg_cards_saved:>4.1f} | R Max CS {recent_max_cards_saved:>2.0f} | LS {learning_iters:5.0f}')

    def train(self):
        scores = []
        cards_saved = []
        episodes = 0

        running = True
        n_steps = 0
        learning_iters = self.performance_checkpoint.learning_steps
        while running:
            self.env.reset()
            score = 0
            observation = self.env.get_state()
            done = False
            truncated = False
            while not done and not truncated:
                n_steps += 1

                action, prob, val = self.agent.choose_legal_action(
                    observation, self.env.get_legal_actions_as_int())
                observation_, reward, done, truncated, _ = self.env.step(
                    action)
                if self.normalize_reward:
                    reward = reward / self.env.reward.max_reward
                score += reward

                self.agent.remember(observation, action,
                                    prob, val, reward, done)

                if n_steps % self.steps_per_learn_step == 0:
                    self.agent.learn()
                    learning_iters += 1

                observation = observation_
            episodes += 1
            scores.append(score)
            cards_saved.append(self.env.game.foundation.total_cards())
            if episodes % self.episodes_between_performance_checks == 0:
                self.update_performance_checkpoint(
                    episodes, learning_iters, scores, cards_saved)
