import os
import numpy as np

from cellitaire.environment.agents.ppo_agent.agent import Agent


class AgentTrainer:
    def __init__(
        self,
        agent: Agent,
        env,
        checkpoint_dir,
        steps_per_learn_step=10000,
        batch_size=2000,
        normalize_reward=False,
        episodes_between_saves=100,
        max_learn_steps=-1
    ):
        self.agent = agent
        try:
            self.agent.load_models()
        except Exception as e:
            print('New run, godspeed lad')
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            self.agent.save_models()

        self.env = env
        self.steps_per_learn_step = steps_per_learn_step
        self.batch_size = batch_size
        self.normalize_reward = normalize_reward
        self.max_learn_steps = max_learn_steps
        self.episodes_between_saves = episodes_between_saves

    def print_metrics(self, episode_number, learning_iters, scores, cards_saved):
        recent_max_score = np.max(scores[-100:])
        recent_avg_score = np.mean(scores[-100:])

        recent_max_cards_saved = np.max(cards_saved[-100:])
        recent_avg_cards_saved = np.mean(cards_saved[-100:])

        print(f'Current Episode {episode_number:>7}: R Avg Score {recent_avg_score:>6.1f} | R Max Score {recent_max_score:>4.0f} | R Avg CS {recent_avg_cards_saved:>4.1f} | R Max CS {recent_max_cards_saved:>2.0f} | LS {learning_iters:5.0f}')

    def train(self):
        scores = []
        cards_saved = []
        episodes = 0

        running = True
        n_steps = 0
        learning_iters = 0
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
            if episodes % self.episodes_between_saves == 0:
                self.print_metrics(episodes, learning_iters,
                                   scores, cards_saved)
                self.agent.save_models()
