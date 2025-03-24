

import numpy as np
from cellitaire.environment.cellitaire_env import CellitaireEnv


class Agent:
    """
    Basic Agent that will randomly select legal actions for Cellitaire for one run
    """

    def __init__(self, cellitaire_env: CellitaireEnv):
        self.cellitaire_env = cellitaire_env
        self.step_count = 0
        pass

    def train(self):
        self.step_count = 0
        observation, reward, done, truncation = self.cellitaire_env.reset()
        while not done:
            possible_actions = self.cellitaire_env.get_legal_actions()
            action = np.random.choice(np.shape(possible_actions)[0])
            observation, reward, done, truncation = self.cellitaire_env.step(possible_actions[action])
            self.step_count += 1
            print(f'taking a random action out of {np.shape(possible_actions)[0]} choices!')
        
        print(f'Finished a run with {self.step_count} random steps!')