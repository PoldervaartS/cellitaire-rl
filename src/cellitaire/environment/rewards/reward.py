

from typing import List, Tuple
from numpy import ndarray


class Reward:
    def __init__(self):
        self.prev_state: ndarray | None = None
        self.new_state: ndarray | None = None
        pass

    def calculate_reward(self, new_state: ndarray) -> float:

        raise NotImplementedError


class ConstantReward(Reward):
    def __init__(self):
        super().__init__()

    def calculate_reward(self, new_state):
        return 1

class CombinedReward(Reward):
    def __init__(self, rewards_list: List[Tuple[Reward, float]]):
        self.rewards_list = rewards_list
    
    def calculate_reward(self, new_state):
        return sum([reward.calculate_reward(new_state) * weight for reward, weight in self.rewards_list])
