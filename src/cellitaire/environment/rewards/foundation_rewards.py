from numpy import ndarray
from cellitaire.environment.rewards.reward import Reward, get_cards_in_foundation

class PlacedCardInFoundationReward(Reward):
    def __init__(self, weight=1):
        super().__init__(weight=weight)

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        num_cards_in_foundation = get_cards_in_foundation(new_state)
        reward = self.weight if num_cards_in_foundation > self.prev_cards_in_foundation else 0
        self.prev_cards_in_foundation = num_cards_in_foundation
        return reward

class PeriodicPlacedCardInFoundationReward(Reward):
    def __init__(self, weight=1, reward_period=2):
        super().__init__(weight=weight)        
        self.reward_period = reward_period

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        num_cards_in_foundation = get_cards_in_foundation(new_state)
        self.prev_state = new_state
        if num_cards_in_foundation <= self.prev_cards_in_foundation:
            return 0
        reward = self.weight if num_cards_in_foundation % self.reward_period else 0
        self.prev_cards_in_foundation = num_cards_in_foundation
        return reward
    

class ScalingPlacedCardInFoundationReward(Reward):
    def __init__(self, weight=1):
        super().__init__(weight)

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        num_cards_in_foundation = get_cards_in_foundation(new_state)
        reward = self.weight * num_cards_in_foundation if num_cards_in_foundation > self.prev_cards_in_foundation else 0
        self.prev_cards_in_foundation = num_cards_in_foundation
        return reward
    
