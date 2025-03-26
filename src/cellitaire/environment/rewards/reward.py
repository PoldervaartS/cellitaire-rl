from typing import List, Tuple
from numpy import ndarray

def get_available_moves_count(state):
    return state[-1].item()

def get_stockpile_cards(state):
    return state[-6].item()

def get_cards_in_foundation(state):
    return state[-5:-1].sum().item()

class Reward:
    def __init__(self, weight=1, rows=7, cols=12, num_reserved=6):
        self.prev_state: ndarray | None = None
        self.weight = weight
        self.rows = rows
        self.cols = cols
        self.num_reserved = num_reserved
        pass

    def reset(self):
        self.prev_cards_in_stockpile = 52 - self.num_reserved
        self.prev_cards_in_foundation = 0
        self.prev_state_available_moves = (self.num_reserved - 2) * 2

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any) -> float:

        raise NotImplementedError


class ConstantReward(Reward):
    def __init__(self, weight=1):
        super().__init__(weight)

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        return self.weight
    
class WinReward(Reward):
    def __init__(self, weight=1000000):
        super().__init__(weight=weight)

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        return self.weight if done and get_cards_in_foundation(new_state) == 52 else 0
    
class PlacedCardInFoundationReward(Reward):
    def __init__(self, weight=1):
        super().__init__(weight=weight)

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        num_cards_in_foundation = get_cards_in_foundation(new_state)
        reward = self.weight if num_cards_in_foundation > self.prev_cards_in_foundation else 0
        self.prev_cards_in_foundation = num_cards_in_foundation
        return reward
    
class PlayedLegalMoveReward(Reward):
    def __init__(self, weight=1, num_reserved=6):
        super().__init__(weight=weight, num_reserved=num_reserved)

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        num_cards_in_foundation = get_cards_in_foundation(new_state)
        num_cards_in_stockpile = get_stockpile_cards(new_state)
        
        reward = self.weight if self.prev_cards_in_foundation != num_cards_in_foundation or self.prev_cards_in_stockpile != num_cards_in_stockpile else 0
        
        self.prev_cards_in_foundation = num_cards_in_foundation
        self.prev_cards_in_stockpile = num_cards_in_stockpile
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
    
class TruncatedReward(Reward):
    def __init__(self, weight=-1):
        super().__init__(weight=weight)

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        return self.weight if truncated else 0
    
class CreatedMovesReward(Reward):
    def __init__(self, weight=1, num_reserved=6, foundation_count_dropoff=52):
        super().__init__(weight=weight, num_reserved=num_reserved)
        self.foundation_count_dropoff = foundation_count_dropoff

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        num_available_moves = get_available_moves_count(new_state)
        foundation_count = get_cards_in_foundation(new_state)
        reward = 0 if foundation_count >= self.foundation_count_dropoff else self.weight if num_available_moves > self.prev_state_available_moves else 0
        self.prev_state_available_moves = num_available_moves
        return reward

class CombinedReward(Reward):
    def __init__(self, rewards_list: List[Reward]):
        self.rewards_list = rewards_list
    
    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        return sum([reward.calculate_reward(new_state, done, truncated, info) for reward in self.rewards_list])
    
    def reset(self):
        for reward in self.rewards_list:
            reward.reset()
