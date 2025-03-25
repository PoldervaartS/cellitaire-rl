from typing import List, Tuple
from numpy import ndarray

def get_stockpile_cards(state, rows, cols, num_reserved):
    if state is None:
        return 52 - num_reserved
    board_size = rows * cols
    return state[board_size + 1].item()

def get_cards_in_foundation(state, rows, cols):
    if state is None:
        return 0
    board_size = rows * cols
    return state[board_size + 2:].sum().item()

class Reward:
    def __init__(self, weight=1, rows=7, cols=12, num_reserved=6):
        self.prev_state: ndarray | None = None
        self.weight = weight
        self.rows = rows
        self.cols = cols
        self.num_reserved = num_reserved
        pass

    def reset(self):
        self.prev_state = None

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any) -> float:

        raise NotImplementedError


class ConstantReward(Reward):
    def __init__(self, weight=1):
        super().__init__(weight)

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        return self.weight
    
class WinReward(Reward):
    def __init__(self, weight=1000000, rows=7, cols=12, num_reserved=6):
        super().__init__(weight, rows, cols, num_reserved)

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        return self.weight if done and get_cards_in_foundation(new_state, self.rows, self.cols) == 52 else 0
    
class PlacedCardInFoundationReward(Reward):
    def __init__(self, weight=1, rows=7, cols=12, num_reserved=6):
        super().__init__(weight=weight, rows=rows, cols=cols, num_reserved=num_reserved)

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        reward = self.weight if get_cards_in_foundation(new_state, self.rows, self.cols) > get_cards_in_foundation(self.prev_state, self.rows, self.cols) else 0
        self.prev_state = new_state
        return reward
    
class PlayedLegalMoveReward(Reward):
    def __init__(self, weight=1, rows=7, cols=12, num_reserved=6):
        super().__init__(weight=weight, rows=rows, cols=cols, num_reserved=num_reserved)

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        prev_cards_in_foundation = get_cards_in_foundation(self.prev_state, self.rows, self.cols)
        prev_cards_in_stockpile = get_stockpile_cards(self.prev_state, self.rows, self.cols, self.num_reserved)
        new_cards_in_foundation = get_cards_in_foundation(new_state, self.rows, self.cols)
        new_cards_in_stockpile = get_stockpile_cards(new_state, self.rows, self.cols, self.num_reserved)
        
        reward = self.weight if prev_cards_in_foundation != new_cards_in_foundation or prev_cards_in_stockpile != new_cards_in_stockpile else 0
        
        self.prev_state = new_state
        return reward
    
class PeriodicPlacedCardInFoundationReward(Reward):
    def __init__(self, weight=1, rows=7, cols=12, num_reserved=6, reward_period=2):
        super().__init__(weight=weight, rows=rows, cols=cols, num_reserved=num_reserved)        
        self.reward_period = reward_period

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        new_cards_in_foundation = get_cards_in_foundation(new_state, self.rows, self.cols)
        old_cards_in_foundation = get_cards_in_foundation(self.prev_state, self.rows, self.cols)
        self.prev_state = new_state
        if new_cards_in_foundation <= old_cards_in_foundation:
            return 0
        reward = self.weight if new_cards_in_foundation % self.reward_period else 0
        return reward
    
class TruncatedReward(Reward):
    def __init__(self, weight=-1, rows=7, cols=12, num_reserved=6):
        super().__init__(weight, rows, cols, num_reserved)

    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        return self.weight if truncated else 0

class CombinedReward(Reward):
    def __init__(self, rewards_list: List[Reward]):
        self.rewards_list = rewards_list
    
    def calculate_reward(self, new_state: ndarray, done: bool, truncated: bool, info: any):
        return sum([reward.calculate_reward(new_state, done, truncated, info) for reward in self.rewards_list])
    
    def reset(self):
        for reward in self.rewards_list:
            reward.reset()
