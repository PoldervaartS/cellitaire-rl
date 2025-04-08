from typing import List, Tuple
from numpy import ndarray


class Reward:
    def __init__(self, weight=1, rows=7, cols=12, num_reserved=6):
        self.prev_state: ndarray | None = None
        self.weight = weight
        self.stockpile_start = rows * cols
        self.foundation_start = self.stockpile_start + 2
        self.lonely_start = self.foundation_start + 4
        self.suffocated_start = self.lonely_start + rows * cols
        self.placeable_start = self.suffocated_start + rows * cols
        self.num_reserved = num_reserved
        pass

    def get_max_step_reward(self):
        return self.weight

    def get_available_moves_count(self, state):
        return state[self.lonely_start:].sum().item()

    def get_stockpile_cards_count(self, state):
        return state[self.stockpile_start + 1].item()

    def get_foundation_cards_count(self, state):
        return state[self.foundation_start:self.foundation_start + 4].sum().item()

    def reset(self):
        self.prev_cards_in_stockpile = 52 - self.num_reserved
        self.prev_cards_in_foundation = 0
        self.prev_state_available_moves = (self.num_reserved - 2) * 2

    def calculate_reward(self, new_state: ndarray, done: bool,
                         truncated: bool, info: any) -> float:

        raise NotImplementedError


class ConstantReward(Reward):
    def __init__(self, weight=1):
        super().__init__(weight)

    def calculate_reward(self, new_state: ndarray,
                         done: bool, truncated: bool, info: any):
        return self.weight


class WinReward(Reward):
    def __init__(self, weight=1000000, rows=7, cols=12):
        super().__init__(weight=weight, rows=rows, cols=cols)

    def calculate_reward(self, new_state: ndarray,
                         done: bool, truncated: bool, info: any):
        return self.weight if done and self.get_foundation_cards_count(
            new_state) == 52 else 0


class PlayedLegalMoveReward(Reward):
    def __init__(self, weight=1, rows=7, cols=12, num_reserved=6):
        super().__init__(weight=weight, rows=rows, cols=cols, num_reserved=num_reserved)

    def calculate_reward(self, new_state: ndarray,
                         done: bool, truncated: bool, info: any):
        num_cards_in_foundation = self.get_foundation_cards_count(new_state)
        num_cards_in_stockpile = self.get_stockpile_cards_count(new_state)

        reward = self.weight if self.prev_cards_in_foundation != num_cards_in_foundation or self.prev_cards_in_stockpile != num_cards_in_stockpile else 0

        self.prev_cards_in_foundation = num_cards_in_foundation
        self.prev_cards_in_stockpile = num_cards_in_stockpile
        return reward


class TruncatedReward(Reward):
    def __init__(self, weight=-1):
        super().__init__(weight=weight)

    def calculate_reward(self, new_state: ndarray,
                         done: bool, truncated: bool, info: any):
        return self.weight if truncated else 0


class CreatedMovesReward(Reward):
    def __init__(self, weight=1, rows=7, cols=12, foundation_count_dropoff=52):
        super().__init__(weight=weight, rows=rows, cols=cols)
        self.foundation_count_dropoff = foundation_count_dropoff

    def calculate_reward(self, new_state: ndarray,
                         done: bool, truncated: bool, info: any):
        num_available_moves = self.get_available_moves_count(new_state)
        foundation_count = self.get_foundation_cards_count(new_state)
        reward = 0 if foundation_count >= self.foundation_count_dropoff else self.weight if num_available_moves > self.prev_state_available_moves else 0
        self.prev_state_available_moves = num_available_moves
        return reward


class CombinedReward(Reward):
    def __init__(self, rewards_list: List[Reward]):
        self.rewards_list = rewards_list
        self.max_reward = self.get_max_step_reward()

    def calculate_reward(self, new_state: ndarray,
                         done: bool, truncated: bool, info: any):
        return sum([reward.calculate_reward(new_state, done, truncated, info)
                   for reward in self.rewards_list])

    def reset(self):
        for reward in self.rewards_list:
            reward.reset()

    def get_max_step_reward(self):
        return sum([reward.get_max_step_reward()
                   for reward in self.rewards_list])
