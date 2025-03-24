import numpy as np
from gymnasium import spaces
from cellitaire.game.card import Card
from cellitaire.game.game import Game  # Assumes your card is defined in model/model_builder.py

class CellitaireEnv:
    def __init__(self, reward, rows = 7, cols = 12, num_reserved = 6, max_moves = 300, max_illegal_moves = 300):
        self.game = None
        self.prev_foundation_count = 0
        self.prev_legal_moves = 0
        self.prev_stockpile_count = 0
        self.reward = reward
        self.rows = rows
        self.cols = cols
        self.num_reserved = num_reserved

        self.action_space = spaces.Discrete(rows * cols)

        self.num_moves = 0
        self.max_moves = max_moves
        self.num_illegal_moves = 0
        self.max_illegal_moves = max_illegal_moves

        # TODO: this definitely isn't right
        self.observation_space = spaces.Box(low=0.0, high=53.0, shape=(1, rows * cols + 6)) 

    def reset(self):
        self.game = Game()
        self.game.new_game(self.rows, self.cols, self.num_reserved)
        
        # Initialize previous feature values.
        self.prev_foundation_count = self.game.foundation.total_cards()
        self.prev_legal_moves = self.legal_actions_count()
        self.prev_stockpile_count = self.game.stockpile.count()

        self.num_moves = 0
        self.num_illegal_moves = 0

        reward = 0
        done = False
        state = self.get_state()
        info = {}
        return state, reward, done, info
    
    def get_legal_actions(self):
        special_coords, placeable_coords = self.game.board.get_special_slots()
        legal_actions = set(special_coords)
        if self.game.stockpile.count() > 0:
            legal_actions.update(placeable_coords)
        return list(legal_actions)

    def legal_actions_count(self):
        return len(self.get_legal_actions())

    def compute_reward(self):
        return 1
    
    def get_board_state(self):
        return np.array([[slot.card.card_id if slot.card != None else 0 for slot in row] for row in self.game.board.slots])

    def get_stockpile_state(self):
        top_card = self.game.stockpile.top_card()
        return np.concatenate(
            (np.array([top_card.card_id if top_card != None else 0], dtype=np.float32),
            np.array([self.game.stockpile.count()], dtype=np.float32))
        )

    def get_foundation_state(self):
        return np.array([Card.RANKS.index(card.rank) + 1 if card != None else 0 for _, card in self.game.foundation.foundation.items()], dtype=np.float32)

    def get_state(self):
        board_state = self.get_board_state()
        stockpile_state = self.get_stockpile_state()
        foundation_state = self.get_foundation_state()
        return np.concatenate((
            board_state.reshape(1, -1), 
            stockpile_state.reshape(1, -1), 
            foundation_state.reshape(1, -1)
            ), axis=1).squeeze(0)
    
    def step(self, action):
        action = self.get_action_by_index(action)
        if not self.game.board.can_change_slot(action):
            self.num_illegal_moves += 1
            # Illegal move: assign a penalty.
            reward = -1
            done = self.num_illegal_moves > self.max_illegal_moves
            state = self.get_state()
            info = {"illegal_move": True}
            return state, reward, done, info

        move_executed = self.game.make_move(action)
        if not move_executed:
            self.num_illegal_moves += 1
            # Move execution failed; treat as illegal.
            reward = -1
            done = self.num_illegal_moves > self.max_illegal_moves
            state = self.get_state()
            info = {"illegal_move": True}
            return state, reward, done, info
        
        self.num_moves += 1

        reward = self.compute_reward()
        # TODO: need to do truncate instead of just putting everything in done
        done = self.legal_actions_count() < 1 or self.num_moves > self.max_moves
        state = self.get_state()

        self.prev_foundation_count = self.game.foundation.total_cards()
        self.prev_legal_moves = self.legal_actions_count()
        self.prev_stockpile_count = self.game.stockpile.count()
        
        return state, reward, done, None
    
    def get_action_by_index(self, action_index):
        row = action_index // self.cols
        col = action_index % self.cols
        return (row, col)
    
    # TODO: would be cool to have human rendering
    def render(self):
        return self.__str__()

    def __str__(self):
        return f"CellitaireEnv(game={self.game})"
