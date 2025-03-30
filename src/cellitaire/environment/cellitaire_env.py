import numpy as np
import pygame
import time
from gymnasium import spaces
from cellitaire.environment.ui.cellitaire_ui import CellitaireUI
from cellitaire.game.card import Card
from cellitaire.game.game import Game
from cellitaire.environment.ui.event_types import *

class CellitaireEnv:
    def __init__(self, reward, rows = 7, cols = 12, num_reserved = 6, max_moves = 300, max_illegal_moves = 300, render_mode=None, frame_rate=0.2):
        self.game = None
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

        self.render_mode = render_mode
        self.ui = None
        self.frame_rate = frame_rate

    def reset(self):
        self.game = Game()
        self.game.new_game(self.rows, self.cols, self.num_reserved)
        self.reward.reset()

        if self.render_mode == 'human' and self.ui is not None:
            self.publish_updates(0.0)
            pygame.event.post(pygame.event.Event(RESET))

        self.num_moves = 0
        self.num_illegal_moves = 0

        reward = 0
        done = False
        truncated = False
        state = self.get_state()
        info = {}
        return state, reward, done, truncated, info
    
    def get_legal_actions(self):
        special_coords, placeable_coords = self.game.board.get_special_slots()
        legal_actions = set(special_coords)
        if self.game.stockpile.count() > 0:
            legal_actions.update(placeable_coords)
        return list(legal_actions)
    
    def get_legal_actions_as_int(self):
        legal_actions = self.get_legal_actions()
        return [x * self.cols + y for x, y in legal_actions]
    
    def get_coords_map(self, coords):
        if not coords:
            return np.zeros(self.rows * self.cols, dtype=np.float32)
        
        coords = np.asarray(coords)
        coords_as_int = coords[:, 0] * self.cols + coords[:, 1]
        coords_map = np.zeros(self.rows * self.cols, dtype=np.float32)
        coords_map[coords_as_int] = 1.0
        return coords_map
    
    def get_lonely_coordinates_map(self):
        return self.get_coords_map(self.game.get_possible_lonely_coords())

    def get_suffocated_coordinates_map(self):
        return self.get_coords_map(self.game.get_possible_suffocated_coords())
    
    def get_placeable_coordinates_map(self):
        return self.get_coords_map(self.game.get_possible_placeable_coords())
    
    def get_board_state(self):
        board_state = np.array(
            [[slot.card.card_id if slot.card is not None else 0 for slot in row]
            for row in self.game.board.slots],
            dtype=np.float32
        )
        return board_state.flatten()

    def get_stockpile_state(self):
        top_card = self.game.stockpile.top_card()
        top_card_id = top_card.card_id if top_card is not None else 0
        return np.array([top_card_id, self.game.stockpile.count()], dtype=np.float32)

    def get_foundation_state(self):
        foundation_values = [
            Card.RANKS.index(card.rank) + 1 if card is not None else 0 
            for _, card in self.game.foundation.foundation.items()
        ]
        return np.array(foundation_values, dtype=np.float32)

    def get_state(self):
        board_state = self.get_board_state()
        stockpile_state = self.get_stockpile_state()
        foundation_state = self.get_foundation_state()
        lonely_coords_map = self.get_lonely_coordinates_map()
        suffocated_coords_map = self.get_suffocated_coordinates_map()
        placeable_coords_map = self.get_placeable_coordinates_map()
        return np.hstack((
            board_state,
            stockpile_state,
            foundation_state,
            lonely_coords_map,
            suffocated_coords_map,
            placeable_coords_map,
        ))
    
    def step(self, action):
        move = self.get_action_by_index(action)
        info = {}

        move_executed = self.game.make_move(move)
        if not move_executed:
            self.num_illegal_moves += 1
            info = {"illegal_move": True}
        else:
            self.num_moves += 1

        new_state = self.get_state()
        done = len(self.get_legal_actions()) < 1
        truncated = self.num_moves > self.max_moves or self.num_illegal_moves > self.max_illegal_moves

        reward = self.reward.calculate_reward(new_state, done, truncated, info)
        
        if self.render_mode == 'human' and self.ui is not None:
            self.publish_updates(reward)

        return new_state, reward, done, truncated, info
    
    def get_action_by_index(self, action_index):
        row = action_index // self.cols
        col = action_index % self.cols
        return (row, col)
    
    def publish_updates(self, reward):
        events = [
            pygame.event.Event(
                GU_SLOT_UPDATE,
                coordinate=(i, j),
                card=slot.card, 
                is_lonenly=slot.is_lonely, 
                is_suffocated=slot.is_suffocated,
                is_placeable=slot.is_placeable
            ) for i, row in enumerate(self.game.board.slots) for j, slot in enumerate(row)
        ]
        events.append(pygame.event.Event(GU_STOCKPILE_UPDATE, top_card=self.game.stockpile.top_card(), count=self.game.stockpile.count()))
        events.append(pygame.event.Event(GU_FOUNDATION_UPDATE, foundation_dict=self.game.foundation.foundation, total_saved=self.game.foundation.total_cards())) 
        events.append(pygame.event.Event(REWARD_UPDATED, reward=reward))
        events.extend(pygame.event.get())

        self.ui.add_events(events)

        time.sleep(self.frame_rate)
    
    def render(self):
        if self.render_mode == 'human':
            self.ui = CellitaireUI()
            self.ui.start()
        
    def close(self):
        if self.ui is not None:
            self.ui.kill()
            self.ui = None

    def __str__(self):
        return f"CellitaireEnv(game={self.game})"
