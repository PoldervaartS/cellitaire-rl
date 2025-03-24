# game/cellitaire_env.py

import numpy as np
from cellitaire.game.card import Card
from cellitaire.game.game import Game  # Assumes your card is defined in model/model_builder.py

class CellitaireEnv:
    def __init__(self, config, reward):
        """
        Initialize the environment with a configuration dictionary.
        
        Expected config keys include:
          - board_rows, board_cols, initial_reserved, etc.
          - reward_weights (optional): a dict with keys "r1", "r2", ..., "rN".
          - num_reward_features (optional): the number of reward features; default is 6.
          - All configuration options needed by the ModelBuilder (e.g., card_representation, board_rnn_units, etc.)
        """
        self.config = config
        self.game = None
        self.prev_foundation_count = 0
        self.prev_legal_moves = 0
        self.prev_stockpile_count = 0
        self.reward = reward

        # Set the number of reward features (default 6).
        self.num_reward_features = config.get("num_reward_features", 6)
        
        # Randomize reward weights if not provided.
        if "reward_weights" not in self.config:
            random_weights = np.random.uniform(low=0.5, high=1.5, size=(self.num_reward_features,))
            self.reward_weights = {
                f"r{i+1}": float(random_weights[i]) for i in range(self.num_reward_features)
            }
        else:
            self.reward_weights = self.config["reward_weights"]

    def reset(self, rows=None, cols=None, initial_reserved=None):
        """
        Resets the environment by creating a new game.
        Initializes previous features for foundation count, legal moves, and stockpile count.
        """
        if rows is None:
            rows = self.config.get("board_rows", 7)
        if cols is None:
            cols = self.config.get("board_cols", 12)
        if initial_reserved is None:
            initial_reserved = self.config.get("initial_reserved", 6)
        
        self.game = Game()
        self.game.new_game(rows, cols, initial_reserved)
        
        # Initialize previous feature values.
        self.prev_foundation_count = self.game.foundation.total_cards()
        self.prev_legal_moves = self._compute_legal_moves_count()
        self.prev_stockpile_count = self.game.stockpile.count()

        reward = 0
        done = False
        state = self.get_state()
        info = {}
        return state, reward, done, info


    def _compute_legal_moves_count(self):
        """
        Computes the total number of legal moves available on the board.
        A move is legal if the slot is empty and placeable or if the slot is occupied
        and the card is lonely or suffocated.
        """
        board = self.game.board
        legal_moves = 0
        for r in range(board.rows):
            for c in range(board.cols):
                if board.can_change_slot((r, c)):
                    legal_moves += 1
        return legal_moves

    def compute_reward(self):
        """
        Computes the reward for the most recent move using weighted features.
        The reward components are:
        - r1: Standard per-move penalty.
        - r2: Bonus for an increase in foundation count.
        - r3: The portion of cards on the board relative to the total in play.
        - r4: The change in the number of legal moves (normalized).
        - r5: Bonus if the stockpile count increased (i.e. a card was added to the stockpile).
        - r6: Bonus if the stockpile is empty.
        
        Returns:
            (normalized_reward, reward_components): where normalized_reward is the weighted sum
            normalized to roughly be in [-1, 1], and reward_components is a dict with each raw component.
        """
        # r1: A constant penalty per move.
        r1 = -0.01

        # r2: Bonus for an increase in the foundation count.
        current_foundation_count = self.game.foundation.total_cards()
        r2 = 0.0
        if current_foundation_count > self.prev_foundation_count:
            r2 = 0.5 * (current_foundation_count - self.prev_foundation_count)

        # r3: Portion of cards on the board.
        board_cards = sum(1 for row in self.game.board.slots for slot in row if slot.has_card())
        stock_count = self.game.stockpile.count()
        total_in_play = board_cards + stock_count
        r3 = board_cards / total_in_play if total_in_play > 0 else 0

        # r4: Change in the number of legal moves, normalized by total slots.
        current_legal_moves = self._compute_legal_moves_count()
        total_slots = self.game.board.rows * self.game.board.cols
        r4 = (current_legal_moves - self.prev_legal_moves) / total_slots

        # r5: Bonus if the stockpile count increased (i.e., a card was added to the stockpile).
        current_stock_count = stock_count
        r5 = 0.0
        if current_stock_count > self.prev_stockpile_count:
            r5 = 0.1 * (current_stock_count - self.prev_stockpile_count)

        # r6: Bonus if the stockpile is empty.
        r6 = 1.0 if current_stock_count == 0 else 0.0

        # Bundle reward components.
        reward_components = {"r1": r1, "r2": r2, "r3": r3, "r4": r4, "r5": r5, "r6": r6}

        # Compute the raw weighted reward.
        raw_reward = sum(self.reward_weights[k] * reward_components[k] for k in reward_components)

        # Normalize the reward by dividing by the sum of the absolute weights.
        normalization_factor = sum(abs(self.reward_weights[k]) for k in reward_components)
        if normalization_factor == 0:
            normalized_reward = raw_reward
        else:
            normalized_reward = raw_reward / normalization_factor

        # Update previous state features for the next move.
        self.prev_foundation_count = current_foundation_count
        self.prev_legal_moves = current_legal_moves
        self.prev_stockpile_count = current_stock_count

        return normalized_reward, reward_components

    def get_state(self):
        """
        Constructs and returns a state representation from the current game.
        
        The state is returned as a dictionary with keys:
          - 'board': A matrix of size (rows x cols) where each slot is encoded either as:
              • a one-hot vector of length 52 (if card_representation == "one_hot")
              • or an integer (the card id; 0 if empty) if using embeddings.
          - 'stock': A vector that represents the stockpile, e.g.:
              • If one_hot: a one-hot vector for the top card (52 dims) concatenated with a scalar count.
              • If embedding: a 2-element vector: [top card id (or 0), count].
          - 'foundation': A list (length 4, one per suit) where each element is encoded similarly.
        
        This method allows the neural network to see all observable information in the game.
        """
        # Board state:
        board_state = []
        card_rep = self.config.get("card_representation", "one_hot")
        for row in self.game.board.slots:
            board_row = []
            for slot in row:
                if slot.has_card():
                    card = slot.card
                    if card_rep == "one_hot":
                        # Create a one-hot vector for this card (length 52).
                        one_hot = [0] * 52
                        one_hot[card.card_id - 1] = 1
                        board_row.append(one_hot)
                    else:
                        # Use card id; for an empty slot we'll use 0.
                        board_row.append(card.card_id)
                else:
                    if card_rep == "one_hot":
                        board_row.append([0] * 52)
                    else:
                        board_row.append(0)
            board_state.append(board_row)
        
        # Stockpile state:
        top_card = self.game.stockpile.top_card()
        count = self.game.stockpile.count()
        if card_rep == "one_hot":
            if top_card is not None:
                one_hot = [0] * 52
                one_hot[top_card.card_id - 1] = 1
            else:
                one_hot = [0] * 52
            stock_state = one_hot + [count]  # concatenate the scalar count.
        else:
            stock_state = [top_card.card_id if top_card is not None else 0, count]
        
        # Foundation state:
        foundation_state = []
        # Assume order of suits is defined in Card.SUITS, e.g., ['s', 'h', 'd', 'c']
        for suit in Card.SUITS:
            current = self.game.foundation.foundation.get(suit)
            if current is not None:
                if card_rep == "one_hot":
                    one_hot = [0] * 52
                    one_hot[current.card_id - 1] = 1
                    foundation_state.append(one_hot)
                else:
                    foundation_state.append(current.card_id)
            else:
                if card_rep == "one_hot":
                    foundation_state.append([0] * 52)
                else:
                    foundation_state.append(0)
        
        state = {
            "board": board_state,
            "stock": stock_state,
            "foundation": foundation_state
        }
        return state
    
    def flatten_state(self, state):
        # Flatten board: assume board is a list of lists
        board_flat = np.array(state["board"]).flatten()
        
        # Stock is assumed to already be a flat vector
        stock_flat = np.array(state["stock"]).flatten()
        
        # Foundation: flatten the list of cards (e.g., one-hot vectors)
        foundation_flat = np.array(state["foundation"]).flatten()
        
        # Concatenate all components into one vector.
        full_state = np.concatenate([board_flat, stock_flat, foundation_flat])
        return full_state
    
    def get_legal_actions(self):
        special_coords, placeable_coords = self.game.board.get_special_slots()
        return list(set( special_coords + placeable_coords))
    
    def get_action(self, network_output):
        """
        Determines the action (coordinate) based on the network's output and a mask generated
        from the board's special slots (legal moves).

        Uses board.get_special_slots() to retrieve:
        - special_coords: Coordinates where a card is present and is lonely or suffocated.
        - placeable_coords: Coordinates where the slot is empty and a card can be placed.

        The union of these two sets defines the legal moves.

        Args:
            network_output (np.array): 1D numpy array of shape (num_slots,) representing raw output.

        Returns:
            A tuple ((row, col), legal_mask) corresponding to the chosen board slot and the boolean mask array.
        """
        # Retrieve special and placeable coordinates.
        special_coords, placeable_coords = self.game.board.get_special_slots()
        # Legal moves are the union of these lists.
        legal_coords = set(special_coords + placeable_coords)
        
        num_slots = self.game.board.rows * self.game.board.cols
        # Create a boolean mask where legal moves are True.
        legal_mask = np.zeros(num_slots, dtype=bool)
        for (r, c) in legal_coords:
            idx = r * self.game.board.cols + c
            legal_mask[idx] = True

        # Define a large penalty to subtract from illegal moves.
        penalty = 1e9
        # For illegal moves, subtract the penalty.
        adjusted_output = np.where(legal_mask, network_output, network_output - penalty)
        
        # Select the index of the highest legal output.
        chosen_index = int(np.argmax(adjusted_output))
        chosen_row = chosen_index // self.game.board.cols
        chosen_col = chosen_index % self.game.board.cols
        return (chosen_row, chosen_col)

    def step(self, action):
        """
        Executes an action (a coordinate tuple) in the game and returns:
        (new_state, reward, done, info)
        
        Steps:
        1. Validate if the slot at 'action' can be changed.
        2. If illegal, return a penalty.
        3. Otherwise, execute the move via game.make_move.
        4. Compute the reward based on the move.
        5. Determine if the game is done (e.g., no legal moves remain).
        6. Get the new state via get_state().
        7. Return the tuple (state, reward, done, info).
        
        :param action: A tuple (row, col) indicating the target slot.
        :return: (state, reward, done, info)
        """
        # Validate the move using board's can_change_slot method.
        if not self.game.board.can_change_slot(action):
            # Illegal move: assign a penalty.
            reward = -0.1
            done = False
            state = self.get_state()
            info = {"illegal_move": True}
            return state, reward, done, info

        # Attempt to execute the move.
        move_executed = self.game.make_move(action)
        if not move_executed:
            # Move execution failed; treat as illegal.
            reward = -0.1
            done = False
            state = self.get_state()
            info = {"illegal_move": True}
            return state, reward, done, info

        # If we reach here, the move was executed successfully.
        reward, reward_components = self.compute_reward()

        # Determine if the game is over.
        done = not self.game.possible_moves_remaining()
        
        # Retrieve the new state.
        state = self.get_state()

        # Return additional info if desired (e.g., reward components for debugging).
        info = {
            "reward_components": reward_components, 
            # "lonely or suffocated cells": self.game.board.get_suffocated_or_lonely_coords(),
            # "placeable cells": self.game.board.get_placeable_coords(),
            # "stockpile size": self.game.stockpile.count(),
            # "cards in foundation": self.game.foundation.total_cards(),
            # "mask": mask.tolist()
        }
        
        return state, reward, done, info

    def conduct_episode(self, max_moves):
        self.reset()

        state = self.get_state()
        board = np.array(state["board"])
        stock = np.array(state["stock"])
        foundation = np.array(state["foundation"])

        rows = self.config["board_rows"]
        cols = self.config["board_cols"]
        card_rep = self.config.get("card_representation", "one_hot")

        if card_rep == "one_hot":
            board_input = board.reshape(1, rows * cols, 52)
            foundation_input = foundation.reshape(1, 4, 52)
        else:
            board_input = board.reshape(1, rows * cols)
            foundation_input = foundation.reshape(1, -1)
        stock_input = stock.reshape(1, -1)
        inputs = [board_input, stock_input, foundation_input]

        move_logs = []

        for i in range(max_moves):
            state_dict = self.get_state()
            board = np.array(state_dict["board"])
            stock = np.array(state_dict["stock"])
            foundation = np.array(state_dict["foundation"])

            if card_rep == "one_hot":
                board_input = board.reshape(1, rows * cols, 52)
                foundation_input = foundation.reshape(1, 4, 52)
            else:
                board_input = board.reshape(1, rows * cols)
                foundation_input = foundation.reshape(1, -1)
            stock_input = stock.reshape(1, -1)
            inputs = [board_input, stock_input, foundation_input]

            network_output = self.network.predict(inputs, verbose=0)[0]

            action = self.get_action(network_output)

            new_state, reward, done, info = self.step(action)
            move_logs.append({
                "move": i + 1,
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
                "cards in stockpile": self.game.stockpile.count(),
                "cards in foundation": self.game.foundation.total_cards(),
            })
            if done:
                break
        return move_logs

    def update_critic(self, value):
        raise NotImplementedError


    def __str__(self):
        return f"CellitaireEnv(game={self.game})"
