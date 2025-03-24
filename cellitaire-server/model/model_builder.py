# model/model_builder.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Embedding, GRU
from tensorflow.keras.models import Model

class ModelBuilder:
    def __init__(self, config):
        """
        Initializes the ModelBuilder with a configuration dictionary.
        
        Expected config keys (with example values):
          - card_representation: "one_hot" or "embedding"
          - board_rows: 7
          - board_cols: 12
          - embedding_dim: 10  (only used if card_representation is "embedding")
          - board_rnn_units: 64  (units in the RNN processing the board)
          - stock_hidden_size: 32  (size of a dense layer for stock input)
          - foundation_hidden_size: 32  (size of a dense layer for foundation input)
          - hidden_layer_sizes: [64, 32]  (two hidden dense layers after concatenation)
        """
        self.config = config
        
    def build_model(self):
        cfg = self.config
        
        board_rows = cfg["board_rows"]
        board_cols = cfg["board_cols"]
        board_length = board_rows * board_cols
        
        card_rep = cfg.get("card_representation", "one_hot")
        
        # --- BOARD INPUT ---
        # For the board, we have board_length slots.
        # If one_hot: each slot is a vector of length 52.
        # If embedding: each slot is an integer (card id) and we use an Embedding layer.
        if card_rep == "one_hot":
            board_feature_dim = 52  # fixed for a 52-card deck.
            board_input = Input(shape=(board_length, board_feature_dim), name="board_input")
            # Process board input with an RNN (e.g., GRU). Return sequences.
            board_rnn = GRU(cfg.get("board_rnn_units", 64), return_sequences=True, name="board_rnn")(board_input)
            # Flatten the RNN output.
            board_features = Flatten(name="board_flatten")(board_rnn)
        elif card_rep == "embedding":
            board_input = Input(shape=(board_length,), dtype='int32', name="board_input")
            # Embedding layer to convert card IDs to dense vectors.
            emb_dim = cfg.get("embedding_dim", 10)
            board_emb = Embedding(input_dim=53, output_dim=emb_dim, name="board_embedding")(board_input)
            # Process with RNN.
            board_rnn = GRU(cfg.get("board_rnn_units", 64), return_sequences=True, name="board_rnn")(board_emb)
            board_features = Flatten(name="board_flatten")(board_rnn)
        else:
            raise ValueError("Unknown card representation method")
        
        # --- STOCK PILE INPUT ---
        # Stock input: Represented as top card + number of cards remaining.
        # For one_hot: top card is one-hot (52 dim), plus a scalar.
        if card_rep == "one_hot":
            stock_top_dim = 52
            stock_input = Input(shape=(stock_top_dim + 1,), name="stock_input")
        elif card_rep == "embedding":
            # For embedding, top card is an integer, and the remaining count is a scalar.
            stock_input = Input(shape=(2,), dtype='int32', name="stock_input")
            # We'll embed the top card later in the processing branch.
        # Process the stock input with a dense layer.
        stock_hidden = Dense(cfg.get("stock_hidden_size", 32), activation="relu", name="stock_dense")(stock_input)
        
        # --- FOUNDATION INPUT ---
        # Foundation input: Represented as highest card for each of the 4 suits.
        # For one_hot: shape is (4, 52), then flatten.
        if card_rep == "one_hot":
            foundation_input = Input(shape=(4, 52), name="foundation_input")
            foundation_flat = Flatten(name="foundation_flatten")(foundation_input)
        elif card_rep == "embedding":
            # For embedding, foundation input is 4 integers.
            foundation_input = Input(shape=(4,), dtype='int32', name="foundation_input")
            # Embed each integer.
            emb_dim = cfg.get("embedding_dim", 10)
            foundation_emb = Embedding(input_dim=53, output_dim=emb_dim, name="foundation_embedding")(foundation_input)
            foundation_flat = Flatten(name="foundation_flatten")(foundation_emb)
        
        # --- COMBINE INPUTS ---
        # Concatenate board, stock, and foundation features.
        combined = Concatenate(name="combined_features")([board_features, stock_hidden, foundation_flat])
        
        # Two hidden dense layers.
        hidden1 = Dense(cfg["hidden_layer_sizes"][0], activation="relu", name="hidden1")(combined)
        hidden2 = Dense(cfg["hidden_layer_sizes"][1], activation="relu", name="hidden2")(hidden1)
        
        # --- OUTPUT LAYER ---
        # Output one node per board slot.
        output_dim = board_length
        output = Dense(output_dim, activation="linear", name="output")(hidden2)
        
        # Build the model. We combine all inputs into one model.
        if card_rep == "one_hot":
            inputs = [board_input, stock_input, foundation_input]
        elif card_rep == "embedding":
            inputs = [board_input, stock_input, foundation_input]
        
        model = Model(inputs=inputs, outputs=output, name="CellitaireModel")
        model.compile(optimizer="adam", loss="mse")  # For example, using MSE loss for value regression.
        model.summary()  # Helpful for debugging.
        return model
