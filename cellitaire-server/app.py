# app.py
from fastapi import FastAPI, HTTPException
import uuid
import traceback
from storage.storage import create_metadata, get_metadata, delete_metadata

# Import our combined model storage service.
from storage.model_storage import ModelStorage

# Import TensorFlow/Keras to create a dummy model for testing.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

app = FastAPI()

# app.py (or a separate file for your API endpoints)
from fastapi import FastAPI
import numpy as np
from game.cellitaire_env import CellitaireEnv

app = FastAPI()

@app.get("/test_env")
def test_env(num_moves: int = 15):
    try:
        # 1. Create a configuration dictionary with defaults.
        config = {
            "board_rows": 7,
            "board_cols": 12,
            "initial_reserved": 6,
            "card_representation": "one_hot",  # or "embedding"
            "num_reward_features": 6,
            "embedding_dim": 10,
            "board_rnn_units": 64,
            "stock_hidden_size": 32,
            "foundation_hidden_size": 32,
            "hidden_layer_sizes": [64, 32],
            "reward_weights": {
                "r1": 1.0,
                "r2": 1.0,
                "r3": 1.0,
                "r4": 1.0,
                "r5": 1.0,
                "r6": 1.0
            }
        }
        
        # 2. Create a new environment and reset it.
        env = CellitaireEnv(config)
        env.reset()  # Uses defaults from config.
        
        # 3. Retrieve the state and verify dimensions.
        state = env.get_state()
        board = np.array(state["board"])
        stock = np.array(state["stock"])
        foundation = np.array(state["foundation"])
        
        rows = config["board_rows"]
        cols = config["board_cols"]
        card_rep = config.get("card_representation", "one_hot")
        
        # Verify board dimensions.
        if card_rep == "one_hot":
            expected_board_shape = (rows, cols, 52)
            if board.shape != expected_board_shape:
                raise Exception(f"Board shape incorrect: expected {expected_board_shape}, got {board.shape}")
        else:
            expected_board_shape = (rows, cols)
            if board.shape != expected_board_shape:
                raise Exception(f"Board shape incorrect: expected {expected_board_shape}, got {board.shape}")
        
        # Verify stock state dimensions.
        if card_rep == "one_hot":
            if stock.shape[0] != 53:
                raise Exception(f"Stock state dimension incorrect: expected 53, got {stock.shape[0]}")
        else:
            if stock.shape[0] != 2:
                raise Exception(f"Stock state dimension incorrect: expected 2, got {stock.shape[0]}")
        
        # Verify foundation state dimensions.
        if card_rep == "one_hot":
            if foundation.shape != (4, 52):
                raise Exception(f"Foundation state dimension incorrect: expected (4, 52), got {foundation.shape}")
        else:
            if foundation.shape[0] != 4:
                raise Exception(f"Foundation state dimension incorrect: expected 4, got {foundation.shape[0]}")
        
        # 4. Prepare inputs for the network.
        # The model expects three separate inputs: board, stock, foundation.
        # For the board, we add a batch dimension and reshape to (1, rows*cols, 52).
        if card_rep == "one_hot":
            board_input = board.reshape(1, rows * cols, 52)
            foundation_input = foundation.reshape(1, 4, 52)
        else:
            board_input = board.reshape(1, rows * cols)
            foundation_input = foundation.reshape(1, -1)
        stock_input = stock.reshape(1, -1)
        inputs = [board_input, stock_input, foundation_input]
        
        # 5. Test a series of moves.
        move_logs = []
        for i in range(num_moves):
            # Get current state.
            state_dict = env.get_state()
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
            
            # Get network output.
            network_output = env.network.predict(inputs)[0]  # Shape: (rows*cols,)
            
            # Determine action using our get_action function.
            action = env.get_action(network_output)
            
            # Execute the step.
            new_state, reward, done, info = env.step(action)
            move_logs.append({
                "move": i + 1,
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
                "cards in stockpile": env.game.stockpile.count(),
                "cards in foundation": env.game.foundation.total_cards(),
            })
            if done:
                break
        
        return {"passed": True, "move_logs": move_logs}
    
    except Exception as e:
        return {"passed": False, "error": str(e), "trace": traceback.format_exc()}

@app.get("/test_model_storage")
def test_model_storage():
    # Create a simple dummy model for testing purposes.
    try:
        model = Sequential([
            Dense(10, activation='relu', input_shape=(20,)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')

        # Use the ModelStorage service to store the model.
        # Optionally, supply some dummy training parameters and metrics.
        model_id = ModelStorage.store_model(
            model_object=model,
            parameters={"epochs": 1, "batch_size": 32},
            metrics={"accuracy": 0.5}
        )

        # Retrieve the stored model and its metadata.
        retrieved_model, metadata = ModelStorage.retrieve_model(model_id)

        # Delete the stored model and metadata.
        delete_success = ModelStorage.delete_model(model_id)

        # Prepare a dictionary from the metadata for display.
        metadata_dict = {
            "model_id": metadata.model_id,
            "file_path": metadata.file_path,
            "parameters": metadata.parameters,
            "metrics": metadata.metrics,
            "created_at": str(metadata.created_at)
        }

        return {
            "success": True,
            "model_id": model_id,
            "deleted": delete_success,
            "metadata": metadata_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test_metadata")
def test_metadata():
    # Generate random metadata values
    random_model_id = str(uuid.uuid4())
    file_path = f"/models/{random_model_id}.h5"
    parameters = {"epochs": 10, "batch_size": 32}
    metrics = {"accuracy": 0.85}

    try:
        # Create the metadata record
        created = create_metadata(
            model_id=random_model_id,
            file_path=file_path,
            parameters=parameters,
            metrics=metrics
        )
        if not created:
            return {"success": False, "error": "Failed to create metadata."}

        # Retrieve the metadata record
        retrieved = get_metadata(random_model_id)
        if not retrieved:
            return {"success": False, "error": "Failed to retrieve metadata."}

        # Delete the metadata record
        deleted = delete_metadata(random_model_id)
        if not deleted:
            return {"success": False, "error": "Failed to delete metadata."}

        # If all operations succeed, return a success response
        return {"success": True}

    except Exception as e:
        # Raise an HTTP exception if anything goes wrong
        raise HTTPException(status_code=500, detail=str(e))
