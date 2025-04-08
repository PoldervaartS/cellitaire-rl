# storage/model_storage.py

import os
from uuid import uuid4
from .storage import create_metadata, get_metadata, delete_metadata, update_metadata
# or use your framework's loading method
from tensorflow.keras.models import load_model

# Define a directory to store model files
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models_files")
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)


class ModelStorage:
    @staticmethod
    def store_model(model_object, parameters=None, metrics=None):
        # Generate a unique model_id
        model_id = str(uuid4())
        file_name = f"{model_id}.h5"
        file_path = os.path.join(MODELS_DIR, file_name)

        # Save the model to disk
        model_object.save(file_path)

        # Store metadata about the model
        create_metadata(
            model_id=model_id,
            file_path=file_path,
            parameters=parameters,
            metrics=metrics
        )
        return model_id

    @staticmethod
    def retrieve_model(model_id):
        # Retrieve metadata from the database
        metadata = get_metadata(model_id)
        if not metadata:
            raise FileNotFoundError("Model metadata not found.")

        file_path = metadata.file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError("Model file not found.")

        # Load the model from disk
        model_object = load_model(file_path)
        return model_object, metadata

    @staticmethod
    def delete_model(model_id):
        metadata = get_metadata(model_id)
        if not metadata:
            return False

        file_path = metadata.file_path
        # Delete the model file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)

        # Delete the metadata record
        return delete_metadata(model_id)

    @staticmethod
    def update_model(model_id, new_model_object=None,
                     parameters=None, metrics=None):
        metadata = get_metadata(model_id)
        if not metadata:
            raise FileNotFoundError("Model metadata not found for update.")

        # Overwrite the model file if a new model is provided.
        if new_model_object is not None:
            new_model_object.save(metadata.file_path)

        # Update the metadata record using our helper function.
        updated_metadata = update_metadata(
            model_id, parameters=parameters, metrics=metrics)
        if updated_metadata is None:
            raise Exception("Failed to update metadata.")
        return updated_metadata
