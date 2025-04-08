import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Load environment variables from the .env file
load_dotenv()

# Retrieve the database URL from the environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the environment variables")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class ModelMetadata(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, unique=True, index=True, nullable=False)
    file_path = Column(Text, nullable=False)
    parameters = Column(JSON)
    metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


# This will create the "models" table in your database if it doesn't exist
Base.metadata.create_all(bind=engine)


def create_metadata(model_id: str, file_path: str,
                    parameters: dict = None, metrics: dict = None):
    db = SessionLocal()
    try:
        metadata = ModelMetadata(
            model_id=model_id,
            file_path=file_path,
            parameters=parameters,
            metrics=metrics
        )
        db.add(metadata)
        db.commit()
        db.refresh(metadata)
        return metadata
    finally:
        db.close()


def get_metadata(model_id: str):
    db = SessionLocal()
    try:
        metadata = db.query(ModelMetadata).filter(
            ModelMetadata.model_id == model_id).first()
        return metadata
    finally:
        db.close()


def delete_metadata(model_id: str):
    db = SessionLocal()
    try:
        metadata = db.query(ModelMetadata).filter(
            ModelMetadata.model_id == model_id).first()
        if metadata:
            db.delete(metadata)
            db.commit()
            return True
        return False
    finally:
        db.close()


def update_metadata(model_id: str, parameters: dict = None,
                    metrics: dict = None):
    db = SessionLocal()
    try:
        metadata = db.query(ModelMetadata).filter(
            ModelMetadata.model_id == model_id).first()
        if metadata is None:
            return None
        if parameters is not None:
            metadata.parameters = parameters
        if metrics is not None:
            metadata.metrics = metrics
        db.commit()
        db.refresh(metadata)
        return metadata
    finally:
        db.close()
