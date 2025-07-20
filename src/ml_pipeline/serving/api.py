from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
from ...feature_engineering.feature_store import FeatureStore
from ...ml_pipeline.utils.data_utils import DataUtils
import numpy as np

router = APIRouter()
model = None
feature_store = FeatureStore()
data_utils = DataUtils()
mlflow_client = MlflowClient()

class PredictionRequest(BaseModel):
    features: list[float] # Expects a list of numerical features

@router.on_event("startup")
async def startup_event():
    """Load the best model from MLflow Model Registry on startup"""
    global model
    try:
        # Load the model from the "Production" stage
        # Replace 'random_forest' with the actual model name you want to serve
        model_name = "random_forest"
        
        # Get the latest version of the model in the "Production" stage
        latest_version = mlflow_client.get_latest_versions(model_name, stages=["Production"])
        if not latest_version:
            print(f"No model found for '{model_name}' in 'Production' stage. Loading latest available.")
            latest_version = mlflow_client.get_latest_versions(model_name, stages=["None"]) # Fallback to latest
            if not latest_version:
                print(f"No model found for '{model_name}' at all. Please train and register a model.")
                return
        
        model_uri = f"models:/{model_name}/{latest_version[0].version}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded model: {model_uri}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None # Ensure model is None if loading fails

@router.post("/predict")
async def predict(request: PredictionRequest):
    """Make predictions using the loaded model"""
    if model is None:
        return {"prediction": "Error: Model not loaded. Please check server logs."}
    
    try:
        # Convert incoming features to a DataFrame
        # The number of features should match the training data
        # For simplicity, assuming a single row of features
        input_df = pd.DataFrame([request.features])
        
        # In a real application, the scaler used during training should be saved and loaded
        # to ensure consistent scaling for inference. For this example, we'll pass raw features.
        prediction = model.predict(input_df.values)
        
        # Convert prediction to a list for JSON response
        return {"prediction": prediction.tolist()}
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"prediction": f"Error during prediction: {e}"}