"""
FastAPI application for Maintenance Cost Prediction
Automatically retrieves data from MongoDB, runs predictions, and saves results
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "maintenance_db")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "maintenances")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Initialize FastAPI app
app = FastAPI(
    title="Maintenance Cost Prediction API",
    description="API for predicting maintenance costs using ML model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessing objects
model = None
scaler = None
label_encoders = None
model_summary = None
mongo_client = None
db = None

# Pydantic models for API
class MaintenanceRecord(BaseModel):
    """Schema for individual maintenance record input"""
    typeMaintenance: str
    priorite: str
    type: str
    marque: str
    etat: str
    heuresFonctionnement: float
    equipment_age_days: float
    nombre_pannes_historique: float
    cout_maintenance_moyen_historique: float
    dureeEstimee: float
    coutEstime: float
    garantie: bool
    _id: Optional[str] = Field(None, alias="_id")

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    _id: Optional[str]
    predicted_cost: float
    estimated_cost: float
    difference: float
    percentage_difference: float
    prediction_timestamp: str
    model_confidence: Optional[float] = None


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    total_records: int
    successful_predictions: int
    failed_predictions: int
    average_predicted_cost: float
    timestamp: str


def init_models():
    """Initialize ML model and preprocessing objects"""
    global model, scaler, label_encoders, model_summary
    
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(os.path.dirname(script_dir), "models")
        
        # Try to load model files
        model_path = os.path.join(models_dir, "best_maintenance_cost_model.pkl")
        scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
        encoders_path = os.path.join(models_dir, "label_encoders.pkl")
        summary_path = os.path.join(models_dir, "model_summary.json")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}")
            return False
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        label_encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else None
        
        with open(summary_path, 'r') as f:
            model_summary = json.load(f)
        
        logger.info("✓ ML models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False


def init_mongodb():
    """Initialize MongoDB connection"""
    global mongo_client, db
    
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Test connection
        mongo_client.admin.command('ping')
        db = mongo_client[MONGO_DB_NAME]
        logger.info(f"✓ Connected to MongoDB: {MONGO_DB_NAME}")
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        logger.info("Note: Make sure MongoDB is running and connection string in .env is correct")
        return False
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        return False


def preprocess_record(record: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Preprocess a maintenance record for prediction
    Returns DataFrame ready for model prediction
    """
    try:
        df = pd.DataFrame([record])
        
        # Feature engineering
        df['is_critical_priority'] = (df['priorite'] == 'Critique').astype(int)
        df['is_corrective'] = (df['typeMaintenance'] == 'Corrective').astype(int)
        df['is_preventive'] = (df['typeMaintenance'] == 'Préventive').astype(int)
        df['has_garantie'] = df['garantie'].astype(int)
        df['equipment_age_years'] = df['equipment_age_days'] / 365.25
        df['hours_per_day'] = df['heuresFonctionnement'] / (df['equipment_age_days'] + 1)
        
        # Encode categorical variables
        if label_encoders:
            for col in ['typeMaintenance', 'priorite', 'type', 'marque', 'etat']:
                if col in label_encoders and col in df.columns:
                    df[f'{col}_encoded'] = label_encoders[col].transform(df[col].astype(str))
        
        # Handle missing values
        df = df.fillna(df.median())
        
        return df
    except Exception as e:
        logger.error(f"Error preprocessing record: {str(e)}")
        return None


def make_prediction(record: Dict[str, Any]) -> Optional[float]:
    """
    Make prediction for a single maintenance record
    Returns predicted cost or None if prediction fails
    """
    try:
        if model is None or model_summary is None:
            logger.error("Model not initialized")
            return None
        
        df = preprocess_record(record)
        if df is None:
            return None
        
        # Select features in the same order as training
        feature_columns = model_summary['features_used']
        X = df[feature_columns]
        
        # Scale if needed
        if model_summary['best_model'] in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            if scaler is None:
                logger.warning("Scaler not available for linear model")
                X_scaled = X
            else:
                X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
        else:
            prediction = model.predict(X)[0]
        
        return float(prediction)
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None


async def predict_and_save_to_db(record_id: Any, record: Dict[str, Any]):
    """
    Background task to make prediction and save to MongoDB
    """
    try:
        prediction = make_prediction(record)
        if prediction is None:
            logger.error(f"Failed to predict for record {record_id}")
            return
        
        # Calculate difference
        estimated_cost = record.get('coutEstime', 0)
        difference = prediction - estimated_cost
        percentage_diff = (difference / estimated_cost * 100) if estimated_cost > 0 else 0
        
        # Update record in MongoDB
        if db is not None:
            collection = db[MONGO_COLLECTION_NAME]
            result = collection.update_one(
                {"_id": record_id},
                {
                    "$set": {
                        "predicted_cout_maintenance": prediction,
                        "difference_from_estimate": difference,
                        "percentage_difference": percentage_diff,
                        "prediction_timestamp": datetime.utcnow().isoformat(),
                        "model_used": model_summary['best_model'],
                        "model_r2_score": model_summary['r2_score']
                    }
                }
            )
            logger.info(f"✓ Prediction saved for record {record_id}")
            return result
    except Exception as e:
        logger.error(f"Error in predict_and_save_to_db: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize models and database on startup"""
    logger.info("Starting up Maintenance Cost Prediction API...")
    
    # Load ML models
    if not init_models():
        logger.error("Failed to load ML models. API will run but predictions may fail.")
    
    # Connect to MongoDB
    if not init_mongodb():
        logger.warning("MongoDB connection failed. Some endpoints will not work.")
    
    logger.info("Startup complete!")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global mongo_client
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API status"""
    return {
        "status": "running",
        "service": "Maintenance Cost Prediction API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "database_connected": db is not None
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_available": model is not None,
        "database_available": db is not None
    }


@app.get("/model-info", tags=["Model"])
async def get_model_info():
    """Get information about the loaded ML model"""
    if model_summary is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model": model_summary.get('best_model', 'Unknown'),
        "r2_score": model_summary.get('r2_score', 0),
        "mae": model_summary.get('mae', 0),
        "rmse": model_summary.get('rmse', 0),
        "features_count": len(model_summary.get('features_used', [])),
        "features_used": model_summary.get('features_used', [])
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(record: MaintenanceRecord):
    """Predict maintenance cost for a single record"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    record_dict = record.dict(exclude_none=True, by_alias=True)
    
    prediction = make_prediction(record_dict)
    if prediction is None:
        raise HTTPException(status_code=400, detail="Failed to make prediction")
    
    estimated_cost = record.coutEstime
    difference = prediction - estimated_cost
    percentage_diff = (difference / estimated_cost * 100) if estimated_cost > 0 else 0
    
    return PredictionResponse(
        predicted_cost=prediction,
        estimated_cost=estimated_cost,
        difference=difference,
        percentage_difference=percentage_diff,
        prediction_timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(records: List[MaintenanceRecord], background_tasks: BackgroundTasks):
    """
    Predict maintenance costs for multiple records and automatically save to MongoDB
    """
    if model is None or db is None:
        raise HTTPException(status_code=503, detail="Model or Database not available")
    
    if len(records) == 0:
        raise HTTPException(status_code=400, detail="No records provided")
    
    successful = 0
    failed = 0
    predictions = []
    
    try:
        collection = db[MONGO_COLLECTION_NAME]
        
        for record in records:
            try:
                record_dict = record.dict(exclude_none=True, by_alias=True)
                record_id = record_dict.get('_id')
                
                prediction = make_prediction(record_dict)
                if prediction is None:
                    failed += 1
                    continue
                
                predictions.append(prediction)
                successful += 1
                
                # Add background task to save prediction to MongoDB
                background_tasks.add_task(predict_and_save_to_db, record_id, record_dict)
                
            except Exception as e:
                logger.error(f"Error processing record: {str(e)}")
                failed += 1
                continue
        
        avg_cost = float(np.mean(predictions)) if predictions else 0
        
        return BatchPredictionResponse(
            total_records=len(records),
            successful_predictions=successful,
            failed_predictions=failed,
            average_predicted_cost=avg_cost,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retrieve-pending", tags=["Database"])
async def retrieve_pending_predictions(background_tasks: BackgroundTasks):
    """
    Retrieve maintenance records from MongoDB that need predictions and process them
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        collection = db[MONGO_COLLECTION_NAME]
        
        # Find records without predictions
        pending_records = list(collection.find(
            {"predicted_cout_maintenance": {"$exists": False}},
            limit=100
        ))
        
        if not pending_records:
            return {
                "message": "No pending records found",
                "processed": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Process each record
        for record in pending_records:
            try:
                # Create a dict without MongoDB's _id for processing
                record_data = {k: v for k, v in record.items() if k != '_id'}
                background_tasks.add_task(predict_and_save_to_db, record['_id'], record_data)
            except Exception as e:
                logger.error(f"Error processing record {record.get('_id')}: {str(e)}")
        
        return {
            "message": f"Processing {len(pending_records)} pending records",
            "processed": len(pending_records),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving pending records: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/{limit}", tags=["Database"])
async def get_predictions(limit: int = 10):
    """
    Retrieve recent predictions from MongoDB
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        collection = db[MONGO_COLLECTION_NAME]
        predictions = list(collection.find(
            {"predicted_cout_maintenance": {"$exists": True}},
            sort=[("prediction_timestamp", -1)],
            limit=limit
        ))
        
        # Convert ObjectId to string for JSON serialization
        for pred in predictions:
            if '_id' in pred:
                pred['_id'] = str(pred['_id'])
        
        return {
            "count": len(predictions),
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Error retrieving predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sync-predictions", tags=["Database"])
async def sync_all_predictions(background_tasks: BackgroundTasks):
    """
    Sync all records from MongoDB - retrieve and predict for those without predictions
    """
    if db is None or model is None:
        raise HTTPException(status_code=503, detail="Database or Model not available")
    
    try:
        collection = db[MONGO_COLLECTION_NAME]
        all_records = list(collection.find())
        
        pending_count = 0
        existing_predictions = 0
        
        for record in all_records:
            if "predicted_cout_maintenance" in record:
                existing_predictions += 1
            else:
                try:
                    record_data = {k: v for k, v in record.items() if k != '_id'}
                    background_tasks.add_task(predict_and_save_to_db, record['_id'], record_data)
                    pending_count += 1
                except Exception as e:
                    logger.error(f"Error queuing record {record.get('_id')}: {str(e)}")
        
        return {
            "total_records": len(all_records),
            "already_predicted": existing_predictions,
            "queued_for_prediction": pending_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in sync_all_predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
