"""
Python client for Maintenance Cost Prediction API
Usage examples and helper functions for API integration
"""

import requests
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

class MaintenanceAPIClient:
    """Client for Maintenance Cost Prediction API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client
        
        Args:
            base_url: API base URL (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ML model information"""
        response = self.session.get(f"{self.base_url}/model-info")
        response.raise_for_status()
        return response.json()
    
    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict maintenance cost for a single record
        
        Args:
            record: Maintenance record data
        
        Returns:
            Prediction result with cost, difference, and timestamp
        """
        response = self.session.post(
            f"{self.base_url}/predict",
            json=record
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict for multiple records and auto-save to MongoDB
        
        Args:
            records: List of maintenance records
        
        Returns:
            Batch prediction summary
        """
        response = self.session.post(
            f"{self.base_url}/predict-batch",
            json=records
        )
        response.raise_for_status()
        return response.json()
    
    def retrieve_pending(self) -> Dict[str, Any]:
        """
        Retrieve and process pending predictions from MongoDB
        Returns immediately, processing happens in background
        
        Returns:
            Status of pending records processing
        """
        response = self.session.get(f"{self.base_url}/retrieve-pending")
        response.raise_for_status()
        return response.json()
    
    def sync_all_predictions(self) -> Dict[str, Any]:
        """
        Sync all records - find and predict for those without predictions
        
        Returns:
            Sync status with counts
        """
        response = self.session.post(f"{self.base_url}/sync-predictions")
        response.raise_for_status()
        return response.json()
    
    def get_recent_predictions(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get recent predictions from MongoDB
        
        Args:
            limit: Number of predictions to retrieve
        
        Returns:
            List of predictions
        """
        response = self.session.get(f"{self.base_url}/predictions/{limit}")
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close the session"""
        self.session.close()


def example_basic_prediction():
    """Example: Single record prediction"""
    print("=" * 70)
    print("EXAMPLE 1: Single Record Prediction")
    print("=" * 70)
    
    client = MaintenanceAPIClient()
    
    # Create a maintenance record
    record = {
        "typeMaintenance": "Corrective",
        "priorite": "Critique",
        "type": "Compresseur",
        "marque": "Siemens",
        "etat": "En Maintenance",
        "heuresFonctionnement": 15000,
        "equipment_age_days": 730,
        "nombre_pannes_historique": 5,
        "cout_maintenance_moyen_historique": 5000,
        "dureeEstimee": 6.5,
        "coutEstime": 6500,
        "garantie": False
    }
    
    # Get prediction
    result = client.predict_single(record)
    
    print(f"\nEquipment Type: {record['type']} ({record['marque']})")
    print(f"Maintenance Type: {record['typeMaintenance']}")
    print(f"Priority: {record['priorite']}")
    print(f"\nEstimated Cost: ${result['estimated_cost']:.2f}")
    print(f"Predicted Cost: ${result['predicted_cost']:.2f}")
    print(f"Difference: ${result['difference']:.2f} ({result['percentage_difference']:+.1f}%)")
    print(f"\nPrediction Time: {result['prediction_timestamp']}")
    
    client.close()


def example_batch_prediction():
    """Example: Batch prediction with auto-save to MongoDB"""
    print("=" * 70)
    print("EXAMPLE 2: Batch Prediction with Auto-Save to MongoDB")
    print("=" * 70)
    
    client = MaintenanceAPIClient()
    
    # Create multiple maintenance records
    records = [
        {
            "typeMaintenance": "Corrective",
            "priorite": "Critique",
            "type": "Compresseur",
            "marque": "Siemens",
            "etat": "En Maintenance",
            "heuresFonctionnement": 15000,
            "equipment_age_days": 730,
            "nombre_pannes_historique": 5,
            "cout_maintenance_moyen_historique": 5000,
            "dureeEstimee": 6.5,
            "coutEstime": 6500,
            "garantie": False
        },
        {
            "typeMaintenance": "Préventive",
            "priorite": "Moyenne",
            "type": "Pompe",
            "marque": "ABB",
            "etat": "Opérationnel",
            "heuresFonctionnement": 8000,
            "equipment_age_days": 365,
            "nombre_pannes_historique": 1,
            "cout_maintenance_moyen_historique": 2000,
            "dureeEstimee": 2.5,
            "coutEstime": 2200,
            "garantie": True
        },
        {
            "typeMaintenance": "Prédictive",
            "priorite": "Haute",
            "type": "Générateur",
            "marque": "Schneider",
            "etat": "Opérationnel",
            "heuresFonctionnement": 25000,
            "equipment_age_days": 1460,
            "nombre_pannes_historique": 8,
            "cout_maintenance_moyen_historique": 7500,
            "dureeEstimee": 4.0,
            "coutEstime": 8000,
            "garantie": False
        }
    ]
    
    print(f"\nProcessing {len(records)} maintenance records...")
    print("(Predictions will be automatically saved to MongoDB)")
    
    # Get batch predictions (auto-saves in background)
    result = client.predict_batch(records)
    
    print(f"\n✓ API Response Received (Non-blocking)")
    print(f"  Total Records: {result['total_records']}")
    print(f"  Successful Predictions: {result['successful_predictions']}")
    print(f"  Failed: {result['failed_predictions']}")
    print(f"  Average Predicted Cost: ${result['average_predicted_cost']:.2f}")
    print(f"\n✓ Background tasks queued for MongoDB storage")
    print(f"  Timestamp: {result['timestamp']}")
    
    client.close()


def example_retrieve_and_process():
    """Example: Retrieve pending records and process automatically"""
    print("=" * 70)
    print("EXAMPLE 3: Retrieve & Process Pending Records")
    print("=" * 70)
    
    client = MaintenanceAPIClient()
    
    print("\nRetrieving pending records from MongoDB...")
    print("(Will process without user interruption)")
    
    result = client.retrieve_pending()
    
    print(f"\n✓ Operation queued successfully")
    print(f"  Records to process: {result['processed']}")
    print(f"  Timestamp: {result['timestamp']}")
    print(f"\n  Note: Background tasks are running predictions and saving to MongoDB")
    
    client.close()


def example_sync_all():
    """Example: Sync all records in database"""
    print("=" * 70)
    print("EXAMPLE 4: Sync All Database Records")
    print("=" * 70)
    
    client = MaintenanceAPIClient()
    
    print("\nSyncing all maintenance records...")
    
    result = client.sync_all_predictions()
    
    print(f"\n✓ Sync operation complete")
    print(f"  Total records: {result['total_records']}")
    print(f"  Already predicted: {result['already_predicted']}")
    print(f"  Queued for prediction: {result['queued_for_prediction']}")
    print(f"  Timestamp: {result['timestamp']}")
    
    client.close()


def example_get_recent_predictions():
    """Example: Retrieve recent predictions from MongoDB"""
    print("=" * 70)
    print("EXAMPLE 5: Get Recent Predictions from MongoDB")
    print("=" * 70)
    
    client = MaintenanceAPIClient()
    
    print("\nRetrieving 10 most recent predictions...")
    
    result = client.get_recent_predictions(limit=10)
    
    print(f"\nFound {result['count']} predictions:\n")
    
    for i, pred in enumerate(result['predictions'], 1):
        print(f"{i}. {pred.get('type', 'N/A')} - {pred.get('marque', 'N/A')}")
        print(f"   Estimated: ${pred.get('coutEstime', 0):.2f}")
        print(f"   Predicted: ${pred.get('predicted_cout_maintenance', 0):.2f}")
        print(f"   Saved: {pred.get('prediction_timestamp', 'N/A')}\n")
    
    client.close()


def example_full_workflow():
    """Example: Complete workflow"""
    print("=" * 70)
    print("EXAMPLE 6: Complete Workflow")
    print("=" * 70)
    
    client = MaintenanceAPIClient()
    
    # Step 1: Check API health
    print("\nStep 1: Checking API health...")
    health = client.health_check()
    print(f"✓ API Status: {health['status']}")
    print(f"✓ Model Available: {health['model_available']}")
    print(f"✓ Database Available: {health['database_available']}")
    
    # Step 2: Get model info
    print("\nStep 2: Getting model information...")
    model_info = client.get_model_info()
    print(f"✓ Model: {model_info['model']}")
    print(f"✓ R² Score: {model_info['r2_score']:.4f}")
    print(f"✓ MAE: ${model_info['mae']:.2f}")
    
    # Step 3: Process pending records
    print("\nStep 3: Processing pending records from database...")
    pending_result = client.retrieve_pending()
    print(f"✓ Queued {pending_result['processed']} records for processing")
    
    # Step 4: Get recent predictions
    print("\nStep 4: Getting recent predictions...")
    predictions = client.get_recent_predictions(limit=5)
    print(f"✓ Retrieved {predictions['count']} recent predictions")
    
    print("\n" + "=" * 70)
    print("✓ Complete workflow executed successfully!")
    print("=" * 70)
    
    client.close()


if __name__ == "__main__":
    try:
        # Run all examples
        example_basic_prediction()
        print("\n")
        
        example_batch_prediction()
        print("\n")
        
        example_retrieve_and_process()
        print("\n")
        
        example_sync_all()
        print("\n")
        
        example_get_recent_predictions()
        print("\n")
        
        example_full_workflow()
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API")
        print("Make sure the API is running: python -m uvicorn src.api:app --reload")
    except Exception as e:
        print(f"ERROR: {str(e)}")
