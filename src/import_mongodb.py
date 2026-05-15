"""
MongoDB Data Import Script
Import maintenance data from CSV to MongoDB for API testing
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
except ImportError:
    print("ERROR: pymongo not installed. Install it with: pip install pymongo")
    sys.exit(1)

def get_mongo_connection():
    """Get MongoDB connection"""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db = os.getenv("MONGO_DB_NAME", "maintenance_db")
    mongo_collection = os.getenv("MONGO_COLLECTION_NAME", "maintenances")
    
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        db = client[mongo_db]
        collection = db[mongo_collection]
        print(f"✓ Connected to MongoDB")
        print(f"  Database: {mongo_db}")
        print(f"  Collection: {mongo_collection}")
        return client, db, collection
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"ERROR: Failed to connect to MongoDB")
        print(f"  Connection String: {mongo_uri}")
        print(f"  Error: {str(e)}")
        print(f"\nTroubleshooting:")
        print(f"  1. Make sure MongoDB is running")
        print(f"  2. Check connection string in .env file")
        print(f"  3. Verify credentials if using authentication")
        return None, None, None


def import_from_csv(csv_path: str, collection):
    """Import data from CSV to MongoDB"""
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return 0
    
    print(f"\nImporting from: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Convert DataFrame to list of dicts
    records = df.to_dict('records')
    
    # Handle NaN values
    for record in records:
        for key in record:
            if pd.isna(record[key]):
                record[key] = None
    
    # Insert records
    result = collection.insert_many(records)
    
    print(f"✓ Imported {len(result.inserted_ids)} records")
    return len(result.inserted_ids)


def import_sample_data(collection):
    """Import sample maintenance records for testing"""
    print("\nImporting sample test data...")
    
    sample_records = [
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
            "garantie": False,
            "id_equipement": 1,
            "nom_equipement": "Compresseur Principal"
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
            "garantie": True,
            "id_equipement": 2,
            "nom_equipement": "Pompe Hydraulique"
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
            "garantie": False,
            "id_equipement": 3,
            "nom_equipement": "Générateur de Secours"
        },
        {
            "typeMaintenance": "Corrective",
            "priorite": "Haute",
            "type": "Moteur",
            "marque": "Mitsubishi",
            "etat": "En Maintenance",
            "heuresFonctionnement": 12000,
            "equipment_age_days": 1095,
            "nombre_pannes_historique": 3,
            "cout_maintenance_moyen_historique": 3500,
            "dureeEstimee": 5.0,
            "coutEstime": 4200,
            "garantie": False,
            "id_equipement": 4,
            "nom_equipement": "Moteur Principal"
        },
        {
            "typeMaintenance": "Préventive",
            "priorite": "Basse",
            "type": "Convoyeur",
            "marque": "Bosch",
            "etat": "Opérationnel",
            "heuresFonctionnement": 5000,
            "equipment_age_days": 180,
            "nombre_pannes_historique": 0,
            "cout_maintenance_moyen_historique": 1000,
            "dureeEstimee": 1.5,
            "coutEstime": 1200,
            "garantie": True,
            "id_equipement": 5,
            "nom_equipement": "Convoyeur Ligne A"
        }
    ]
    
    result = collection.insert_many(sample_records)
    print(f"✓ Inserted {len(result.inserted_ids)} sample records")
    return len(result.inserted_ids)


def get_collection_stats(collection):
    """Get collection statistics"""
    count = collection.count_documents({})
    print(f"\nCollection Statistics:")
    print(f"  Total documents: {count}")
    
    # Count by maintenance type
    types_count = {}
    for doc in collection.find({}, {"typeMaintenance": 1}):
        maint_type = doc.get("typeMaintenance", "Unknown")
        types_count[maint_type] = types_count.get(maint_type, 0) + 1
    
    if types_count:
        print(f"  By maintenance type:")
        for maint_type, count in types_count.items():
            print(f"    - {maint_type}: {count}")
    
    # Count with predictions
    with_predictions = collection.count_documents({"predicted_cout_maintenance": {"$exists": True}})
    without_predictions = count - with_predictions
    print(f"  With predictions: {with_predictions}")
    print(f"  Without predictions: {without_predictions}")


def clear_collection(collection):
    """Clear all documents from collection"""
    result = collection.delete_many({})
    print(f"✓ Deleted {result.deleted_count} documents")


def main():
    """Main function"""
    print("=" * 70)
    print("MongoDB Data Import Tool for Maintenance Cost Predictor API")
    print("=" * 70)
    
    # Connect to MongoDB
    client, db, collection = get_mongo_connection()
    if client is None:
        sys.exit(1)
    
    # Menu
    while True:
        print("\n" + "=" * 70)
        print("Options:")
        print("  1. Import sample test data")
        print("  2. Import from CSV file")
        print("  3. View collection statistics")
        print("  4. Clear all data")
        print("  5. Exit")
        print("=" * 70)
        
        choice = input("Select option (1-5): ").strip()
        
        if choice == "1":
            # Import sample data
            count = import_sample_data(collection)
            get_collection_stats(collection)
            
        elif choice == "2":
            # Import from CSV
            csv_path = input("Enter CSV file path: ").strip()
            if csv_path:
                count = import_from_csv(csv_path, collection)
                get_collection_stats(collection)
            
        elif choice == "3":
            # View statistics
            get_collection_stats(collection)
            
        elif choice == "4":
            # Clear collection
            confirm = input("Are you sure? Type 'yes' to confirm: ").strip().lower()
            if confirm == "yes":
                clear_collection(collection)
            
        elif choice == "5":
            # Exit
            print("Exiting...")
            break
        
        else:
            print("Invalid option. Please try again.")
    
    # Close connection
    client.close()
    print("✓ Connection closed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)
