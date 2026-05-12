import pandas as pd
import numpy as np
import joblib
import json

print("="*70)
print("MAINTENANCE COST PREDICTION SYSTEM")
print("="*70)

# Load trained model and preprocessing objects
print("\n1. Loading trained model and preprocessing objects...")
model = joblib.load('/mnt/user-data/outputs/best_maintenance_cost_model.pkl')
scaler = joblib.load('/mnt/user-data/outputs/feature_scaler.pkl')
label_encoders = joblib.load('/mnt/user-data/outputs/label_encoders.pkl')

# Load model summary
with open('/mnt/user-data/outputs/model_summary.json', 'r') as f:
    model_summary = json.load(f)

print(f"✓ Model loaded: {model_summary['best_model']}")
print(f"✓ Model R² Score: {model_summary['r2_score']:.4f}")
print(f"✓ Model MAE: {model_summary['mae']:.2f}")

# Example: Predict cost for new maintenance records
print("\n2. Example Predictions on New Data")
print("-" * 70)

# Define new maintenance scenarios to predict
new_maintenances = [
    {
        'typeMaintenance': 'Corrective',
        'priorite': 'Critique',
        'type': 'Compresseur',
        'marque': 'Siemens',
        'etat': 'En Maintenance',
        'heuresFonctionnement': 15000,
        'equipment_age_days': 730,  # 2 years
        'nombre_pannes_historique': 5,
        'cout_maintenance_moyen_historique': 5000,
        'dureeEstimee': 6.5,
        'coutEstime': 6500,
        'garantie': False
    },
    {
        'typeMaintenance': 'Préventive',
        'priorite': 'Moyenne',
        'type': 'Pompe',
        'marque': 'ABB',
        'etat': 'Opérationnel',
        'heuresFonctionnement': 8000,
        'equipment_age_days': 365,  # 1 year
        'nombre_pannes_historique': 1,
        'cout_maintenance_moyen_historique': 2000,
        'dureeEstimee': 2.5,
        'coutEstime': 2200,
        'garantie': True
    },
    {
        'typeMaintenance': 'Prédictive',
        'priorite': 'Haute',
        'type': 'Générateur',
        'marque': 'Schneider',
        'etat': 'Opérationnel',
        'heuresFonctionnement': 25000,
        'equipment_age_days': 1460,  # 4 years
        'nombre_pannes_historique': 8,
        'cout_maintenance_moyen_historique': 7500,
        'dureeEstimee': 4.0,
        'coutEstime': 8000,
        'garantie': False
    }
]

# Convert to DataFrame
df_new = pd.DataFrame(new_maintenances)

# Feature engineering (same as training)
df_new['is_critical_priority'] = (df_new['priorite'] == 'Critique').astype(int)
df_new['is_corrective'] = (df_new['typeMaintenance'] == 'Corrective').astype(int)
df_new['is_preventive'] = (df_new['typeMaintenance'] == 'Préventive').astype(int)
df_new['has_garantie'] = df_new['garantie'].astype(int)
df_new['equipment_age_years'] = df_new['equipment_age_days'] / 365.25
df_new['hours_per_day'] = df_new['heuresFonctionnement'] / (df_new['equipment_age_days'] + 1)

# Encode categorical variables
for col in ['typeMaintenance', 'priorite', 'type', 'marque', 'etat']:
    df_new[f'{col}_encoded'] = label_encoders[col].transform(df_new[col].astype(str))

# Select features in the same order as training
feature_columns = model_summary['features_used']
X_new = df_new[feature_columns]

# Handle any missing values
X_new = X_new.fillna(X_new.median())

# Make predictions
# Check if model needs scaling (linear models)
if model_summary['best_model'] in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
else:
    predictions = model.predict(X_new)

# Display predictions
print("\nPredicted Maintenance Costs:\n")
for i, (idx, row) in enumerate(df_new.iterrows()):
    print(f"Scenario {i+1}:")
    print(f"  Type: {row['typeMaintenance']}")
    print(f"  Priority: {row['priorite']}")
    print(f"  Equipment: {row['type']} ({row['marque']})")
    print(f"  Equipment Age: {row['equipment_age_years']:.1f} years")
    print(f"  Operating Hours: {row['heuresFonctionnement']:,} hours")
    print(f"  Historical Failures: {row['nombre_pannes_historique']:.0f}")
    print(f"  Estimated Cost (input): ${row['coutEstime']:,.2f}")
    print(f"  ➡️  PREDICTED ACTUAL COST: ${predictions[i]:,.2f}")
    print(f"  Difference from estimate: ${predictions[i] - row['coutEstime']:,.2f} ({(predictions[i] - row['coutEstime'])/row['coutEstime']*100:+.1f}%)")
    print()

# Save predictions
df_new['predicted_cout_maintenance'] = predictions
df_new['difference_from_estimate'] = predictions - df_new['coutEstime']
df_new.to_csv('/mnt/user-data/outputs/example_predictions.csv', index=False)
print("✓ Predictions saved to example_predictions.csv")

print("\n" + "="*70)
print("PREDICTION COMPLETE!")
print("="*70)

# Create a function for easy prediction
def predict_maintenance_cost(
    type_maintenance,
    priorite,
    equipment_type,
    marque,
    etat,
    heures_fonctionnement,
    equipment_age_days,
    nombre_pannes_historique,
    cout_maintenance_moyen_historique,
    duree_estimee,
    cout_estime,
    garantie
):
    """
    Predict the actual cost of a maintenance operation.
    
    Parameters:
    -----------
    type_maintenance : str
        Type of maintenance: 'Préventive', 'Corrective', or 'Prédictive'
    priorite : str
        Priority level: 'Basse', 'Moyenne', 'Haute', or 'Critique'
    equipment_type : str
        Type of equipment: 'Pompe', 'Compresseur', 'Moteur', 'Générateur', etc.
    marque : str
        Equipment brand: 'Siemens', 'ABB', 'Schneider', 'Bosch', etc.
    etat : str
        Current state: 'Opérationnel', 'En Maintenance', 'Hors Service', 'En Attente'
    heures_fonctionnement : int
        Total operating hours of the equipment
    equipment_age_days : int
        Age of equipment in days
    nombre_pannes_historique : int
        Number of historical failures
    cout_maintenance_moyen_historique : float
        Average historical maintenance cost
    duree_estimee : float
        Estimated duration in hours
    cout_estime : float
        Estimated cost
    garantie : bool
        Whether equipment is under warranty
    
    Returns:
    --------
    float : Predicted maintenance cost
    """
    
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'typeMaintenance': type_maintenance,
        'priorite': priorite,
        'type': equipment_type,
        'marque': marque,
        'etat': etat,
        'heuresFonctionnement': heures_fonctionnement,
        'equipment_age_days': equipment_age_days,
        'nombre_pannes_historique': nombre_pannes_historique,
        'cout_maintenance_moyen_historique': cout_maintenance_moyen_historique,
        'dureeEstimee': duree_estimee,
        'coutEstime': cout_estime,
        'garantie': garantie
    }])
    
    # Feature engineering
    input_data['is_critical_priority'] = (input_data['priorite'] == 'Critique').astype(int)
    input_data['is_corrective'] = (input_data['typeMaintenance'] == 'Corrective').astype(int)
    input_data['is_preventive'] = (input_data['typeMaintenance'] == 'Préventive').astype(int)
    input_data['has_garantie'] = input_data['garantie'].astype(int)
    input_data['equipment_age_years'] = input_data['equipment_age_days'] / 365.25
    input_data['hours_per_day'] = input_data['heuresFonctionnement'] / (input_data['equipment_age_days'] + 1)
    
    # Encode categorical variables
    for col in ['typeMaintenance', 'priorite', 'type', 'marque', 'etat']:
        input_data[f'{col}_encoded'] = label_encoders[col].transform(input_data[col].astype(str))
    
    # Select features
    X = input_data[feature_columns]
    X = X.fillna(X.median())
    
    # Predict
    if model_summary['best_model'] in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
    else:
        prediction = model.predict(X)[0]
    
    return prediction

# Save prediction function info
print("\n3. Usage Example Code:")
print("-" * 70)
print("""
# To use the prediction function in your code:

from predict_maintenance_cost import predict_maintenance_cost

predicted_cost = predict_maintenance_cost(
    type_maintenance='Corrective',
    priorite='Critique',
    equipment_type='Compresseur',
    marque='Siemens',
    etat='En Maintenance',
    heures_fonctionnement=15000,
    equipment_age_days=730,
    nombre_pannes_historique=5,
    cout_maintenance_moyen_historique=5000,
    duree_estimee=6.5,
    cout_estime=6500,
    garantie=False
)

print(f"Predicted cost: ${predicted_cost:,.2f}")
""")

print("\n" + "="*70)
