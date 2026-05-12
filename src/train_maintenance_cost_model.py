import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("MAINTENANCE COST PREDICTION MODEL")
print("="*70)

# Load the ML dataset
print("\n1. Loading dataset...")
df = pd.read_csv('/mnt/user-data/outputs/maintenance_cost_ml_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Target variable (coutReel) range: {df['coutReel'].min():.2f} - {df['coutReel'].max():.2f}")
print(f"Mean maintenance cost: {df['coutReel'].mean():.2f}")

# Display dataset info
print("\n2. Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Feature Engineering
print("\n3. Feature Engineering...")

# Create additional features
df['is_critical_priority'] = (df['priorite'] == 'Critique').astype(int)
df['is_corrective'] = (df['typeMaintenance'] == 'Corrective').astype(int)
df['is_preventive'] = (df['typeMaintenance'] == 'Préventive').astype(int)
df['has_garantie'] = df['garantie'].astype(int)
df['equipment_age_years'] = df['equipment_age_days'] / 365.25
df['hours_per_day'] = df['heuresFonctionnement'] / (df['equipment_age_days'] + 1)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['typeMaintenance', 'priorite', 'type', 'marque', 'etat']

for col in categorical_columns:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Select features for the model
feature_columns = [
    'typeMaintenance_encoded',
    'priorite_encoded',
    'type_encoded',
    'marque_encoded',
    'etat_encoded',
    'heuresFonctionnement',
    'equipment_age_days',
    'equipment_age_years',
    'nombre_pannes_historique',
    'cout_maintenance_moyen_historique',
    'dureeEstimee',
    'coutEstime',
    'is_critical_priority',
    'is_corrective',
    'is_preventive',
    'has_garantie',
    'hours_per_day'
]

X = df[feature_columns]
y = df['coutReel']

# Handle missing values
print(f"\nChecking for missing values...")
print(X.isnull().sum())
X = X.fillna(X.median())  # Fill missing values with median
print("✓ Missing values filled with median")

print(f"\nFeatures selected: {len(feature_columns)}")
print("Feature list:", feature_columns)

# Correlation analysis
print("\n4. Feature Correlation Analysis...")
correlation_with_target = X.corrwith(y).abs().sort_values(ascending=False)
print("\nTop 10 features correlated with coutReel:")
print(correlation_with_target.head(10))

# Split the data
print("\n5. Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features
print("\n6. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
print("\n7. Training multiple models...")
print("-" * 70)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train on scaled data for linear models, original for tree-based
    if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation score
    if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    results[name] = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'CV_R2_mean': cv_scores.mean(),
        'CV_R2_std': cv_scores.std(),
        'model': model,
        'predictions': y_pred
    }
    
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Find best model
print("\n" + "="*70)
print("8. MODEL COMPARISON")
print("="*70)

results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'MAE': [results[m]['MAE'] for m in results.keys()],
    'RMSE': [results[m]['RMSE'] for m in results.keys()],
    'R²': [results[m]['R2'] for m in results.keys()],
    'CV_R²': [results[m]['CV_R2_mean'] for m in results.keys()]
})

print("\n" + results_df.to_string(index=False))

best_model_name = results_df.loc[results_df['R²'].idxmax(), 'Model']
print(f"\n🏆 Best Model: {best_model_name} (R² = {results[best_model_name]['R2']:.4f})")

# Save best model
best_model = results[best_model_name]['model']
print(f"\n9. Saving best model ({best_model_name})...")
joblib.dump(best_model, '/mnt/user-data/outputs/best_maintenance_cost_model.pkl')
joblib.dump(scaler, '/mnt/user-data/outputs/feature_scaler.pkl')
joblib.dump(label_encoders, '/mnt/user-data/outputs/label_encoders.pkl')
print("✓ Model saved successfully!")

# Feature importance (for tree-based models)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("\n10. Feature Importance Analysis...")
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'].head(10), feature_importance['Importance'].head(10))
    plt.xlabel('Importance')
    plt.title(f'Top 10 Feature Importance - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Feature importance plot saved!")

# Prediction vs Actual plot
print("\n11. Creating visualizations...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, results[best_model_name]['predictions'], alpha=0.6, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Cost (coutReel)')
plt.ylabel('Predicted Cost')
plt.title(f'Actual vs Predicted Maintenance Cost - {best_model_name}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/prediction_vs_actual.png', dpi=300, bbox_inches='tight')
print("✓ Prediction vs Actual plot saved!")

# Residual plot
plt.figure(figsize=(10, 6))
residuals = y_test - results[best_model_name]['predictions']
plt.scatter(results[best_model_name]['predictions'], residuals, alpha=0.6, edgecolors='k')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Cost')
plt.ylabel('Residuals')
plt.title(f'Residual Plot - {best_model_name}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/residuals_plot.png', dpi=300, bbox_inches='tight')
print("✓ Residual plot saved!")

# Model comparison plot
plt.figure(figsize=(12, 6))
x_pos = np.arange(len(results_df))
width = 0.35

plt.subplot(1, 2, 1)
plt.bar(x_pos, results_df['R²'], width)
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.title('Model Comparison - R² Score')
plt.xticks(x_pos, results_df['Model'], rotation=45, ha='right')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(x_pos, results_df['MAE'], width)
plt.xlabel('Model')
plt.ylabel('MAE')
plt.title('Model Comparison - Mean Absolute Error')
plt.xticks(x_pos, results_df['Model'], rotation=45, ha='right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Model comparison plot saved!")

# Create a prediction example
print("\n12. Example Prediction...")
print("-" * 70)

# Take a sample from test set
sample_idx = 0
sample_features = X_test.iloc[sample_idx:sample_idx+1]
actual_cost = y_test.iloc[sample_idx]

if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
    sample_scaled = scaler.transform(sample_features)
    predicted_cost = best_model.predict(sample_scaled)[0]
else:
    predicted_cost = best_model.predict(sample_features)[0]

print("\nSample Maintenance Record:")
print(f"  Type: {df.iloc[y_test.index[sample_idx]]['typeMaintenance']}")
print(f"  Priority: {df.iloc[y_test.index[sample_idx]]['priorite']}")
print(f"  Equipment Type: {df.iloc[y_test.index[sample_idx]]['type']}")
print(f"  Equipment Age (years): {sample_features['equipment_age_years'].values[0]:.2f}")
print(f"  Historical Failures: {sample_features['nombre_pannes_historique'].values[0]:.0f}")
print(f"\n  Actual Cost:    {actual_cost:.2f}")
print(f"  Predicted Cost: {predicted_cost:.2f}")
print(f"  Error:          {abs(actual_cost - predicted_cost):.2f} ({abs(actual_cost - predicted_cost)/actual_cost*100:.1f}%)")

# Save model summary
print("\n13. Saving model summary...")
summary = {
    'best_model': best_model_name,
    'r2_score': results[best_model_name]['R2'],
    'mae': results[best_model_name]['MAE'],
    'rmse': results[best_model_name]['RMSE'],
    'cv_r2_mean': results[best_model_name]['CV_R2_mean'],
    'cv_r2_std': results[best_model_name]['CV_R2_std'],
    'features_used': feature_columns,
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

import json
with open('/mnt/user-data/outputs/model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✓ Model summary saved!")

print("\n" + "="*70)
print("MODEL TRAINING COMPLETE!")
print("="*70)
print("\nFiles saved:")
print("  ✓ best_maintenance_cost_model.pkl")
print("  ✓ feature_scaler.pkl")
print("  ✓ label_encoders.pkl")
print("  ✓ model_summary.json")
print("  ✓ feature_importance.png")
print("  ✓ prediction_vs_actual.png")
print("  ✓ residuals_plot.png")
print("  ✓ model_comparison.png")
print("\n" + "="*70)
