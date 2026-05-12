import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of records to generate
N_UTILISATEURS = 100
N_EQUIPEMENTS = 500
N_MAINTENANCES = 2000
N_INTERVENTIONS = 3000
N_PANNES = 1500
N_PIECES_STOCK = 200
N_MOUVEMENTS_STOCK = 3000

# Helper function to generate random dates
def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

# Generate Utilisateur data
print("Generating Utilisateur data...")
utilisateurs = []
roles = ['Admin', 'Manager', 'Technician']
specialites = ['Électricité', 'Mécanique', 'Hydraulique', 'Électronique', 'Généraliste']

for i in range(N_UTILISATEURS):
    role = random.choice(roles)
    utilisateur = {
        'id_utilisateur': i + 1,
        'nom': f'User{i+1}',
        'email': f'user{i+1}@company.com',
        'motDePasse': f'hash_{i+1}',
        'role': role,
        'numTel': f'+216{random.randint(20000000, 99999999)}',
        'specialite': random.choice(specialites) if role == 'Technician' else None
    }
    utilisateurs.append(utilisateur)

df_utilisateurs = pd.DataFrame(utilisateurs)

# Generate Equipement data
print("Generating Equipement data...")
equipements = []
types_equipement = ['Pompe', 'Compresseur', 'Moteur', 'Générateur', 'Convoyeur', 'Robot', 'CNC Machine', 'Presse']
etats = ['Opérationnel', 'En Maintenance', 'Hors Service', 'En Attente']
marques = ['Siemens', 'ABB', 'Schneider', 'Bosch', 'Mitsubishi', 'Fanuc']

start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 1, 1)

for i in range(N_EQUIPEMENTS):
    date_acquisition = random_date(start_date, end_date)
    equipement = {
        'id_equipement': i + 1,
        'nom': f'{random.choice(types_equipement)}_{i+1}',
        'code': f'EQP{str(i+1).zfill(5)}',
        'type': random.choice(types_equipement),
        'marque': random.choice(marques),
        'numeroSerie': f'SN{random.randint(100000, 999999)}',
        'etat': random.choice(etats),
        'dateAcquisition': date_acquisition.strftime('%Y-%m-%d'),
        'prixAchat': round(random.uniform(5000, 100000), 2),
        'localisation': f'Zone {random.choice(["A", "B", "C", "D"])}-{random.randint(1, 20)}',
        'modele': f'Model-{random.randint(100, 999)}',
        'documentation': f'doc_eq_{i+1}.pdf',
        'heuresFonctionnement': random.randint(0, 50000),
        'garantie': random.choice([True, False]),
        'dateDerniereMaintenance': random_date(date_acquisition, datetime.now()).strftime('%Y-%m-%d'),
        'dateProchaineMaintenance': random_date(datetime.now(), datetime.now() + timedelta(days=180)).strftime('%Y-%m-%d'),
        'fabricant': random.choice(marques),
        'dateDebut': date_acquisition.strftime('%Y-%m-%d')
    }
    equipements.append(equipement)

df_equipements = pd.DataFrame(equipements)

# Generate Maintenance data with cost
print("Generating Maintenance data...")
maintenances = []
types_maintenance = ['Préventive', 'Corrective', 'Prédictive']
statuts = ['Planifiée', 'En Cours', 'Terminée', 'Annulée']
priorites = ['Basse', 'Moyenne', 'Haute', 'Critique']

for i in range(N_MAINTENANCES):
    eq_id = random.randint(1, N_EQUIPEMENTS)
    type_maint = random.choice(types_maintenance)
    statut = random.choice(statuts)
    priorite = random.choice(priorites)
    
    # Get equipment data for realistic dates
    eq_date_acq = df_equipements[df_equipements['id_equipement'] == eq_id]['dateAcquisition'].values[0]
    eq_date_acq = datetime.strptime(eq_date_acq, '%Y-%m-%d')
    
    date_planifiee = random_date(eq_date_acq, datetime.now() + timedelta(days=30))
    
    # Cost factors
    base_cost = random.uniform(100, 5000)
    
    # Type multiplier
    type_multiplier = {'Préventive': 0.7, 'Corrective': 1.5, 'Prédictive': 1.0}[type_maint]
    
    # Priority multiplier
    priority_multiplier = {'Basse': 0.8, 'Moyenne': 1.0, 'Haute': 1.3, 'Critique': 2.0}[priorite]
    
    # Equipment age factor
    equipment_age_days = (datetime.now() - eq_date_acq).days
    age_factor = 1 + (equipment_age_days / 3650) * 0.5  # 50% increase over 10 years
    
    # Calculate maintenance cost
    cout_maintenance = round(base_cost * type_multiplier * priority_multiplier * age_factor * random.uniform(0.8, 1.2), 2)
    
    # Duration based on type and priority
    if type_maint == 'Préventive':
        duree_estimee = random.uniform(1, 4)
    elif type_maint == 'Corrective':
        duree_estimee = random.uniform(2, 8)
    else:  # Prédictive
        duree_estimee = random.uniform(1.5, 5)
    
    if priorite == 'Critique':
        duree_estimee *= 1.5
    
    maintenance = {
        'id_maintenance': i + 1,
        'codeMaintenace': f'MAINT{str(i+1).zfill(6)}',
        'idEquipement': eq_id,
        'typeMaintenance': type_maint,
        'datePlanifiee': date_planifiee.strftime('%Y-%m-%d'),
        'dateDebut': (date_planifiee + timedelta(days=random.randint(-2, 5))).strftime('%Y-%m-%d') if statut in ['En Cours', 'Terminée'] else None,
        'dateFin': (date_planifiee + timedelta(days=random.randint(1, 10))).strftime('%Y-%m-%d') if statut == 'Terminée' else None,
        'statut': statut,
        'priorite': priorite,
        'description': f'Maintenance {type_maint.lower()} pour équipement {eq_id}',
        'coutEstime': round(cout_maintenance * random.uniform(0.9, 1.0), 2),
        'coutReel': cout_maintenance if statut == 'Terminée' else None,
        'technicienAssigne': random.choice(df_utilisateurs[df_utilisateurs['role'] == 'Technician']['id_utilisateur'].values) if len(df_utilisateurs[df_utilisateurs['role'] == 'Technician']) > 0 else None,
        'dureeEstimee': round(duree_estimee, 2),
        'dureeReelle': round(duree_estimee * random.uniform(0.8, 1.3), 2) if statut == 'Terminée' else None,
        'fichesProbleme': f'fiche_{i+1}.pdf' if type_maint == 'Corrective' else None,
        'fichesActions': f'actions_{i+1}.pdf',
        'notesTechniques': f'Notes techniques pour maintenance {i+1}',
        'causePanne': random.choice(['Usure', 'Surchauffe', 'Vibration', 'Fuite', 'Panne électrique']) if type_maint == 'Corrective' else None,
        'niveauIntervention': random.choice(['Niveau 1', 'Niveau 2', 'Niveau 3']),
        'prevueSatisfaction': random.randint(1, 5) if statut == 'Terminée' else None,
        'dateCreation': (date_planifiee - timedelta(days=random.randint(7, 30))).strftime('%Y-%m-%d'),
        'dateMiseAJour': datetime.now().strftime('%Y-%m-%d')
    }
    maintenances.append(maintenance)

df_maintenances = pd.DataFrame(maintenances)

# Generate Intervention data
print("Generating Intervention data...")
interventions = []

for i in range(N_INTERVENTIONS):
    maint_id = random.randint(1, N_MAINTENANCES)
    maintenance = df_maintenances[df_maintenances['id_maintenance'] == maint_id].iloc[0]
    
    date_planif = datetime.strptime(maintenance['datePlanifiee'], '%Y-%m-%d')
    date_debut = date_planif + timedelta(hours=random.randint(0, 48))
    date_fin = date_debut + timedelta(hours=random.uniform(1, 12))
    
    intervention = {
        'id_intervention': i + 1,
        'code': f'INT{str(i+1).zfill(6)}',
        'nomEquipement': df_equipements[df_equipements['id_equipement'] == maintenance['idEquipement']]['nom'].values[0],
        'datePlanification': date_planif.strftime('%Y-%m-%d %H:%M:%S'),
        'dureeIntervention': f'{random.randint(1, 8)} heures',
        'cout': round(random.uniform(200, 3000), 2),
        'dateMiseAJour': datetime.now().strftime('%Y-%m-%d'),
        'etat': random.choice(['Planifiée', 'En Cours', 'Terminée']),
        'dateDebut': date_debut.strftime('%Y-%m-%d %H:%M:%S'),
        'dateOuverture': (date_planif - timedelta(days=random.randint(1, 7))).strftime('%Y-%m-%d'),
        'dateCloture': date_fin.strftime('%Y-%m-%d') if random.random() > 0.3 else None
    }
    interventions.append(intervention)

df_interventions = pd.DataFrame(interventions)

# Generate Panne data
print("Generating Panne data...")
pannes = []
categories = ['Électrique', 'Mécanique', 'Hydraulique', 'Pneumatique', 'Électronique']
causes = ['Usure normale', 'Défaut de fabrication', 'Mauvaise utilisation', 'Surchauffe', 'Corrosion', 'Vibration excessive']

for i in range(N_PANNES):
    eq_id = random.randint(1, N_EQUIPEMENTS)
    eq_date_acq = datetime.strptime(df_equipements[df_equipements['id_equipement'] == eq_id]['dateAcquisition'].values[0], '%Y-%m-%d')
    
    date_observation = random_date(eq_date_acq, datetime.now())
    
    panne = {
        'id_panne': i + 1,
        'codeReference': f'PAN{str(i+1).zfill(6)}',
        'idEquipement': eq_id,
        'nomEquipement': df_equipements[df_equipements['id_equipement'] == eq_id]['nom'].values[0],
        'emplacement': df_equipements[df_equipements['id_equipement'] == eq_id]['localisation'].values[0],
        'nomPanne': f'Panne {random.choice(categories).lower()} {i+1}',
        'descriptionPanne': f'Description détaillée de la panne {i+1}',
        'dateObservation': date_observation.strftime('%Y-%m-%d %H:%M:%S'),
        'dateProchaineIntervention': (date_observation + timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
        'categorieProbleme': random.choice(categories),
        'dateOperation': date_observation.strftime('%Y-%m-%d %H:%M:%S'),
        'etatReparation': random.choice(['En attente', 'En cours', 'Réparée', 'Non réparable']),
        'influenceMaintenance': random.choice(['Faible', 'Moyenne', 'Forte']),
        'detTechnician': random.choice(df_utilisateurs[df_utilisateurs['role'] == 'Technician']['id_utilisateur'].values) if len(df_utilisateurs[df_utilisateurs['role'] == 'Technician']) > 0 else None,
        'piece': f'Pièce_{random.randint(1, 100)}',
        'nomTechnician': f'Tech_{random.randint(1, 50)}',
        'pieceTechnician': f'Pièce de rechange {random.randint(1, 200)}',
        'dateCloture': (date_observation + timedelta(days=random.randint(1, 15))).strftime('%Y-%m-%d') if random.random() > 0.4 else None,
        'dateModifJour': datetime.now().strftime('%Y-%m-%d')
    }
    pannes.append(panne)

df_pannes = pd.DataFrame(pannes)

# Generate PieceStock data
print("Generating PieceStock data...")
pieces_stock = []
types_pieces = ['Roulement', 'Joint', 'Filtre', 'Courroie', 'Capteur', 'Valve', 'Moteur électrique', 'Pompe']
unites = ['pcs', 'kg', 'm', 'l']

for i in range(N_PIECES_STOCK):
    piece = {
        'id_piece': i + 1,
        'codePiece': f'PC{str(i+1).zfill(5)}',
        'nomPiece': f'{random.choice(types_pieces)} {i+1}',
        'typepiece': random.choice(types_pieces),
        'unite': random.choice(unites),
        'emplacement': f'Étagère {random.choice(["A", "B", "C", "D"])}-{random.randint(1, 50)}',
        'emploiementStock': f'Stock {random.choice(["Principal", "Secondaire", "Urgent"])}',
        'numerodecommande': f'CMD{random.randint(10000, 99999)}',
        'quantiteDisponible': random.randint(0, 200),
        'dateReception': random_date(datetime(2020, 1, 1), datetime.now()).strftime('%Y-%m-%d'),
        'prixUnitaire': round(random.uniform(10, 1000), 2)
    }
    pieces_stock.append(piece)

df_pieces_stock = pd.DataFrame(pieces_stock)

# Generate MouvementStock data
print("Generating MouvementStock data...")
mouvements_stock = []
types_mouvement = ['Entrée', 'Sortie', 'Transfert', 'Ajustement']

for i in range(N_MOUVEMENTS_STOCK):
    piece_id = random.randint(1, N_PIECES_STOCK)
    type_mvt = random.choice(types_mouvement)
    
    if type_mvt == 'Sortie':
        quantite = -random.randint(1, 20)
    else:
        quantite = random.randint(1, 50)
    
    mouvement = {
        'id_mouvement': i + 1,
        'idPiece': piece_id,
        'typeMouvement': type_mvt,
        'quantite': quantite,
        'dateCreation': random_date(datetime(2023, 1, 1), datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
        'notes': f'Mouvement {type_mvt.lower()} - {abs(quantite)} unités',
        'objetId': random.randint(1, 100),
        'quantiteApres': df_pieces_stock[df_pieces_stock['id_piece'] == piece_id]['quantiteDisponible'].values[0] + quantite,
        'nomUtilisateur': random.choice(df_utilisateurs['nom'].values),
        'dateMovement': random_date(datetime(2023, 1, 1), datetime.now()).strftime('%Y-%m-%d')
    }
    mouvements_stock.append(mouvement)

df_mouvements_stock = pd.DataFrame(mouvements_stock)

# Generate Notification data
print("Generating Notification data...")
notifications = []
for i in range(500):
    notif = {
        'id_notification': i + 1,
        'id_utilisateur': random.randint(1, N_UTILISATEURS),
        'titre': f'Notification {i+1}',
        'message': f'Message de notification {i+1}',
        'dateCreation': random_date(datetime(2024, 1, 1), datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
        'estLu': random.choice([True, False])
    }
    notifications.append(notif)

df_notifications = pd.DataFrame(notifications)

# Save all datasets to CSV
print("\nSaving datasets to CSV files...")
df_utilisateurs.to_csv('/mnt/user-data/outputs/utilisateurs.csv', index=False)
df_equipements.to_csv('/mnt/user-data/outputs/equipements.csv', index=False)
df_maintenances.to_csv('/mnt/user-data/outputs/maintenances.csv', index=False)
df_interventions.to_csv('/mnt/user-data/outputs/interventions.csv', index=False)
df_pannes.to_csv('/mnt/user-data/outputs/pannes.csv', index=False)
df_pieces_stock.to_csv('/mnt/user-data/outputs/pieces_stock.csv', index=False)
df_mouvements_stock.to_csv('/mnt/user-data/outputs/mouvements_stock.csv', index=False)
df_notifications.to_csv('/mnt/user-data/outputs/notifications.csv', index=False)

# Create a master dataset for ML model training
print("\nCreating master dataset for ML model...")

# Merge maintenances with equipements
ml_dataset = df_maintenances.merge(
    df_equipements[['id_equipement', 'type', 'marque', 'etat', 'dateAcquisition', 'heuresFonctionnement', 'garantie']],
    left_on='idEquipement',
    right_on='id_equipement',
    how='left'
)

# Calculate equipment age at maintenance date
ml_dataset['equipment_age_days'] = (
    pd.to_datetime(ml_dataset['datePlanifiee']) - pd.to_datetime(ml_dataset['dateAcquisition'])
).dt.days

# Add panne count per equipment
panne_counts = df_pannes.groupby('idEquipement').size().reset_index(name='nombre_pannes_historique')
ml_dataset = ml_dataset.merge(panne_counts, left_on='idEquipement', right_on='idEquipement', how='left')
ml_dataset['nombre_pannes_historique'].fillna(0, inplace=True)

# Add previous maintenance cost (average)
prev_maint_cost = df_maintenances[df_maintenances['coutReel'].notna()].groupby('idEquipement')['coutReel'].mean().reset_index(name='cout_maintenance_moyen_historique')
ml_dataset = ml_dataset.merge(prev_maint_cost, on='idEquipement', how='left')
ml_dataset['cout_maintenance_moyen_historique'].fillna(ml_dataset['coutEstime'], inplace=True)

# Select relevant features for ML
ml_features = ml_dataset[[
    'id_maintenance',
    'codeMaintenace',
    'idEquipement',
    'typeMaintenance',
    'priorite',
    'type',  # equipment type
    'marque',
    'etat',
    'heuresFonctionnement',
    'garantie',
    'equipment_age_days',
    'nombre_pannes_historique',
    'cout_maintenance_moyen_historique',
    'dureeEstimee',
    'coutEstime',
    'coutReel'  # TARGET VARIABLE
]]

# Only keep records with actual costs (completed maintenances)
ml_features_complete = ml_features[ml_features['coutReel'].notna()].copy()

# Save ML dataset
ml_features_complete.to_csv('/mnt/user-data/outputs/maintenance_cost_ml_dataset.csv', index=False)
ml_features.to_csv('/mnt/user-data/outputs/maintenance_cost_ml_dataset_all.csv', index=False)

print("\n" + "="*70)
print("DATASET GENERATION COMPLETE")
print("="*70)
print(f"\nGenerated datasets:")
print(f"  - Utilisateurs: {len(df_utilisateurs)} records")
print(f"  - Equipements: {len(df_equipements)} records")
print(f"  - Maintenances: {len(df_maintenances)} records")
print(f"  - Interventions: {len(df_interventions)} records")
print(f"  - Pannes: {len(df_pannes)} records")
print(f"  - PiecesStock: {len(df_pieces_stock)} records")
print(f"  - MouvementsStock: {len(df_mouvements_stock)} records")
print(f"  - Notifications: {len(df_notifications)} records")
print(f"\nML Training Dataset:")
print(f"  - Complete records (with coutReel): {len(ml_features_complete)} records")
print(f"  - All records: {len(ml_features)} records")
print("\n" + "="*70)

# Display sample statistics
print("\nMaintenance Cost Statistics (coutMaintenance):")
print(df_maintenances[df_maintenances['coutReel'].notna()]['coutReel'].describe())

print("\nMaintenance Type Distribution:")
print(df_maintenances['typeMaintenance'].value_counts())

print("\nPriority Distribution:")
print(df_maintenances['priorite'].value_counts())

print("\nFiles saved to /mnt/user-data/outputs/")
