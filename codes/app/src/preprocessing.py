"""
Module de prétraitement et feature engineering pour la maintenance prédictive.

Ce module contient les fonctions pour transformer les données brutes des capteurs
en features utilisables par les modèles de détection d'anomalies.

Supporte deux types de modèles :
- Modèles classiques (Autoencodeur, IF, OCSVM, LOF) : 6 features
- LSTM Autoencoder : 12 features + séquences temporelles
"""

import pandas as pd
import numpy as np


# ============================================
# CONSTANTES - FEATURES PAR TYPE DE MODÈLE
# ============================================

# Features pour LSTM Autoencoder (12 features)
FEATURES_LSTM = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'power_kw',
    'temp_diff',
    'speed_torque_ratio',
    'wear_level',
    'high_wear',
    'thermal_load',
    'mechanical_stress'
]

# Features pour modèles classiques (6 features optimales)
FEATURES_CLASSIC = [
    'Tool wear [min]',
    'temp_diff',
    'power_kw',
    'high_wear',
    'Torque [Nm]',
    'mechanical_stress'
]


def create_all_features(data_dict):
    """
    Applique le feature engineering complet (9 nouvelles features).
    
    Paramètres :
    -----------
    data_dict : dict
        Dictionnaire avec les valeurs des capteurs :
        {
            'Air temperature [K]': float,
            'Process temperature [K]': float,
            'Rotational speed [rpm]': float,
            'Torque [Nm]': float,
            'Tool wear [min]': float
        }
    
    Retourne :
    ---------
    pd.DataFrame
        DataFrame avec toutes les features (5 originales + 9 engineerées)
    """
    # Créer DataFrame à partir du dictionnaire
    df = pd.DataFrame([data_dict])
    
    # ============================================
    # FEATURES PHYSIQUES
    # ============================================
    
    # Puissance en kW
    df['power_kw'] = (df['Torque [Nm]'] * df['Rotational speed [rpm]']) / 9550
    
    # Différence de température
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    
    # Ratio vitesse/couple (éviter division par zéro)
    df['speed_torque_ratio'] = df['Rotational speed [rpm]'] / (df['Torque [Nm]'] + 1)
    
    # ============================================
    # FEATURES D'USURE
    # ============================================
    
    # Niveau d'usure catégorique (valeur max fixe pour la prédiction)
    max_wear = 250
    df['wear_level'] = pd.cut(
        df['Tool wear [min]'], 
        bins=[0, 80, 160, max_wear + 1],
        labels=[0, 1, 2],
        include_lowest=True
    )
    df['wear_level'] = pd.to_numeric(df['wear_level'], errors='coerce').fillna(0).astype(int)
    
    # Usure élevée (binaire)
    df['high_wear'] = (df['Tool wear [min]'] > 200).astype(int)
    
    # ============================================
    # FEATURES D'INTERACTION
    # ============================================
    
    # Charge thermique
    df['thermal_load'] = df['temp_diff'] * df['power_kw']
    
    # Stress mécanique
    df['mechanical_stress'] = df['Torque [Nm]'] * (1 + df['Tool wear [min]'] / 250)
    
    return df


def select_features_for_model(df, model_type='classic'):
    """
    Sélectionne les features appropriées selon le type de modèle.
    
    Paramètres :
    -----------
    df : pd.DataFrame
        DataFrame avec toutes les features
    model_type : str
        'lstm' pour LSTM Autoencoder (12 features)
        'classic' pour modèles classiques (6 features)
    
    Retourne :
    ---------
    pd.DataFrame
        DataFrame avec uniquement les features sélectionnées
    """
    if model_type == 'lstm':
        feature_list = FEATURES_LSTM
    else:  # 'classic'
        feature_list = FEATURES_CLASSIC
    
    # Vérifier que toutes les features existent
    missing_features = [f for f in feature_list if f not in df.columns]
    if missing_features:
        raise ValueError(f"Features manquantes : {missing_features}")
    
    return df[feature_list]


def preprocess_for_classic_models(data_dict):
    """
    Prétraite des données pour les modèles classiques (Autoencodeur, IF, OCSVM, LOF).
    
    Paramètres :
    -----------
    data_dict : dict
        Dictionnaire avec les valeurs des capteurs
    
    Retourne :
    ---------
    pd.DataFrame
        DataFrame prêt pour la prédiction (1 ligne, 6 features)
    """
    # Feature engineering complet
    df_features = create_all_features(data_dict)
    
    # Sélectionner les 6 features optimales
    df_selected = select_features_for_model(df_features, model_type='classic')
    
    return df_selected


def preprocess_for_lstm(data_dict, timesteps=20):
    """
    Prétraite des données pour le LSTM Autoencoder (séquence temporelle).
    
    Pour la prédiction en temps réel, on crée une séquence en répétant
    le point actuel 'timesteps' fois.
    
    Paramètres :
    -----------
    data_dict : dict
        Dictionnaire avec les valeurs des capteurs
    timesteps : int
        Nombre de timesteps dans la séquence (défaut: 20)
    
    Retourne :
    ---------
    np.ndarray
        Array 3D prêt pour LSTM : shape (1, timesteps, 12)
    """
    # Feature engineering complet
    df_features = create_all_features(data_dict)
    
    # Sélectionner les 12 features LSTM
    df_selected = select_features_for_model(df_features, model_type='lstm')
    
    # Convertir en array
    features_array = df_selected.values  # Shape: (1, 12)
    
    # Répéter pour créer une séquence de 'timesteps' timesteps
    sequence = np.tile(features_array, (timesteps, 1))  # Shape: (20, 12)
    
    # Reshape pour LSTM : (batch_size=1, timesteps, features)
    sequence_reshaped = sequence.reshape(1, timesteps, 12)
    
    return sequence_reshaped


# ============================================
# FONCTION LEGACY (pour compatibilité)
# ============================================

def preprocess_for_prediction(data_dict):
    """
    Fonction legacy pour compatibilité avec l'ancien code.
    Utilise le preprocessing des modèles classiques.
    
    Paramètres :
    -----------
    data_dict : dict
        Dictionnaire avec les valeurs des capteurs
    
    Retourne :
    ---------
    pd.DataFrame
        DataFrame prêt pour la prédiction (modèles classiques)
    """
    return preprocess_for_classic_models(data_dict)


# ============================================
# EXEMPLE D'UTILISATION
# ============================================

if __name__ == "__main__":
    # Test avec des données d'exemple
    test_data = {
        'Air temperature [K]': 298.1,
        'Process temperature [K]': 308.6,
        'Rotational speed [rpm]': 1551,
        'Torque [Nm]': 42.8,
        'Tool wear [min]': 0
    }
    
    print("="*60)
    print("TEST DU PREPROCESSING")
    print("="*60)
    
    # Test modèles classiques
    print("\n1. Modèles Classiques (6 features) :")
    result_classic = preprocess_for_classic_models(test_data)
    print(f"   Shape : {result_classic.shape}")
    print(f"   Features : {list(result_classic.columns)}")
    print(result_classic)
    
    # Test LSTM
    print("\n2. LSTM Autoencoder (séquence 20x12) :")
    result_lstm = preprocess_for_lstm(test_data, timesteps=20)
    print(f"   Shape : {result_lstm.shape}")
    print(f"   Premier timestep :")
    print(result_lstm[0, 0, :])

