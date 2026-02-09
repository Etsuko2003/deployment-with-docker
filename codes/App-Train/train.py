"""
Script de training pour tous les modèles de détection d'anomalies.

Ce script entraîne :
- Autoencodeur Dense
- LSTM Autoencoder
- Isolation Forest
- One-Class SVM
- LOF (Local Outlier Factor)
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================
# IMPORTS DES MODULES LOCAUX
# ============================================
src_path = Path(__file__).parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from preprocessing import FEATURES_CLASSIC, FEATURES_LSTM, create_all_features

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib


# ============================================
# CONFIGURATION
# ============================================
DATA_PATH = 'data/ai4i2020.csv'
MODELS_DIR = 'models'

# Création des dossiers de sortie
os.makedirs(os.path.join(MODELS_DIR, 'autoencodeur'), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'autoencodeur_lstm'), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'isolation_forest'), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'one_class_svm'), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'Lof'), exist_ok=True)


# ============================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================
def load_and_prepare_data():
    """Charge et prépare les données pour l'entraînement."""
    print("\n" + "="*60)
    print("CHARGEMENT DES DONNÉES")
    print("="*60)
    
    # Chargement
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Feature engineering
    print("✓ Application du feature engineering...")
    df_processed = df.copy()
    
    # Puissance en kW
    df_processed['power_kw'] = (df_processed['Torque [Nm]'] * df_processed['Rotational speed [rpm]']) / 9550
    
    # Différence de température
    df_processed['temp_diff'] = df_processed['Process temperature [K]'] - df_processed['Air temperature [K]']
    
    # Ratio vitesse/couple
    df_processed['speed_torque_ratio'] = df_processed['Rotational speed [rpm]'] / (df_processed['Torque [Nm]'] + 1)
    
    # Niveau d'usure
    max_wear = 250
    df_processed['wear_level'] = pd.cut(
        df_processed['Tool wear [min]'], 
        bins=[0, 80, 160, max_wear + 1],
        labels=[0, 1, 2],
        include_lowest=True
    )
    df_processed['wear_level'] = pd.to_numeric(df_processed['wear_level'], errors='coerce').fillna(0).astype(int)
    
    # Usure élevée (binaire)
    df_processed['high_wear'] = (df_processed['Tool wear [min]'] > 200).astype(int)
    
    # Charge thermique
    df_processed['thermal_load'] = (df_processed['Process temperature [K]'] - 273.15) / 100
    
    # Charge mécanique (couple normalisé)
    df_processed['mechanical_stress'] = df_processed['Torque [Nm]'] / (df_processed['Torque [Nm]'].max() + 1)
    
    print(f"✓ Features engineerées créées")
    
    # Préparation des features pour chaque type de modèle
    X_classic = df_processed[FEATURES_CLASSIC].copy()
    X_lstm = df_processed[FEATURES_LSTM].copy()
    
    # Label (Machine failure colonne)
    if 'Machine failure' in df_processed.columns:
        y = df_processed['Machine failure'].values
    else:
        print("⚠ Colonne 'Machine failure' non trouvée, utilisation de labels binaires basés sur l'usure")
        y = (df_processed['Tool wear [min]'] > 200).astype(int).values
    
    print(f"✓ Distribution des classes : {np.bincount(y)}")
    print(f"  - Normal: {np.sum(y == 0)}")
    print(f"  - Panne: {np.sum(y == 1)}")
    
    return X_classic, X_lstm, y, df_processed


# ============================================
# AUTOENCODEUR DENSE
# ============================================
def train_autoencoder(X_train, X_test):
    """Entraîne l'autoencodeur dense."""
    print("\n" + "="*60)
    print("ENTRAÎNEMENT - AUTOENCODEUR DENSE")
    print("="*60)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    input_dim = X_train_scaled.shape[1]
    
    # Architecture
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(2, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    print(f"✓ Modèle créé : Input={input_dim} features")
    
    # Entraînement
    history = model.fit(
        X_train_scaled, X_train_scaled,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        verbose=1,
        callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)]
    )
    
    print(f"✓ Entraînement terminé : Loss final = {history.history['loss'][-1]:.4f}")
    
    # Sauvegarde
    model.save(os.path.join(MODELS_DIR, 'autoencodeur', 'autoencoder_opt.keras'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'autoencodeur', 'scaler_opt.pkl'))
    
    print("✓ Modèle sauvegardé")
    
    return model, scaler


# ============================================
# LSTM AUTOENCODER
# ============================================
def create_sequences(data, seq_length=10):
    """Crée des séquences pour LSTM."""
    X_seq = []
    for i in range(len(data) - seq_length + 1):
        X_seq.append(data[i:i + seq_length])
    return np.array(X_seq)


def train_lstm_autoencoder(X_train, X_test):
    """Entraîne le LSTM Autoencoder."""
    print("\n" + "="*60)
    print("ENTRAÎNEMENT - LSTM AUTOENCODER")
    print("="*60)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Création de séquences
    seq_length = 10
    X_train_seq = create_sequences(X_train_scaled, seq_length)
    X_test_seq = create_sequences(X_test_scaled, seq_length)
    
    input_shape = X_train_seq.shape[1:]
    feature_dim = X_train_seq.shape[2]
    
    print(f"✓ Séquences créées : shape={X_train_seq.shape}")
    
    # Architecture LSTM optimisée pour CPU
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, activation='relu', return_sequences=True),
        layers.LSTM(16, activation='relu', return_sequences=False),
        layers.RepeatVector(seq_length),
        layers.LSTM(16, activation='relu', return_sequences=True),
        layers.TimeDistributed(layers.Dense(feature_dim))
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    print(f"✓ Modèle LSTM créé")
    
    # Entraînement
    history = model.fit(
        X_train_seq, X_train_seq,
        epochs=15,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)]
    )
    
    print(f"✓ Entraînement terminé : Loss final = {history.history['loss'][-1]:.4f}")
    
    # Sauvegarde
    model.save(os.path.join(MODELS_DIR, 'autoencodeur_lstm', 'autoencoder_lstm.keras'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'autoencodeur_lstm', 'scaler.pkl'))
    
    print("✓ Modèle LSTM sauvegardé")
    
    return model, scaler


# ============================================
# ISOLATION FOREST
# ============================================
def train_isolation_forest(X_train, X_test):
    """Entraîne Isolation Forest."""
    print("\n" + "="*60)
    print("ENTRAÎNEMENT - ISOLATION FOREST")
    print("="*60)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entraînement
    model = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled)
    
    print(f"✓ Isolation Forest entraîné")
    
    # Évaluation
    predictions = model.predict(X_test_scaled)
    predictions = np.where(predictions == -1, 1, 0)  # Convertir -1 -> 1 (anomalie)
    
    # Sauvegarde
    joblib.dump(model, os.path.join(MODELS_DIR, 'isolation_forest', 'isolation_forest.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'isolation_forest', 'scaler_isolation.pkl'))
    
    print("✓ Modèle Isolation Forest sauvegardé")
    
    return model, scaler


# ============================================
# ONE-CLASS SVM
# ============================================
def train_ocsvm(X_train, X_test):
    """Entraîne One-Class SVM."""
    print("\n" + "="*60)
    print("ENTRAÎNEMENT - ONE-CLASS SVM")
    print("="*60)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entraînement
    model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    model.fit(X_train_scaled)
    
    print(f"✓ One-Class SVM entraîné")
    
    # Évaluation
    predictions = model.predict(X_test_scaled)
    predictions = np.where(predictions == -1, 1, 0)
    
    # Sauvegarde
    joblib.dump(model, os.path.join(MODELS_DIR, 'one_class_svm', 'ocsvm.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'one_class_svm', 'scaler_one_class.pkl'))
    
    print("✓ Modèle One-Class SVM sauvegardé")
    
    return model, scaler


# ============================================
# LOCAL OUTLIER FACTOR
# ============================================
def train_lof(X_train, X_test):
    """Entraîne LOF."""
    print("\n" + "="*60)
    print("ENTRAÎNEMENT - LOCAL OUTLIER FACTOR")
    print("="*60)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entraînement
    model = LocalOutlierFactor(n_neighbors=20, novelty=True)
    model.fit(X_train_scaled)
    
    print(f"✓ LOF entraîné")
    
    # Évaluation
    predictions = model.predict(X_test_scaled)
    predictions = np.where(predictions == -1, 1, 0)
    
    # Sauvegarde
    joblib.dump(model, os.path.join(MODELS_DIR, 'Lof', 'lof.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'Lof', 'scaler_lof.pkl'))
    
    print("✓ Modèle LOF sauvegardé")
    
    return model, scaler


# ============================================
# MAIN
# ============================================
def main():
    """Fonction principale."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  TRAINING DE TOUS LES MODÈLES DE DÉTECTION".center(58) + "║")
    print("║" + "="*58 + "║")
    print("║" + " "*58 + "║")
    
    try:
        # Chargement des données
        X_classic, X_lstm, y, df = load_and_prepare_data()
        
        # Split train/test
        X_train_classic, X_test_classic = train_test_split(
            X_classic, test_size=0.2, random_state=42
        )
        X_train_lstm, X_test_lstm = train_test_split(
            X_lstm, test_size=0.2, random_state=42
        )
        
        # Entraînement de tous les modèles
        train_autoencoder(X_train_classic, X_test_classic)
            # train_lstm_autoencoder(X_train_lstm, X_test_lstm)  # Skip LSTM on CPU - too slow
        train_isolation_forest(X_train_classic, X_test_classic)
        train_ocsvm(X_train_classic, X_test_classic)
        train_lof(X_train_classic, X_test_classic)
        
        # Fin
        print("\n" + "="*60)
        print("✓ TOUS LES MODÈLES ONT ÉTÉ ENTRAÎNÉS AVEC SUCCÈS")
        print("="*60)
        print("\nModèles sauvegardés dans :")
        print(f"  - {os.path.join(MODELS_DIR, 'autoencodeur')}")
        print(f"  - {os.path.join(MODELS_DIR, 'autoencodeur_lstm')}")
        print(f"  - {os.path.join(MODELS_DIR, 'isolation_forest')}")
        print(f"  - {os.path.join(MODELS_DIR, 'one_class_svm')}")
        print(f"  - {os.path.join(MODELS_DIR, 'Lof')}")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERREUR : Fichier non trouvé : {e}")
        print(f"   Assurez-vous que {DATA_PATH} existe")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERREUR LORS DE L'ENTRAÎNEMENT : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
