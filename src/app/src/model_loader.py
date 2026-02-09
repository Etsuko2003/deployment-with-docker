"""
Module de chargement des modèles de détection d'anomalies.

Ce module charge les 4 modèles entraînés (Autoencodeur, Isolation Forest, 
One-Class SVM, LOF) ainsi que leurs scalers respectifs.
"""

import os
import joblib
from tensorflow import keras
import warnings

warnings.filterwarnings('ignore')


class ModelLoader:
    """Classe pour charger et gérer tous les modèles."""
    
    def __init__(self, models_dir='models'):
        """
        Initialise le chargeur de modèles.
        
        Paramètres :
        -----------
        models_dir : str
            Chemin vers le dossier contenant les modèles
        """
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        
    def load_autoencoder(self):
        """Charge l'autoencodeur et son scaler."""
        try:
            # Noms de fichiers réels (dossier en français)
            ae_path = os.path.join(self.models_dir, 'autoencodeur', 'autoencoder_opt.keras')
            scaler_path = os.path.join(self.models_dir, 'autoencodeur', 'scaler_opt.pkl')
            
            self.models['autoencoder'] = keras.models.load_model(ae_path)
            self.scalers['autoencoder'] = joblib.load(scaler_path)
            
            print(" Autoencodeur chargé")
            return True
        except Exception as e:
            print(f" Erreur chargement Autoencodeur : {e}")
            return False
    
    def load_isolation_forest(self):
        """Charge Isolation Forest et son scaler."""
        try:
            # Noms de fichiers réels
            if_path = os.path.join(self.models_dir, 'isolation_forest', 'isolation_forest.pkl')
            scaler_path = os.path.join(self.models_dir, 'isolation_forest', 'scaler_isolation.pkl')
            
            self.models['isolation_forest'] = joblib.load(if_path)
            self.scalers['isolation_forest'] = joblib.load(scaler_path)
            
            print("Isolation Forest chargé")
            return True
        except Exception as e:
            print(f" Erreur chargement Isolation Forest : {e}")
            return False
    
    def load_ocsvm(self):
        """Charge One-Class SVM et son scaler."""
        try:
            # Noms de fichiers réels
            ocsvm_path = os.path.join(self.models_dir, 'one_class_svm', 'ocsvm.pkl')
            scaler_path = os.path.join(self.models_dir, 'one_class_svm', 'scaler_one_class.pkl')
            
            self.models['ocsvm'] = joblib.load(ocsvm_path)
            self.scalers['ocsvm'] = joblib.load(scaler_path)
            
            print("One-Class SVM chargé")
            return True
        except Exception as e:
            print(f"Erreur chargement One-Class SVM : {e}")
            return False
    
    def load_lof(self):
        """Charge LOF et son scaler."""
        try:
            # Noms de fichiers réels
            lof_path = os.path.join(self.models_dir, 'lof', 'lof.pkl')
            scaler_path = os.path.join(self.models_dir, 'lof', 'scaler_lof.pkl')
            
            self.models['lof'] = joblib.load(lof_path)
            self.scalers['lof'] = joblib.load(scaler_path)
            
            print("LOF chargé")
            return True
        except Exception as e:
            print(f"Erreur chargement LOF : {e}")
            return False
    
    def load_lstm(self):
        """Charge le LSTM Autoencoder et son scaler."""
        try:
            # Noms de fichiers pour LSTM
            lstm_path = os.path.join(self.models_dir, 'autoencodeur_lstm', 'autoencoder_lstm.keras')
            scaler_path = os.path.join(self.models_dir, 'autoencodeur_lstm', 'scaler.pkl')
            
            self.models['lstm'] = keras.models.load_model(lstm_path)
            self.scalers['lstm'] = joblib.load(scaler_path)
            
            print("LSTM Autoencoder chargé")
            return True
        except Exception as e:
            print(f"Erreur chargement LSTM : {e}")
            return False
    
    def load_all(self):
        """
        Charge tous les modèles disponibles.
        
        Retourne :
        ---------
        dict
            Dictionnaire avec les modèles et scalers chargés
        """
        print("="*60)
        print("CHARGEMENT DES MODÈLES")
        print("="*60)
        
        self.load_autoencoder()
        self.load_lstm()
        self.load_isolation_forest()
        self.load_ocsvm()
        self.load_lof()
        
        print("="*60)
        print(f"Modèles chargés : {len(self.models)}/5")
        print("="*60)
        
        return {
            'models': self.models,
            'scalers': self.scalers
        }
    
    def get_model(self, model_name):
        """Récupère un modèle spécifique."""
        return self.models.get(model_name)
    
    def get_scaler(self, model_name):
        """Récupère un scaler spécifique."""
        return self.scalers.get(model_name)


def load_all_models(models_dir='models'):
    """
    Fonction utilitaire pour charger tous les modèles.
    
    Paramètres :
    -----------
    models_dir : str
        Chemin vers le dossier des modèles
    
    Retourne :
    ---------
    dict
        {
            'models': {...},
            'scalers': {...}
        }
    """
    loader = ModelLoader(models_dir)
    return loader.load_all()


# ============================================
# EXEMPLE D'UTILISATION
# ============================================

if __name__ == "__main__":
    # Test du chargement
    print("Test du chargement des modèles...\n")
    
    # Charger tous les modèles
    loaded = load_all_models(models_dir='../models')
    
    # Vérifier ce qui a été chargé
    print("\nModèles disponibles :")
    for name in loaded['models'].keys():
        print(f"  - {name}")
    
    print("\nScalers disponibles :")
    for name in loaded['scalers'].keys():
        print(f"  - {name}")
