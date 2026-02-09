"""
Tests unitaires pour le module preprocessing.

Ce fichier contient des tests pour vérifier que le feature engineering
fonctionne correctement.

Pour exécuter les tests :
    poetry run pytest tests/test_preprocessing.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ajouter le dossier parent au path pour importer src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import create_features, select_features, preprocess_for_prediction


# ============================================
# FIXTURES (Données de test réutilisables)
# ============================================

@pytest.fixture
def sample_data():
    """
    Crée un DataFrame de test avec des données réalistes.
    
    Une fixture est une fonction qui prépare des données de test
    réutilisables dans plusieurs tests.
    """
    return pd.DataFrame({
        'Air temperature [K]': [298.1, 300.0, 295.5],
        'Process temperature [K]': [308.6, 310.0, 305.0],
        'Rotational speed [rpm]': [1551, 1600, 1500],
        'Torque [Nm]': [42.8, 40.0, 45.0],
        'Tool wear [min]': [0, 150, 220]
    })


@pytest.fixture
def sample_dict():
    """Crée un dictionnaire de test pour preprocess_for_prediction."""
    return {
        'Air temperature [K]': 298.1,
        'Process temperature [K]': 308.6,
        'Rotational speed [rpm]': 1551,
        'Torque [Nm]': 42.8,
        'Tool wear [min]': 0
    }


# ============================================
# TESTS DE create_features()
# ============================================

class TestCreateFeatures:
    """Groupe de tests pour la fonction create_features."""
    
    def test_creates_all_features(self, sample_data):
        """
        Test 1 : Vérifie que toutes les features sont créées.
        
        Principe :
        - On appelle create_features()
        - On vérifie que les 6 features attendues sont présentes
        """
        result = create_features(sample_data)
        
        expected_features = [
            'Tool wear [min]',
            'temp_diff',
            'power_kw',
            'high_wear',
            'Torque [Nm]',
            'mechanical_stress'
        ]
        
        for feature in expected_features:
            assert feature in result.columns, f"Feature '{feature}' manquante"
    
    def test_temp_diff_calculation(self, sample_data):
        """
        Test 2 : Vérifie le calcul de temp_diff.
        
        Formule : temp_diff = Process_temp - Air_temp
        """
        result = create_features(sample_data)
        
        # Calcul attendu pour la première ligne
        expected_temp_diff = 308.6 - 298.1  # = 10.5
        
        assert result['temp_diff'].iloc[0] == pytest.approx(expected_temp_diff, rel=1e-5)
    
    def test_power_kw_calculation(self, sample_data):
        """
        Test 3 : Vérifie le calcul de power_kw.
        
        Formule : power_kw = (Torque * Rotational_speed) / 9550
        """
        result = create_features(sample_data)
        
        # Calcul attendu pour la première ligne
        expected_power = (42.8 * 1551) / 9550  # ≈ 6.95
        
        assert result['power_kw'].iloc[0] == pytest.approx(expected_power, rel=1e-5)
    
    def test_high_wear_binary(self, sample_data):
        """
        Test 4 : Vérifie que high_wear est binaire (0 ou 1).
        
        Règle : high_wear = 1 si Tool_wear > 200, sinon 0
        """
        result = create_features(sample_data)
        
        # Vérifier les valeurs attendues
        assert result['high_wear'].iloc[0] == 0  # Tool_wear = 0
        assert result['high_wear'].iloc[1] == 0  # Tool_wear = 150
        assert result['high_wear'].iloc[2] == 1  # Tool_wear = 220
        
        # Vérifier que toutes les valeurs sont 0 ou 1
        assert result['high_wear'].isin([0, 1]).all()
    
    def test_mechanical_stress_calculation(self, sample_data):
        """
        Test 5 : Vérifie le calcul de mechanical_stress.
        
        Formule : mechanical_stress = Torque * (1 + Tool_wear / 250)
        """
        result = create_features(sample_data)
        
        # Calcul attendu pour la première ligne
        expected_stress = 42.8 * (1 + 0 / 250)  # = 42.8
        
        assert result['mechanical_stress'].iloc[0] == pytest.approx(expected_stress, rel=1e-5)
    
    def test_no_missing_values(self, sample_data):
        """
        Test 6 : Vérifie qu'il n'y a pas de valeurs manquantes.
        """
        result = create_features(sample_data)
        
        assert not result.isnull().any().any(), "Des valeurs manquantes ont été détectées"
    
    def test_preserves_original_columns(self, sample_data):
        """
        Test 7 : Vérifie que les colonnes originales sont préservées.
        """
        result = create_features(sample_data)
        
        original_columns = sample_data.columns.tolist()
        
        for col in original_columns:
            assert col in result.columns, f"Colonne originale '{col}' perdue"


# ============================================
# TESTS DE select_features()
# ============================================

class TestSelectFeatures:
    """Groupe de tests pour la fonction select_features."""
    
    def test_selects_correct_features(self, sample_data):
        """
        Test 8 : Vérifie que seules les 6 features sont sélectionnées.
        """
        df_with_features = create_features(sample_data)
        result = select_features(df_with_features)
        
        assert result.shape[1] == 6, f"Attendu 6 features, obtenu {result.shape[1]}"
    
    def test_feature_order(self, sample_data):
        """
        Test 9 : Vérifie l'ordre des features.
        """
        df_with_features = create_features(sample_data)
        result = select_features(df_with_features)
        
        expected_order = [
            'Tool wear [min]',
            'temp_diff',
            'power_kw',
            'high_wear',
            'Torque [Nm]',
            'mechanical_stress'
        ]
        
        assert list(result.columns) == expected_order
    
    def test_raises_error_if_missing_feature(self):
        """
        Test 10 : Vérifie qu'une erreur est levée si une feature manque.
        """
        incomplete_df = pd.DataFrame({
            'Tool wear [min]': [0],
            'temp_diff': [10.5]
            # Manque 4 features
        })
        
        with pytest.raises(ValueError, match="Features manquantes"):
            select_features(incomplete_df)


# ============================================
# TESTS DE preprocess_for_prediction()
# ============================================

class TestPreprocessForPrediction:
    """Groupe de tests pour la fonction preprocess_for_prediction."""
    
    def test_returns_dataframe(self, sample_dict):
        """
        Test 11 : Vérifie que la fonction retourne un DataFrame.
        """
        result = preprocess_for_prediction(sample_dict)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_single_row_output(self, sample_dict):
        """
        Test 12 : Vérifie que le résultat contient une seule ligne.
        """
        result = preprocess_for_prediction(sample_dict)
        
        assert result.shape[0] == 1, "Le résultat devrait contenir 1 ligne"
    
    def test_correct_number_of_features(self, sample_dict):
        """
        Test 13 : Vérifie que le résultat contient 6 features.
        """
        result = preprocess_for_prediction(sample_dict)
        
        assert result.shape[1] == 6, f"Attendu 6 features, obtenu {result.shape[1]}"
    
    def test_end_to_end_pipeline(self, sample_dict):
        """
        Test 14 : Test complet du pipeline (end-to-end).
        
        Vérifie que le pipeline complet fonctionne correctement.
        """
        result = preprocess_for_prediction(sample_dict)
        
        # Vérifier la structure
        assert result.shape == (1, 6)
        
        # Vérifier qu'il n'y a pas de NaN
        assert not result.isnull().any().any()
        
        # Vérifier que les valeurs sont numériques
        assert result.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()


# ============================================
# TESTS DE CAS LIMITES (Edge Cases)
# ============================================

class TestEdgeCases:
    """Tests pour les cas limites et situations exceptionnelles."""
    
    def test_zero_values(self):
        """
        Test 15 : Vérifie le comportement avec des valeurs à zéro.
        """
        zero_data = pd.DataFrame({
            'Air temperature [K]': [0],
            'Process temperature [K]': [0],
            'Rotational speed [rpm]': [0],
            'Torque [Nm]': [0],
            'Tool wear [min]': [0]
        })
        
        result = create_features(zero_data)
        
        # Vérifier qu'il n'y a pas d'erreur de division par zéro
        assert not result.isnull().any().any()
    
    def test_large_values(self):
        """
        Test 16 : Vérifie le comportement avec de grandes valeurs.
        """
        large_data = pd.DataFrame({
            'Air temperature [K]': [500],
            'Process temperature [K]': [600],
            'Rotational speed [rpm]': [5000],
            'Torque [Nm]': [100],
            'Tool wear [min]': [500]
        })
        
        result = create_features(large_data)
        
        # Vérifier que high_wear = 1 (car > 200)
        assert result['high_wear'].iloc[0] == 1
    
    def test_negative_temp_diff(self):
        """
        Test 17 : Vérifie que temp_diff peut être négatif.
        """
        data = pd.DataFrame({
            'Air temperature [K]': [310],
            'Process temperature [K]': [300],  # Plus bas que Air
            'Rotational speed [rpm]': [1500],
            'Torque [Nm]': [40],
            'Tool wear [min]': [100]
        })
        
        result = create_features(data)
        
        assert result['temp_diff'].iloc[0] < 0


# ============================================
# COMMANDES POUR EXÉCUTER LES TESTS
# ============================================

"""
Pour exécuter tous les tests :
    poetry run pytest tests/test_preprocessing.py -v

Pour exécuter un test spécifique :
    poetry run pytest tests/test_preprocessing.py::TestCreateFeatures::test_temp_diff_calculation -v

Pour voir la couverture de code :
    poetry run pytest tests/test_preprocessing.py --cov=src.preprocessing --cov-report=html

Options utiles :
    -v : Mode verbose (affiche plus de détails)
    -s : Affiche les print() dans les tests
    -x : Arrête au premier échec
    --tb=short : Affiche un traceback court en cas d'erreur
"""
