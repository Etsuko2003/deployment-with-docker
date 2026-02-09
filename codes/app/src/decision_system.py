"""
Système de décision hybride pour agréger les prédictions des modèles.

Ce module combine les prédictions des 4 modèles pour prendre une décision finale.
"""


def hybrid_decision(predictions):
    """
    Agrège les prédictions de tous les modèles avec pondération.
    
    Pondération basée sur les performances réelles (Recall) :
    - Autoencodeur : 60% (Recall 89.4%, F1 0.235)
    - Isolation Forest : 20% (Recall 41.9%)
    - One-Class SVM : 10% (Recall 29.8%)
    - LOF : 10%
    
    Paramètres :
    -----------
    predictions : dict
        Dictionnaire retourné par predict_all_models()
        {
            'autoencoder': {'prediction': 0/1, ...},
            'isolation_forest': {'prediction': 0/1, ...},
            ...
        }
    
    Retourne :
    ---------
    dict
        {
            'status': str ('NORMAL', 'SURVEILLANCE', 'ALERTE'),
            'risk_score': int (0-100),
            'recommendation': str,
            'confidence': float,
            'details': dict
        }
    """
    # Poids basés sur les performances (Recall)
    weights = {
        'autoencoder': 0.80,      # Meilleur Recall (89.4%) - Poids dominant
        'isolation_forest': 0.10,  # Recall moyen (41.9%)
        'ocsvm': 0.05,            # Recall faible (29.8%)
        'lof': 0.05               # Recall estimé faible
    }
    
    # Calculer le score de risque pondéré
    weighted_score = 0
    total_weight = 0
    
    for model_name, pred_dict in predictions.items():
        weight = weights.get(model_name, 0.1)  # Poids par défaut si modèle inconnu
        weighted_score += pred_dict['prediction'] * weight
        total_weight += weight
    
    # Normaliser le score entre 0 et 100
    risk_score = int((weighted_score / total_weight) * 100)
    
    # Calculer la confiance moyenne
    avg_confidence = sum(
        pred_dict.get('confidence', 50) for pred_dict in predictions.values()
    ) / len(predictions)
    
    # Déterminer le statut basé sur le score pondéré
    if risk_score >= 70:  # Si autoencodeur + 1 autre détectent
        status = 'ALERTE'
        recommendation = "ARRÊT IMMÉDIAT - Maintenance urgente requise"
    elif risk_score >= 40:  # Si autoencodeur seul ou 2 modèles faibles
        status = 'SURVEILLANCE'
        recommendation = "SURVEILLANCE ACCRUE - Planifier maintenance préventive"
    elif risk_score >= 20:  # Si 1 modèle faible détecte
        status = 'SURVEILLANCE'
        recommendation = "SURVEILLER - Anomalie détectée"
    else:
        status = 'NORMAL'
        recommendation = "FONCTIONNEMENT NORMAL - Aucune action requise"
    
    # Détails par modèle avec poids
    details = {}
    for model_name, pred_dict in predictions.items():
        details[model_name] = {
            'prediction': 'Panne' if pred_dict['prediction'] == 1 else 'Normal',
            'confidence': pred_dict.get('confidence', 0),
            'weight': weights.get(model_name, 0.1) * 100  # En pourcentage
        }
    
    return {
        'status': status,
        'risk_score': risk_score,
        'recommendation': recommendation,
        'confidence': float(avg_confidence),
        'weighted_score': float(weighted_score),
        'details': details
    }


# ============================================
# EXEMPLE D'UTILISATION
# ============================================

if __name__ == "__main__":
    # Test avec des prédictions simulées
    test_predictions = {
        'autoencoder': {'prediction': 1, 'mse': 0.85, 'confidence': 75},
        'isolation_forest': {'prediction': 1, 'score': 0.12, 'confidence': 80},
        'ocsvm': {'prediction': 0, 'score': -0.05, 'confidence': 60},
        'lof': {'prediction': 1, 'score': 1.8, 'confidence': 70}
    }
    
    print("Test du système de décision...\n")
    decision = hybrid_decision(test_predictions)
    
    print(f"Statut : {decision['status']}")
    print(f"Score de risque : {decision['risk_score']}%")
    print(f"Recommandation : {decision['recommendation']}")
    print(f"Confiance : {decision['confidence']:.1f}%")
    print(f"Score pondéré : {decision['weighted_score']:.2f}")
    
    print(f"\nDétails par modèle :")
    for model_name, details in decision['details'].items():
        print(f"  - {model_name}: {details['prediction']} (Poids: {details['weight']:.0f}%)")
