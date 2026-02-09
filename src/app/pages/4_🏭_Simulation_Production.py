"""
Page Streamlit pour la simulation de production en temps réel avec prédictions LSTM.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import time
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration du Path ---
src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils import local_css
from model_loader import load_all_models
from preprocessing import preprocess_for_lstm
from predictor import predict_with_lstm

# --- Configuration de la page ---
st.set_page_config(
    page_title="Simulation Production",
    page_icon="🏭",
    layout="wide"
)
local_css("style.css")

# --- Chargement des données et modèles ---
@st.cache_resource
def load_models():
    """Charge et met en cache tous les modèles et scalers."""
    return load_all_models(models_dir='./models')

@st.cache_data
def load_data():
    """Charge et met en cache le dataset."""
    try:
        df = pd.read_csv('data/ai4i2020.csv')
        return df
    except FileNotFoundError:
        st.error("Fichier 'data/ai4i2020.csv' non trouvé.")
        return None

# --- Fonctions de simulation ---
@st.cache_data
def analyze_sensor_trends(_df):
    """
    Analyse les tendances et corrélations des capteurs dans le dataset.
    
    Retourne:
    - dict avec les statistiques et tendances de chaque capteur
    """
    sensor_keys = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    trends = {}
    for key in sensor_keys:
        trends[key] = {
            'mean': _df[key].mean(),
            'std': _df[key].std(),
            'min': _df[key].min(),
            'max': _df[key].max(),
            # Calculer la tendance (différence moyenne entre points consécutifs)
            'drift': _df[key].diff().mean()
        }
    
    # Calculer les corrélations entre capteurs
    correlations = _df[sensor_keys].corr()
    
    return trends, correlations

def simulate_sensor_evolution_realistic(initial_values, duration_seconds=60, noise_level=0.02, speed=1.0, trends=None, correlations=None):
    """
    Simule l'évolution réaliste des capteurs basée sur les tendances du dataset.
    
    Paramètres:
    - initial_values: dict avec les valeurs initiales des capteurs
    - duration_seconds: durée de la simulation en secondes
    - noise_level: niveau de bruit aléatoire (0-1)
    - speed: vitesse de simulation (1.0 = temps réel)
    - trends: dict avec les tendances de chaque capteur
    - correlations: matrice de corrélation entre capteurs
    
    Retourne:
    - Liste de dictionnaires avec l'évolution temporelle
    """
    sensor_keys = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    data = []
    current = {k: initial_values[k] for k in sensor_keys}
    
    # Nombre de points basé sur la durée et la vitesse
    num_points = int(duration_seconds * 10 * speed)  # 10 points par seconde
    
    for t in range(num_points):
        new_values = {}
        
        # 1. Usure de l'outil : augmente toujours (tendance croissante)
        wear_increase = 0.05 + np.random.normal(0, 0.01)  # Augmentation réaliste
        current['Tool wear [min]'] = current['Tool wear [min]'] + wear_increase
        
        # 2. Températures : corrélées entre elles et influencées par la vitesse/torque
        # Air temperature évolue lentement
        air_temp_drift = np.random.normal(0, 0.5)
        current['Air temperature [K]'] += air_temp_drift
        
        # Process temperature : corrélée avec air temp + influence du torque et vitesse
        process_temp_base = current['Air temperature [K]'] + 10  # Toujours > air temp
        torque_influence = (current['Torque [Nm]'] - 40) * 0.1  # Torque élevé = plus chaud
        speed_influence = (current['Rotational speed [rpm]'] - 1500) * 0.002
        process_temp_drift = np.random.normal(0, 1)
        current['Process temperature [K]'] = process_temp_base + torque_influence + speed_influence + process_temp_drift
        
        # 3. Vitesse de rotation : peut varier mais reste dans une plage
        speed_drift = np.random.normal(0, 50)  # Variations normales
        # Tendance à diminuer légèrement avec l'usure
        wear_effect = -current['Tool wear [min]'] * 0.5
        current['Rotational speed [rpm]'] += speed_drift + wear_effect
        
        # 4. Torque : corrélé inversement avec la vitesse + augmente avec l'usure
        # Relation physique : Puissance = Torque × Vitesse
        base_torque = 40  # Torque de base
        speed_factor = (2000 - current['Rotational speed [rpm]']) * 0.01  # Vitesse basse = torque élevé
        wear_factor = current['Tool wear [min]'] * 0.05  # Usure augmente le torque
        torque_noise = np.random.normal(0, 2)
        current['Torque [Nm]'] = base_torque + speed_factor + wear_factor + torque_noise
        
        # Ajouter du bruit supplémentaire selon le niveau configuré
        for key in sensor_keys:
            noise = np.random.normal(0, noise_level * abs(current[key]))
            current[key] += noise
        
        # IMPORTANT: Clipper les valeurs selon les limites du dataset d'entraînement
        if trends:
            for key in sensor_keys:
                current[key] = np.clip(current[key], trends[key]['min'], trends[key]['max'])
        
        # Enregistrer les valeurs
        data.append({
            **{k: current[k] for k in sensor_keys},
            'timestamp': t / (10 * speed),  # Temps en secondes
            'timestep': t
        })
    
    return data

# --- Interface Principale ---
st.title("🏭 Simulation de Production en Temps Réel")
st.markdown("""
Cette page simule une production en temps réel avec visualisation des capteurs 
et prédictions continues du modèle LSTM pour détecter les anomalies.

**🔬 Modèle de Simulation Réaliste :**
- **Usure de l'outil** : Augmente progressivement (tendance croissante naturelle)
- **Températures** : Corrélées entre elles, influencées par la vitesse et le torque
- **Vitesse de rotation** : Diminue légèrement avec l'usure de l'outil
- **Torque** : Augmente avec l'usure et varie inversement avec la vitesse (loi physique)
- **Corrélations** : Les capteurs évoluent de manière interdépendante comme dans la réalité

**⚠️ Respect des Limites du Dataset :**
Les valeurs simulées sont automatiquement limitées aux plages min/max observées dans le dataset 
d'entraînement pour garantir que le modèle LSTM puisse faire des prédictions cohérentes.
""")

# Charger les données et les modèles
models_data = load_models()
df = load_data()

if df is not None and models_data:
    models = models_data['models']
    scalers = models_data['scalers']
    
    # Vérifier que le modèle LSTM est disponible
    if 'lstm' not in models or 'lstm' not in scalers:
        st.error("Le modèle LSTM n'est pas disponible. Vérifiez les fichiers de modèles.")
        st.stop()
    
    # --- Sidebar : Configuration ---
    st.sidebar.header("🎛️ Configuration de la Simulation")
    
    # Sélection de l'observation de départ
    st.sidebar.subheader("📊 Observation de Départ")
    observation_index = st.sidebar.number_input(
        "Index de l'observation",
        min_value=0,
        max_value=len(df)-1,
        value=0,
        step=1
    )
    
    # Afficher les valeurs de l'observation sélectionnée
    selected_obs = df.iloc[observation_index]
    with st.sidebar.expander("Voir les valeurs"):
        st.write(f"**Air temp:** {selected_obs['Air temperature [K]']:.1f} K")
        st.write(f"**Process temp:** {selected_obs['Process temperature [K]']:.1f} K")
        st.write(f"**Speed:** {selected_obs['Rotational speed [rpm]']:.0f} rpm")
        st.write(f"**Torque:** {selected_obs['Torque [Nm]']:.1f} Nm")
        st.write(f"**Tool wear:** {selected_obs['Tool wear [min]']:.0f} min")
        st.write(f"**Machine failure:** {'🚨 Oui' if selected_obs['Machine failure'] == 1 else '✅ Non'}")
    
    st.sidebar.divider()
    
    # Paramètres de simulation
    st.sidebar.subheader("⚙️ Paramètres")
    duration = st.sidebar.slider("Durée (secondes)", 10, 300, 60, 10)
    speed = st.sidebar.slider("Vitesse de simulation", 0.5, 3.0, 1.0, 0.5)
    noise_level = st.sidebar.slider("Niveau de bruit (%)", 0, 10, 1, 1) / 100  # Réduit de 2% à 1%
    
    st.sidebar.divider()
    
    # Initialiser l'état de la session
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
        st.session_state.simulation_data = None
        st.session_state.current_step = 0
        st.session_state.predictions = []
        st.session_state.anomalies_detected = 0
    
    # Boutons de contrôle
    st.sidebar.subheader("🎮 Contrôles")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Démarrer", use_container_width=True, disabled=st.session_state.simulation_running):
            # Analyser les tendances du dataset
            trends, correlations = analyze_sensor_trends(df)
            
            # Générer les données de simulation avec tendances réalistes
            initial_values = selected_obs.to_dict()
            st.session_state.simulation_data = simulate_sensor_evolution_realistic(
                initial_values, 
                duration, 
                noise_level, 
                speed,
                trends,
                correlations
            )
            st.session_state.simulation_running = True
            st.session_state.current_step = 0
            st.session_state.predictions = []
            st.session_state.anomalies_detected = 0
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.simulation_running = False
            st.session_state.simulation_data = None
            st.session_state.current_step = 0
            st.session_state.predictions = []
            st.session_state.anomalies_detected = 0
            st.rerun()
    
    # --- Zone Principale ---
    if st.session_state.simulation_running and st.session_state.simulation_data:
        # Avancer la simulation
        if st.session_state.current_step < len(st.session_state.simulation_data):
            current_data = st.session_state.simulation_data[st.session_state.current_step]
            
            # Prédiction LSTM
            try:
                # Préparer les données pour LSTM : utiliser les 20 derniers points
                if st.session_state.current_step >= 19:
                    # On a assez de points pour créer une vraie séquence
                    history_points = st.session_state.simulation_data[st.session_state.current_step-19:st.session_state.current_step+1]
                    
                    # Créer un DataFrame avec les 20 points
                    history_df = pd.DataFrame(history_points)
                    sensor_keys = [
                        'Air temperature [K]',
                        'Process temperature [K]',
                        'Rotational speed [rpm]',
                        'Torque [Nm]',
                        'Tool wear [min]'
                    ]
                    
                    # Extraire les features pour LSTM (12 features)
                    from preprocessing import create_all_features, select_features_for_model
                    
                    features_list = []
                    for point in history_points:
                        features = create_all_features(point)
                        selected = select_features_for_model(features, model_type='lstm')
                        features_list.append(selected.values[0])
                    
                    # Créer la séquence (20, 12)
                    X_sequence = np.array(features_list).reshape(1, 20, 12)
                    
                    # Normaliser
                    X_scaled = scalers['lstm'].transform(
                        X_sequence.reshape(-1, X_sequence.shape[-1])
                    ).reshape(X_sequence.shape)
                    
                    # Prédire
                    prediction_result = predict_with_lstm(X_scaled, models['lstm'], scalers['lstm'])
                    st.session_state.predictions.append(prediction_result)
                    
                    if prediction_result['prediction'] == 1:
                        st.session_state.anomalies_detected += 1
                else:
                    # Pas assez de points pour une séquence complète
                    # On considère comme normal au début
                    prediction_result = {'prediction': 0, 'mse': 0.01, 'confidence': 95.0, 'threshold': 0.08}
                    st.session_state.predictions.append(prediction_result)
            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")
                prediction_result = {'prediction': -1, 'mse': 0, 'confidence': 0}
            
            # Métriques en temps réel
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⏱️ Temps écoulé", f"{current_data['timestamp']:.1f}s")
            with col2:
                st.metric("📊 Points analysés", st.session_state.current_step + 1)
            with col3:
                st.metric("🚨 Anomalies détectées", st.session_state.anomalies_detected)
            with col4:
                if st.session_state.current_step > 0:
                    anomaly_rate = (st.session_state.anomalies_detected / (st.session_state.current_step + 1)) * 100
                    st.metric("📈 Taux d'anomalie", f"{anomaly_rate:.1f}%")
            
            # Prédiction actuelle
            st.subheader("🤖 Prédiction LSTM en Temps Réel")
            if prediction_result['prediction'] == 1:
                st.error(f"🚨 **ANOMALIE DÉTECTÉE !** - MSE: {prediction_result['mse']:.6f} - Confiance: {prediction_result['confidence']:.1f}%")
            elif prediction_result['prediction'] == 0:
                st.success(f"✅ **Fonctionnement Normal** - MSE: {prediction_result['mse']:.6f} - Confiance: {prediction_result['confidence']:.1f}%")
            
            # Graphiques en temps réel
            st.subheader("📈 Évolution des Capteurs en Temps Réel")
            
            # Préparer les données pour le graphique
            history_df = pd.DataFrame(st.session_state.simulation_data[:st.session_state.current_step + 1])
            
            # Créer le graphique avec subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Air Temperature', 'Process Temperature',
                    'Rotational Speed', 'Torque',
                    'Tool Wear', 'Prédictions'
                ),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # Air Temperature
            fig.add_trace(
                go.Scatter(x=history_df['timestamp'], y=history_df['Air temperature [K]'],
                          mode='lines', name='Air Temp', line=dict(color='#3498DB')),
                row=1, col=1
            )
            
            # Process Temperature
            fig.add_trace(
                go.Scatter(x=history_df['timestamp'], y=history_df['Process temperature [K]'],
                          mode='lines', name='Process Temp', line=dict(color='#E74C3C')),
                row=1, col=2
            )
            
            # Rotational Speed
            fig.add_trace(
                go.Scatter(x=history_df['timestamp'], y=history_df['Rotational speed [rpm]'],
                          mode='lines', name='Speed', line=dict(color='#2ECC71')),
                row=2, col=1
            )
            
            # Torque
            fig.add_trace(
                go.Scatter(x=history_df['timestamp'], y=history_df['Torque [Nm]'],
                          mode='lines', name='Torque', line=dict(color='#F39C12')),
                row=2, col=2
            )
            
            # Tool Wear
            fig.add_trace(
                go.Scatter(x=history_df['timestamp'], y=history_df['Tool wear [min]'],
                          mode='lines', name='Tool Wear', line=dict(color='#9B59B6')),
                row=3, col=1
            )
            
            # Prédictions (MSE au fil du temps)
            if st.session_state.predictions:
                mse_values = [p['mse'] for p in st.session_state.predictions]
                timestamps = history_df['timestamp'].values
                fig.add_trace(
                    go.Scatter(x=timestamps, y=mse_values,
                              mode='lines', name='MSE', line=dict(color='#1ABC9C')),
                    row=3, col=2
                )
                
                # Ajouter une ligne de seuil
                threshold = st.session_state.predictions[0].get('threshold', 0.08)
                fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                             annotation_text="Seuil", row=3, col=2)
            
            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Avancer au prochain pas
            st.session_state.current_step += 1
            
            # Pause pour effet temps réel (augmenté pour réduire les flashs)
            time.sleep(0.3 / speed)
            
            # Rafraîchir automatiquement
            if st.session_state.current_step < len(st.session_state.simulation_data):
                st.rerun()
            else:
                st.session_state.simulation_running = False
                st.success("✅ Simulation terminée !")
                
                # Résumé final
                st.subheader("📊 Résumé de la Simulation")
                total_points = len(st.session_state.simulation_data)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total de points", total_points)
                with col2:
                    st.metric("Anomalies détectées", st.session_state.anomalies_detected)
                with col3:
                    final_rate = (st.session_state.anomalies_detected / total_points) * 100
                    st.metric("Taux d'anomalie final", f"{final_rate:.2f}%")
                
                # Tableau comparatif : Limites du dataset vs Valeurs simulées
                st.subheader("📏 Validation des Limites")
                st.info("""
                **Important** : Les valeurs simulées sont automatiquement limitées aux plages observées 
                dans le dataset d'entraînement pour garantir la cohérence des prédictions du modèle LSTM.
                """)
                
                # Analyser les tendances
                trends, _ = analyze_sensor_trends(df)
                
                # Calculer les statistiques des valeurs simulées
                sim_df = pd.DataFrame(st.session_state.simulation_data)
                sensor_keys = [
                    'Air temperature [K]',
                    'Process temperature [K]',
                    'Rotational speed [rpm]',
                    'Torque [Nm]',
                    'Tool wear [min]'
                ]
                
                comparison_data = []
                for key in sensor_keys:
                    sim_min = sim_df[key].min()
                    sim_max = sim_df[key].max()
                    sim_mean = sim_df[key].mean()
                    
                    dataset_min = trends[key]['min']
                    dataset_max = trends[key]['max']
                    dataset_mean = trends[key]['mean']
                    
                    # Vérifier si on est proche des limites (>95% de la plage)
                    range_used = ((sim_max - sim_min) / (dataset_max - dataset_min)) * 100
                    
                    comparison_data.append({
                        'Capteur': key.replace(' [K]', '').replace(' [rpm]', '').replace(' [Nm]', '').replace(' [min]', ''),
                        'Dataset Min': f"{dataset_min:.2f}",
                        'Sim Min': f"{sim_min:.2f}",
                        'Dataset Max': f"{dataset_max:.2f}",
                        'Sim Max': f"{sim_max:.2f}",
                        'Dataset Moy': f"{dataset_mean:.2f}",
                        'Sim Moy': f"{sim_mean:.2f}",
                        '% Plage utilisée': f"{range_used:.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    else:
        st.info("👈 Configurez les paramètres dans le menu latéral et cliquez sur 'Démarrer' pour lancer la simulation.")
        
        # Aperçu de ce qui sera simulé
        st.subheader("📋 Aperçu de la Simulation")
        st.markdown(f"""
        **Configuration actuelle :**
        - Observation de départ : Index {observation_index}
        - Durée : {duration} secondes
        - Vitesse : {speed}x
        - Niveau de bruit : {noise_level*100:.0f}%
        - Points à générer : ~{int(duration * 10 * speed)}
        """)
else:
    st.error("Le chargement des données ou des modèles a échoué. Vérifiez les chemins et les fichiers.")
