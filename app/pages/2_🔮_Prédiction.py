# """
# Page Streamlit pour la prédiction d'anomalies et la comparaison des modèles.
# """
# import streamlit as st
# import pandas as pd
# import numpy as np
# import sys
# from pathlib import Path

# # --- Configuration du Path ---
# src_path = Path(__file__).parent.parent / 'src'
# if str(src_path) not in sys.path:
#     sys.path.insert(0, str(src_path))

# from utils import local_css
# from model_loader import load_all_models
# from preprocessing import preprocess_for_classic_models, preprocess_for_lstm
# from predictor import predict_all_models

# # --- Configuration de la page et Style ---
# st.set_page_config(
#     page_title="Comparaison des Modèles",
#     page_icon="🔮",
#     layout="wide"
# )
# local_css("style.css")

# # --- Chargement des données et modèles ---
# @st.cache_resource
# def load_models():
#     """Charge et met en cache tous les modèles et scalers."""
#     return load_all_models(models_dir='./models')

# @st.cache_data
# def load_data():
#     """Charge et met en cache le dataset."""
#     try:
#         df = pd.read_csv('data/ai4i2020.csv')
#         return df
#     except FileNotFoundError:
#         return None

# # --- Initialisation ---
# loaded_data = load_models()
# df = load_data()

# # Noms des colonnes pour les capteurs
# SENSOR_COLS = [
#     'Air temperature [K]',
#     'Process temperature [K]',
#     'Rotational speed [rpm]',
#     'Torque [Nm]',
#     'Tool wear [min]'
# ]

# # Initialiser st.session_state pour les inputs
# if 'sensor_values' not in st.session_state:
#     if df is not None:
#         st.session_state.sensor_values = df.iloc[0][SENSOR_COLS].to_dict()
#     else:
#         # Valeurs par défaut si le chargement échoue
#         st.session_state.sensor_values = {col: 0.0 for col in SENSOR_COLS}

# # --- Fonctions de callback pour les boutons ---
# def simulate_random_row():
#     """Charge une ligne aléatoire du dataframe dans le session_state."""
#     if df is not None:
#         random_row = df.sample(n=1).iloc[0]
#         st.session_state.sensor_values = random_row[SENSOR_COLS].to_dict()

# def load_specific_row(index):
#     """Charge une ligne spécifique par son index."""
#     if df is not None and 0 <= index < len(df):
#         specific_row = df.iloc[index]
#         st.session_state.sensor_values = specific_row[SENSOR_COLS].to_dict()

# # --- Interface ---
# st.title("🔮 Comparaison des Modèles en Temps Réel")
# st.markdown("""
# Utilisez les options de la barre latérale pour tester les modèles avec différentes données
# (manuelles, aléatoires ou spécifiques au jeu de données).
# """)

# if not loaded_data or df is None:
#     st.error("Le chargement des données ou des modèles a échoué. Vérifiez les chemins et les fichiers.")
# else:
#     # --- Sidebar ---
#     st.sidebar.header("🕹️ Contrôles de Simulation")
#     st.sidebar.button(
#         "🎲 Simuler une mesure aléatoire",
#         on_click=simulate_random_row,
#         use_container_width=True
#     )

#     st.sidebar.divider()
#     st.sidebar.header("🌡️ Valeurs des Capteurs")

#     # Les number_input sont maintenant liés au session_state
#     for key, value in st.session_state.sensor_values.items():
#         st.session_state.sensor_values[key] = st.sidebar.number_input(
#             key, value=float(value), format="%.1f"
#         )

#     st.sidebar.divider()
#     predict_button = st.sidebar.button("🔬 Lancer la Comparaison", use_container_width=True)

#     # --- Affichage Principal ---
#     main_container = st.container()

#     if predict_button:
#         # Prétraitement avec les valeurs actuelles du session_state
#         input_data = st.session_state.sensor_values
#         processed_classic = preprocess_for_classic_models(input_data)
#         processed_lstm_unscaled = preprocess_for_lstm(input_data, timesteps=20)

#         scaler_lstm = loaded_data['scalers'].get('lstm')
#         if scaler_lstm:
#             processed_lstm_scaled = scaler_lstm.transform(
#                 processed_lstm_unscaled.reshape(-1, processed_lstm_unscaled.shape[-1])
#             ).reshape(processed_lstm_unscaled.shape)
#         else:
#             processed_lstm_scaled = None

#         # Prédictions
#         all_results = predict_all_models(
#             X=processed_classic,
#             models_dict=loaded_data,
#             X_lstm=processed_lstm_scaled
#         )

#         main_container.header("Résultats par Modèle")
#         model_order = ['autoencoder', 'lstm', 'isolation_forest', 'ocsvm']
#         model_names_map = {
#             'autoencoder': 'Autoencodeur Dense', 'lstm': 'Autoencodeur LSTM',
#             'isolation_forest': 'Isolation Forest', 'ocsvm': 'One-Class SVM'
#         }
#         cols = main_container.columns(len(model_order))

#         for i, model_key in enumerate(model_order):
#             if model_key in all_results:
#                 result = all_results[model_key]
#                 with cols[i].container():
#                     st.subheader(model_names_map.get(model_key, model_key))
#                     verdict = "Anomalie" if result.get('prediction') == 1 else "Normal"
#                     icon = "🚨" if verdict == "Anomalie" else "✅"
#                     st.markdown(f"**Verdict : {icon} {verdict}**")
#                     score_label = "MSE" if 'mse' in result else "Score"
#                     score_value = result.get('mse', result.get('score', 0))
#                     st.metric(label=f"Score ({score_label})", value=f"{score_value:.4f}")
#                     st.metric(label="Confiance", value=f"{result.get('confidence', 0):.1f} %")

