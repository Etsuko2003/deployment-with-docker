"""
Page Streamlit pour la visualisation exploratoire des données.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# --- Configuration du Path ---
src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils import local_css
from visualization.sensor_plots import (
    plot_temporal_evolution,
    plot_temperature_scatter,
    plot_speed_torque
)
from visualization.correlation_plots import (
    plot_wear_distribution,
    plot_correlation_heatmaps,
    plot_feature_importance_comparison
)

# --- Configuration de la page et Style ---
st.set_page_config(
    page_title="Visualisation des Capteurs",
    page_icon="📊",
    layout="wide"
)
local_css("style.css")

# --- Chargement des données ---
@st.cache_data
def load_and_prepare_data():
    """Charge le dataset, effectue le feature engineering et le met en cache."""
    try:
        df = pd.read_csv('data/ai4i2020.csv')
        # Feature engineering
        if 'temp_diff' not in df.columns:
            df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
        if 'power_kw' not in df.columns:
            df['power_kw'] = (df['Torque [Nm]'] * df['Rotational speed [rpm]']) / 9550
        if 'mechanical_stress' not in df.columns:
            df['mechanical_stress'] = df['Torque [Nm]'] * (1 + df['Tool wear [min]'] / 250)
        return df
    except FileNotFoundError:
        st.error("Fichier 'data/ai4i2020.csv' non trouvé.")
        return None

# --- Interface Principale ---
st.title("📊 Tableau de Bord - Analyse Exploratoire des Données")
st.markdown("Cette page présente une analyse visuelle du jeu de données de maintenance prédictive.")

# --- Lien vers le rapport YData Profiling ---
st.info(
    "📊 **Analyse Exploratoire Détaillée** : Pour consulter le rapport complet "
    "généré par YData Profiling (distributions, corrélations, valeurs manquantes, etc.), "
    "cliquez sur le bouton ci-dessous."
)

# Bouton pour afficher le rapport
import streamlit.components.v1 as components
from pathlib import Path

# Chemin vers le fichier HTML
html_path = Path(__file__).parent.parent / 'assets' / 'output.html'

# Initialiser l'état pour afficher/masquer le rapport
if 'show_profiling' not in st.session_state:
    st.session_state.show_profiling = False

if html_path.exists():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not st.session_state.show_profiling:
            if st.button("📈 Ouvrir le Rapport YData Profiling", use_container_width=True, type="primary"):
                st.session_state.show_profiling = True
                st.rerun()
        else:
            if st.button("❌ Fermer le Rapport", use_container_width=True, type="secondary"):
                st.session_state.show_profiling = False
                st.rerun()
    
    # Afficher le rapport si demandé
    if st.session_state.show_profiling:
        st.markdown("### 📊 Rapport YData Profiling")
        # Lire le contenu HTML
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Afficher dans un iframe avec hauteur augmentée
        components.html(html_content, height=1200, scrolling=True)
else:
    st.error(f"❌ Le fichier de rapport n'a pas été trouvé : {html_path}")

st.markdown("---")


df = load_and_prepare_data()

if df is not None:
    # --- Métriques Clés ---
    st.subheader("Statistiques Générales")
    col1, col2, col3 = st.columns(3)
    n_anomalies = df['Machine failure'].sum()
    with col1:
        st.metric("Total de Mesures", f"{len(df):,}")
    with col2:
        st.metric("Anomalies Détectées", n_anomalies, f"{n_anomalies/len(df)*100:.1f}%")
    with col3:
        st.metric("Nombre de Capteurs", 5)

    # --- Section 1: Évolution Temporelle ---
    with st.container():
        st.header("Évolution Temporelle des Capteurs")
        st.markdown("Visualisation de l'évolution de tous les capteurs au fil du temps avec marquage des anomalies.")
        fig1 = plot_temporal_evolution(df)
        st.plotly_chart(fig1, use_container_width=True)

    # --- Section 2: Analyses de Corrélation ---
    st.header("Analyses de Corrélation")
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.subheader("Température Air vs. Processus")
            fig2 = plot_temperature_scatter(df)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("💡 Les points éloignés de la diagonale indiquent une forte différence de température.")
    with col2:
        with st.container():
            st.subheader("Vitesse de Rotation vs. Couple")
            fig3 = plot_speed_torque(df)
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("💡 La couleur représente la puissance (kW).")

    # --- Section 3: Distributions et Importance ---
    st.header("Distributions et Importance des Features")
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.subheader("Distribution de l'Usure de l'Outil")
            fig4 = plot_wear_distribution(df)
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("💡 La ligne verticale marque le seuil 'high_wear' (200 min).")
    with col2:
        with st.container():
            st.subheader("Comparaison des Features")
            fig5 = plot_feature_importance_comparison(df)
            st.plotly_chart(fig5, use_container_width=True)
            st.caption("💡 Comparaison des moyennes normalisées entre points normaux et anomalies.")

    # --- Section 4: Matrices de Corrélation ---
    with st.container():
        st.header("Matrices de Corrélation")
        st.markdown("Comparaison des structures de corrélation entre les états normaux et anormaux.")
        fig6 = plot_correlation_heatmaps(df)
        st.pyplot(fig6)
        st.caption("💡 Des changements dans les corrélations peuvent indiquer des comportements anormaux.")
else:
    st.warning("Le chargement des données a échoué. Impossible d'afficher les visualisations.")