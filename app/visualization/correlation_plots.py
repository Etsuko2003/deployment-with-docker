"""
Module de visualisation pour les analyses de corrélation.

Ce module contient les fonctions pour créer des graphiques d'analyse
de corrélation entre les capteurs et les anomalies.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def plot_wear_distribution(df, anomaly_column='Machine failure'):
    """
    Crée un histogramme comparant la distribution de l'usure pour normaux vs anomalies.
    
    Paramètres :
    -----------
    df : pd.DataFrame
        DataFrame avec la colonne Tool wear et anomalie
    anomaly_column : str
        Nom de la colonne indiquant les anomalies
    
    Retourne :
    ---------
    plotly.graph_objects.Figure
        Graphique interactif Plotly
    """
    # Séparer normaux et anomalies
    df_normal = df[df[anomaly_column] == 0]['Tool wear [min]']
    df_anomaly = df[df[anomaly_column] == 1]['Tool wear [min]']
    
    fig = go.Figure()
    
    # Histogramme normaux
    fig.add_trace(go.Histogram(
        x=df_normal,
        name='Normal',
        marker_color='blue',
        opacity=0.6,
        nbinsx=30
    ))
    
    # Histogramme anomalies
    if len(df_anomaly) > 0:
        fig.add_trace(go.Histogram(
            x=df_anomaly,
            name='Anomalie',
            marker_color='red',
            opacity=0.6,
            nbinsx=30
        ))
    
    # Ligne verticale au seuil 200 min
    fig.add_vline(
        x=200,
        line_dash="dash",
        line_color="black",
        annotation_text="Seuil high_wear (200 min)",
        annotation_position="top right"
    )
    
    # Mise en forme
    fig.update_layout(
        title='Distribution de l\'Usure de l\'Outil',
        xaxis_title='Tool Wear (min)',
        yaxis_title='Fréquence',
        barmode='overlay',
        height=500,
        showlegend=True
    )
    
    return fig


def plot_correlation_heatmaps(df, anomaly_column='Machine failure'):
    """
    Crée deux heatmaps de corrélation côte à côte (normaux vs anomalies).
    
    Paramètres :
    -----------
    df : pd.DataFrame
        DataFrame avec toutes les features et la colonne d'anomalie
    anomaly_column : str
        Nom de la colonne indiquant les anomalies
    
    Retourne :
    ---------
    tuple
        (fig_normal, fig_anomaly) : Deux figures matplotlib
    """
    # Features à analyser
    features = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    # Ajouter features engineerées si présentes
    if 'temp_diff' in df.columns:
        features.extend(['temp_diff', 'power_kw', 'mechanical_stress'])
    
    # Filtrer les features disponibles
    available_features = [f for f in features if f in df.columns]
    
    # Séparer normaux et anomalies
    df_normal = df[df[anomaly_column] == 0][available_features]
    df_anomaly = df[df[anomaly_column] == 1][available_features]
    
    # Calculer les corrélations
    corr_normal = df_normal.corr()
    corr_anomaly = df_anomaly.corr() if len(df_anomaly) > 1 else pd.DataFrame()
    
    # Créer les heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap normaux
    sns.heatmap(
        corr_normal,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=axes[0],
        cbar_kws={'label': 'Corrélation'}
    )
    axes[0].set_title('Corrélations - Points Normaux', fontsize=14, fontweight='bold')
    
    # Heatmap anomalies
    if len(corr_anomaly) > 0:
        sns.heatmap(
            corr_anomaly,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=axes[1],
            cbar_kws={'label': 'Corrélation'}
        )
        axes[1].set_title('Corrélations - Anomalies', fontsize=14, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'Pas assez d\'anomalies', 
                     ha='center', va='center', fontsize=16)
        axes[1].set_title('Corrélations - Anomalies', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def plot_feature_importance_comparison(df, anomaly_column='Machine failure'):
    """
    Compare les moyennes des features entre normaux et anomalies.
    
    Paramètres :
    -----------
    df : pd.DataFrame
        DataFrame avec toutes les features
    anomaly_column : str
        Nom de la colonne indiquant les anomalies
    
    Retourne :
    ---------
    plotly.graph_objects.Figure
        Graphique en barres comparatif
    """
    # Features à comparer
    features = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    # Ajouter features engineerées si présentes
    if 'temp_diff' in df.columns:
        features.extend(['temp_diff', 'power_kw', 'mechanical_stress'])
    
    # Filtrer les features disponibles
    available_features = [f for f in features if f in df.columns]
    
    # Calculer les moyennes
    means_normal = df[df[anomaly_column] == 0][available_features].mean()
    means_anomaly = df[df[anomaly_column] == 1][available_features].mean()
    
    # Normaliser pour comparaison
    means_normal_norm = (means_normal - means_normal.min()) / (means_normal.max() - means_normal.min())
    means_anomaly_norm = (means_anomaly - means_anomaly.min()) / (means_anomaly.max() - means_anomaly.min())
    
    fig = go.Figure()
    
    # Barres normaux
    fig.add_trace(go.Bar(
        x=available_features,
        y=means_normal_norm,
        name='Normal',
        marker_color='blue',
        opacity=0.7
    ))
    
    # Barres anomalies
    fig.add_trace(go.Bar(
        x=available_features,
        y=means_anomaly_norm,
        name='Anomalie',
        marker_color='red',
        opacity=0.7
    ))
    
    # Mise en forme
    fig.update_layout(
        title='Comparaison des Moyennes (Normalisées)',
        xaxis_title='Features',
        yaxis_title='Valeur Normalisée',
        barmode='group',
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig
