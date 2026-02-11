"""
Module de visualisation pour les courbes des capteurs.

Ce module contient les fonctions pour créer des graphiques interactifs
montrant l'évolution temporelle des capteurs et les anomalies détectées.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_temporal_evolution(df, anomaly_column='Machine failure'):
    """
    Crée un graphique multi-axes montrant l'évolution temporelle de tous les capteurs.
    
    Paramètres :
    -----------
    df : pd.DataFrame
        DataFrame avec les colonnes des capteurs et la colonne d'anomalie
    anomaly_column : str
        Nom de la colonne indiquant les anomalies (0=Normal, 1=Anomalie)
    
    Retourne :
    ---------
    plotly.graph_objects.Figure
        Graphique interactif Plotly
    """
    # Créer la figure avec axes secondaires
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    # Axe principal (gauche) : Températures
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Air temperature [K]'],
            name='Air Temperature',
            line=dict(color='lightblue', width=2),
            mode='lines'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Process temperature [K]'],
            name='Process Temperature',
            line=dict(color='orange', width=2),
            mode='lines'
        ),
        secondary_y=False
    )
    
    # Axe secondaire (droit) : Vitesse, Couple, Usure
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Rotational speed [rpm]'],
            name='Rotational Speed',
            line=dict(color='green', width=1.5, dash='dot'),
            mode='lines',
            yaxis='y2'
        ),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Torque [Nm]'],
            name='Torque',
            line=dict(color='purple', width=1.5, dash='dash'),
            mode='lines',
            yaxis='y3'
        ),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Tool wear [min]'],
            name='Tool Wear',
            line=dict(color='red', width=1.5),
            mode='lines',
            yaxis='y4'
        ),
        secondary_y=True
    )
    
    # Marquer les anomalies
    if anomaly_column in df.columns:
        anomalies = df[df[anomaly_column] == 1]
        if len(anomalies) > 0:
            fig.add_trace(
                go.Scatter(
                    x=anomalies.index,
                    y=anomalies['Process temperature [K]'],
                    name='Anomalies',
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='x',
                        line=dict(width=2, color='darkred')
                    )
                ),
                secondary_y=False
            )
    
    # Mise en forme
    fig.update_layout(
        title=dict(
            text='Évolution Temporelle des Capteurs',
            font=dict(size=20, family='Arial Black')
        ),
        xaxis_title='Index Temporel',
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_yaxes(title_text="Température (K)", secondary_y=False)
    fig.update_yaxes(title_text="Autres Capteurs (normalisés)", secondary_y=True)
    
    return fig


def plot_temperature_scatter(df, anomaly_column='Machine failure'):
    """
    Crée un scatter plot Air Temperature vs Process Temperature.
    
    Paramètres :
    -----------
    df : pd.DataFrame
        DataFrame avec les colonnes de température et d'anomalie
    anomaly_column : str
        Nom de la colonne indiquant les anomalies
    
    Retourne :
    ---------
    plotly.graph_objects.Figure
        Graphique interactif Plotly
    """
    # Séparer normaux et anomalies
    df_normal = df[df[anomaly_column] == 0]
    df_anomaly = df[df[anomaly_column] == 1]
    
    fig = go.Figure()
    
    # Points normaux
    fig.add_trace(go.Scatter(
        x=df_normal['Air temperature [K]'],
        y=df_normal['Process temperature [K]'],
        mode='markers',
        name='Normal',
        marker=dict(
            color='blue',
            size=5,
            opacity=0.6
        )
    ))
    
    # Points anomalies
    if len(df_anomaly) > 0:
        fig.add_trace(go.Scatter(
            x=df_anomaly['Air temperature [K]'],
            y=df_anomaly['Process temperature [K]'],
            mode='markers',
            name='Anomalie',
            marker=dict(
                color='red',
                size=8,
                opacity=0.8,
                symbol='x'
            )
        ))
    
    # Ligne de référence (Air = Process)
    min_temp = df['Air temperature [K]'].min()
    max_temp = df['Process temperature [K]'].max()
    fig.add_trace(go.Scatter(
        x=[min_temp, max_temp],
        y=[min_temp, max_temp],
        mode='lines',
        name='Air = Process',
        line=dict(color='gray', dash='dash', width=1)
    ))
    
    # Mise en forme
    fig.update_layout(
        title='Température Air vs Process',
        xaxis_title='Air Temperature (K)',
        yaxis_title='Process Temperature (K)',
        height=500,
        hovermode='closest'
    )
    
    return fig


def plot_speed_torque(df, anomaly_column='Machine failure'):
    """
    Crée un scatter plot Rotational Speed vs Torque.
    
    Paramètres :
    -----------
    df : pd.DataFrame
        DataFrame avec les colonnes de vitesse, couple et anomalie
    anomaly_column : str
        Nom de la colonne indiquant les anomalies
    
    Retourne :
    ---------
    plotly.graph_objects.Figure
        Graphique interactif Plotly
    """
    # Calculer la puissance si pas déjà présente
    if 'power_kw' not in df.columns:
        df['power_kw'] = (df['Torque [Nm]'] * df['Rotational speed [rpm]']) / 9550
    
    # Séparer normaux et anomalies
    df_normal = df[df[anomaly_column] == 0]
    df_anomaly = df[df[anomaly_column] == 1]
    
    fig = go.Figure()
    
    # Points normaux
    fig.add_trace(go.Scatter(
        x=df_normal['Rotational speed [rpm]'],
        y=df_normal['Torque [Nm]'],
        mode='markers',
        name='Normal',
        marker=dict(
            color=df_normal['power_kw'],
            colorscale='Blues',
            size=6,
            opacity=0.6,
            colorbar=dict(title="Power (kW)", x=1.15)
        ),
        text=[f"Power: {p:.2f} kW" for p in df_normal['power_kw']],
        hovertemplate='Speed: %{x}<br>Torque: %{y}<br>%{text}<extra></extra>'
    ))
    
    # Points anomalies
    if len(df_anomaly) > 0:
        fig.add_trace(go.Scatter(
            x=df_anomaly['Rotational speed [rpm]'],
            y=df_anomaly['Torque [Nm]'],
            mode='markers',
            name='Anomalie',
            marker=dict(
                color='red',
                size=10,
                opacity=0.8,
                symbol='x'
            ),
            text=[f"Power: {p:.2f} kW" for p in df_anomaly['power_kw']],
            hovertemplate='Speed: %{x}<br>Torque: %{y}<br>%{text}<extra></extra>'
        ))
    
    # Mise en forme
    fig.update_layout(
        title='Vitesse de Rotation vs Couple (coloré par Puissance)',
        xaxis_title='Rotational Speed (rpm)',
        yaxis_title='Torque (Nm)',
        height=500,
        hovermode='closest'
    )
    
    return fig
