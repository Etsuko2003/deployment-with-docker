"""
Page Streamlit pour l'analyse de l'espace latent des autoencodeurs avec clustering.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tensorflow import keras
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration du Path ---
src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils import local_css
from model_loader import load_all_models
from preprocessing import preprocess_for_classic_models, preprocess_for_lstm

# --- Configuration de la page ---
st.set_page_config(
    page_title="Analyse de l'Espace Latent",
    page_icon="🧬",
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

@st.cache_data
def preprocess_dataset_dense(_df):
    """Prétraite tout le dataset pour l'autoencodeur Dense."""
    data_dicts = _df.to_dict(orient='records')
    processed_list = [preprocess_for_classic_models(d) for d in data_dicts]
    return pd.concat(processed_list, ignore_index=True)

@st.cache_data
def preprocess_dataset_lstm(_df, timesteps=20):
    """Prétraite tout le dataset pour l'autoencodeur LSTM."""
    data_dicts = _df.to_dict(orient='records')
    processed_sequences = [preprocess_for_lstm(d, timesteps=timesteps) for d in data_dicts]
    return np.vstack(processed_sequences)

# --- Interface Principale ---
st.title("🧬 Analyse de l'Espace Latent des Autoencodeurs")
st.markdown("""
Cette page vous permet de visualiser comment les modèles d'autoencodeurs "voient" 
les données dans un espace compressé. Les anomalies devraient idéalement former des 
clusters distincts des données normales.
""")

# Charger les données et les modèles
models_data = load_models()
df = load_data()

if df is not None and models_data:
    models = models_data['models']
    scalers = models_data['scalers']
    
    # Sidebar
    st.sidebar.header("🛠️ Options d'Analyse")
    
    # Sélection du modèle
    model_choice = st.sidebar.radio(
        "Choisissez un modèle d'autoencodeur",
        ["Autoencodeur Dense", "Autoencodeur LSTM"]
    )
    
    model_key = 'autoencoder' if 'Dense' in model_choice else 'lstm'
    
    st.sidebar.divider()
    
    # Options spécifiques au modèle
    if model_key == 'lstm':
        st.sidebar.subheader("Réduction de Dimension")
        reduction_method = st.sidebar.selectbox(
            "Méthode de réduction",
            ["PCA", "t-SNE"]
        )
        
        if reduction_method == "PCA":
            n_components_pca = st.sidebar.slider("Nombre de composantes", 2, 10, 2, 1)
        else:  # t-SNE
            perplexity = st.sidebar.slider("Perplexity", 5, 50, 30, 5)
        
        st.sidebar.divider()
    
    # Clustering
    st.sidebar.subheader("Clustering")
    
    # Si LSTM avec t-SNE, désactiver DBSCAN
    if model_key == 'lstm' and reduction_method == "t-SNE":
        st.sidebar.info("ℹ️ DBSCAN désactivé pour t-SNE (distances non-euclidiennes)")
        clustering_method = "KMeans"
        n_clusters = st.sidebar.slider("Nombre de clusters", 2, 10, 3, 1)
    else:
        clustering_method = st.sidebar.selectbox(
            "Méthode de clustering",
            ["KMeans", "DBSCAN"]
        )
        
        if clustering_method == "KMeans":
            n_clusters = st.sidebar.slider("Nombre de clusters", 2, 10, 3, 1)
        else:  # DBSCAN
            eps = st.sidebar.slider("Eps (distance maximale)", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.sidebar.slider("Min samples", 5, 50, 10, 5)
    
    analyze_button = st.sidebar.button("🚀 Lancer l'Analyse", use_container_width=True)
    
    # --- Analyse ---
    if analyze_button:
        try:
            # 1. Prétraitement
            with st.spinner("Prétraitement des données..."):
                if model_key == 'autoencoder':
                    processed_data = preprocess_dataset_dense(df)
                    X_scaled = scalers['autoencoder'].transform(processed_data)
                else:  # LSTM
                    X_lstm = preprocess_dataset_lstm(df, timesteps=20)
                    X_scaled = scalers['lstm'].transform(
                        X_lstm.reshape(-1, X_lstm.shape[-1])
                    ).reshape(X_lstm.shape)
                
                labels = df['Machine failure'].values
            
            # 2. Extraction de l'espace latent
            with st.spinner("Extraction de l'espace latent..."):
                autoencoder_model = models[model_key]
                
                # Nom de la couche selon le modèle
                if model_key == 'autoencoder':
                    latent_layer_name = "espace_latent"
                else:  # LSTM
                    latent_layer_name = "dense_10"  # Couche dense après LSTM encoder
                
                try:
                    encoder = keras.Model(
                        inputs=autoencoder_model.input,
                        outputs=autoencoder_model.get_layer(latent_layer_name).output
                    )
                except ValueError:
                    # Si la couche n'existe pas, afficher les couches disponibles
                    st.error(f"Couche '{latent_layer_name}' non trouvée.")
                    st.info(f"Couches disponibles : {[layer.name for layer in autoencoder_model.layers]}")
                    raise
                
                Z = encoder.predict(X_scaled, verbose=0)
                
                # Si 3D (LSTM avec timesteps), moyenner
                if len(Z.shape) == 3:
                    Z = np.mean(Z, axis=1)
                
                st.success(f"✅ Espace latent extrait : {Z.shape}")

            
            # 3. Réduction de dimension (si LSTM)
            if model_key == 'lstm':
                with st.spinner(f"Réduction de dimension avec {reduction_method}..."):
                    if reduction_method == "PCA":
                        pca = PCA(n_components=n_components_pca)
                        Z_reduced = pca.fit_transform(Z)
                        
                        # Variance expliquée
                        variance_explained = pca.explained_variance_ratio_
                        cumulative_variance = np.cumsum(variance_explained)
                        
                        st.header("📊 Analyse PCA")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Variance expliquée cumulée",
                                f"{cumulative_variance[-1]*100:.2f}%"
                            )
                        with col2:
                            st.metric(
                                f"Variance des {n_components_pca} composantes",
                                f"{', '.join([f'{v*100:.1f}%' for v in variance_explained])}"
                            )
                        
                        # Graphe de la variance expliquée
                        fig_var = go.Figure()
                        fig_var.add_trace(go.Bar(
                            x=[f'PC{i+1}' for i in range(len(variance_explained))],
                            y=variance_explained * 100,
                            name='Variance individuelle'
                        ))
                        fig_var.add_trace(go.Scatter(
                            x=[f'PC{i+1}' for i in range(len(cumulative_variance))],
                            y=cumulative_variance * 100,
                            name='Variance cumulée',
                            mode='lines+markers',
                            yaxis='y2'
                        ))
                        fig_var.update_layout(
                            title="Variance Expliquée par Composante Principale",
                            xaxis_title="Composante",
                            yaxis_title="Variance Expliquée (%)",
                            yaxis2=dict(
                                title="Variance Cumulée (%)",
                                overlaying='y',
                                side='right'
                            ),
                            height=400
                        )
                        st.plotly_chart(fig_var, use_container_width=True)
                        
                        Z = Z_reduced
                    else:  # t-SNE
                        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=300)
                        Z_reduced = tsne.fit_transform(Z)
                        Z = Z_reduced
                        st.success(f"✅ Réduction t-SNE terminée (perplexity={perplexity})")
            
            # 4. Calcul des MSE
            with st.spinner("Calcul des erreurs de reconstruction..."):
                X_reconstructed = autoencoder_model.predict(X_scaled, verbose=0)
                if len(X_reconstructed.shape) == 3:  # LSTM
                    mse = np.mean(np.square(X_scaled - X_reconstructed), axis=(1, 2))
                else:
                    mse = np.mean(np.square(X_scaled - X_reconstructed), axis=1)
            
            # 5. Clustering
            st.header("📊 Résultats du Clustering")
            
            if clustering_method == "KMeans":
                # KMeans
                with st.spinner("Clustering KMeans en cours..."):
                    scaler_latent = StandardScaler()
                    Z_scaled = scaler_latent.fit_transform(Z)
                    
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
                    clusters = kmeans.fit_predict(Z_scaled)
                    
                    sil_score = silhouette_score(Z_scaled, clusters)
                    
                    # Caractérisation des clusters
                    cluster_mse = {}
                    for c in range(n_clusters):
                        cluster_mse[c] = mse[clusters == c].mean()
                    
                    # Attribution des régimes (si 3 clusters)
                    if n_clusters == 3:
                        sorted_clusters = sorted(cluster_mse, key=cluster_mse.get)
                        cluster_to_regime = {
                            sorted_clusters[0]: "Normal",
                            sorted_clusters[1]: "Dégradé",
                            sorted_clusters[2]: "En panne"
                        }
                        regimes = np.array([cluster_to_regime[c] for c in clusters])
                    else:
                        cluster_to_regime = {c: f"Cluster {c}" for c in range(n_clusters)}
                        regimes = np.array([cluster_to_regime[c] for c in clusters])
                
                # Affichage des métriques
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Silhouette Score", f"{sil_score:.3f}")
                with col2:
                    st.metric("Nombre de clusters", n_clusters)
                
                # Tableau de caractérisation
                st.subheader("📈 Caractérisation des Régimes")
                
                # Explication pour KMeans (si 3 clusters)
                if n_clusters == 3:
                    st.info("""
                    **💡 Attribution des Régimes (KMeans) :**
                    
                    Les clusters sont triés par MSE moyenne croissante et assignés automatiquement :
                    - **Normal** : Cluster avec la MSE la plus faible (meilleure reconstruction)
                    - **Dégradé** : Cluster avec MSE intermédiaire
                    - **En panne** : Cluster avec la MSE la plus élevée (pire reconstruction)
                    
                    La MSE (Mean Squared Error) mesure l'erreur de reconstruction de l'autoencodeur.
                    """)
                
                stats_data = []
                for c in range(n_clusters):
                    mask = clusters == c
                    stats_data.append({
                        'Cluster': c,
                        'Régime': cluster_to_regime[c],
                        'Nombre de points': mask.sum(),
                        'MSE moyenne': f"{cluster_mse[c]:.6f}",
                        'Taux de panne (%)': f"{(labels[mask].mean() * 100):.2f}"
                    })
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

                
                # Visualisation
                st.subheader("🎨 Visualisation de l'Espace Latent")
                
                plot_df = pd.DataFrame({
                    'Dimension 1': Z[:, 0],
                    'Dimension 2': Z[:, 1],
                    'Régime': regimes,
                    'Cluster': clusters,
                    'MSE': mse
                })
                
                # Palette de couleurs
                if n_clusters == 3:
                    color_map = {'Normal': '#2ECC71', 'Dégradé': '#F39C12', 'En panne': '#E74C3C'}
                else:
                    color_map = None
                
                # Titre adapté au modèle
                if model_key == 'lstm':
                    title_suffix = f" ({reduction_method})"
                else:
                    title_suffix = ""
                
                fig1 = px.scatter(
                    plot_df,
                    x='Dimension 1',
                    y='Dimension 2',
                    color='Régime',
                    title=f"Clustering {model_choice}{title_suffix} (KMeans, {n_clusters} clusters)",
                    color_discrete_map=color_map,
                    hover_data=['MSE', 'Cluster']
                )
                fig1.update_traces(marker=dict(size=5, opacity=0.7))
                st.plotly_chart(fig1, use_container_width=True)
                
                # Plot 2: Coloré par Machine failure
                plot_df['Machine failure'] = labels
                plot_df['Statut'] = plot_df['Machine failure'].map({0: 'Normal', 1: 'Panne'})
                
                fig2 = px.scatter(
                    plot_df,
                    x='Dimension 1',
                    y='Dimension 2',
                    color='Statut',
                    title=f"Espace Latent {model_choice}{title_suffix} coloré par Machine Failure",
                    color_discrete_map={'Normal': '#3498DB', 'Panne': '#E74C3C'},
                    hover_data=['MSE', 'Régime']
                )
                fig2.update_traces(marker=dict(size=5, opacity=0.7))
                st.plotly_chart(fig2, use_container_width=True)
                
            else:  # DBSCAN
                # DBSCAN
                with st.spinner(f"Clustering DBSCAN (eps={eps}, min_samples={min_samples})..."):
                    scaler_latent = StandardScaler()
                    Z_scaled = scaler_latent.fit_transform(Z)
                    
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    clusters = dbscan.fit_predict(Z_scaled)
                    
                    n_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)
                    n_noise = list(clusters).count(-1)
                    
                    # Calcul du Silhouette Score (si au moins 2 clusters et pas que du bruit)
                    if n_clusters_found >= 2:
                        # Exclure le bruit pour le calcul du score
                        non_noise_mask = clusters != -1
                        if non_noise_mask.sum() > 0:
                            sil_score_dbscan = silhouette_score(
                                Z_scaled[non_noise_mask], 
                                clusters[non_noise_mask]
                            )
                        else:
                            sil_score_dbscan = None
                    else:
                        sil_score_dbscan = None
                
                # Métriques
                if sil_score_dbscan is not None:
                    col1, col2, col3, col4 = st.columns(4)
                    with col4:
                        st.metric("Silhouette Score", f"{sil_score_dbscan:.3f}")
                        # Interprétation du score
                        if sil_score_dbscan > 0.5:
                            st.success("✅ Bon clustering")
                        elif sil_score_dbscan > 0.25:
                            st.warning("⚠️ Clustering moyen")
                        else:
                            st.error("❌ Clustering faible")
                else:
                    col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Nombre de clusters", n_clusters_found)
                with col2:
                    st.metric("Points de bruit", n_noise)
                with col3:
                    st.metric("% de bruit", f"{(n_noise/len(clusters)*100):.1f}%")

                
                # Tableau de caractérisation
                st.subheader("📈 Caractérisation des Régimes (DBSCAN)")
                
                # Calcul du seuil : moyenne des MSE des données normales
                normal_mask = labels == 0
                mse_threshold = mse[normal_mask].mean()
                
                # Explication de la logique de classification
                st.info(f"""
                **💡 Logique de Classification des Clusters :**
                
                - **ANOMALIE (Bruit)** : Points isolés (cluster -1) détectés par DBSCAN comme ne faisant partie d'aucun cluster dense
                - **NORMAL** : Clusters avec MSE moyenne < seuil (faible erreur de reconstruction)
                - **DÉGRADÉ / PANNE** : Clusters avec MSE moyenne ≥ seuil (forte erreur de reconstruction)
                
                **Seuil utilisé** : {mse_threshold:.6f} (moyenne des MSE des données normales)
                
                La MSE (Mean Squared Error) mesure l'erreur de reconstruction de l'autoencodeur. 
                Plus la MSE est élevée, plus le comportement est anormal.
                """)
                
                stats_data = []
                unique_labels = sorted(set(clusters))
                
                for c in unique_labels:
                    mask = clusters == c
                    avg_mse = mse[mask].mean()
                    fail_rate = labels[mask].mean() * 100
                    count = mask.sum()
                    
                    if c == -1:
                        interp = "ANOMALIE (Bruit)"
                        justification = "Points isolés détectés par DBSCAN"
                    elif avg_mse < mse_threshold:
                        interp = "NORMAL"
                        justification = f"MSE ({avg_mse:.6f}) < seuil ({mse_threshold:.6f})"
                    else:
                        interp = "DÉGRADÉ / PANNE"
                        justification = f"MSE ({avg_mse:.6f}) ≥ seuil ({mse_threshold:.6f})"
                    
                    stats_data.append({
                        'Cluster': c,
                        'Interprétation': interp,
                        'Justification': justification,
                        'Nombre de points': count,
                        'MSE moyenne': f"{avg_mse:.6f}",
                        'Taux de panne (%)': f"{fail_rate:.2f}"
                    })
                
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
                
                # Visualisations
                st.subheader("🎨 Visualisations de l'Espace Latent (DBSCAN)")
                
                plot_df = pd.DataFrame({
                    'Dimension 1': Z[:, 0],
                    'Dimension 2': Z[:, 1],
                    'Cluster': clusters.astype(str),
                    'MSE': mse,
                    'Machine failure': labels
                })
                
                # Palette de couleurs
                colors = px.colors.qualitative.Plotly
                color_map = {str(c): colors[i % len(colors)] if c != -1 else '#000000' 
                            for i, c in enumerate(unique_labels)}
                
                # Titre adapté
                if model_key == 'lstm':
                    title_suffix = f" ({reduction_method})"
                else:
                    title_suffix = ""
                
                # Visualisation 1 : Par cluster
                
                fig = px.scatter(
                    plot_df,
                    x='Dimension 1',
                    y='Dimension 2',
                    color='Cluster',
                    title=f"Clustering {model_choice}{title_suffix} (DBSCAN, eps={eps})",
                    color_discrete_map=color_map,
                    hover_data=['MSE', 'Machine failure']
                )
                fig.update_traces(marker=dict(size=5, opacity=0.7))
                st.plotly_chart(fig, use_container_width=True)
                
                # Visualisation 2 : Normal vs Anomalie (Machine failure)
                plot_df['Statut'] = plot_df['Machine failure'].map({0: 'Normal', 1: 'Panne'})
                
                fig2 = px.scatter(
                    plot_df,
                    x='Dimension 1',
                    y='Dimension 2',
                    color='Statut',
                    title=f"Espace Latent {model_choice}{title_suffix} - Normal vs Anomalie",
                    color_discrete_map={'Normal': '#3498DB', 'Panne': '#E74C3C'},
                    hover_data=['MSE', 'Cluster']
                )
                fig2.update_traces(marker=dict(size=5, opacity=0.7))
                st.plotly_chart(fig2, use_container_width=True)
            
            st.success("✅ Analyse terminée !")
            
        except Exception as e:
            st.error(f"Une erreur est survenue durant l'analyse : {e}")
            st.exception(e)
    else:
        st.info("👈 Configurez les options dans le menu latéral et cliquez sur 'Lancer l'Analyse'.")
else:
    st.error("Le chargement des données ou des modèles a échoué. Vérifiez les chemins et les fichiers.")