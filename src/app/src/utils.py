"""
Module utilitaire pour l'application Streamlit.
"""

import streamlit as st

def local_css(file_name):
    """
    Injecte un fichier CSS local dans une application Streamlit.
    
    Paramètres :
    -----------
    file_name : str
        Chemin vers le fichier CSS (ex: "style.css")
    """
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Fichier CSS '{file_name}' non trouvé. Assurez-vous qu'il est dans le bon dossier.")

