import os
import urllib.parse

# --- CONFIGURATION SYSTÈME ANTI-CRASH (Indispensable sur Debian/Kali) ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine

# --- CHARGEMENT SÉCURISÉ DE TENSORFLOW ---
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Configuration de la page
st.set_page_config(layout="wide", page_title="IA Prévision Ventes")
st.title("🚀 Plateforme IA de Prévision des Ventes")

# --- BARRE LATÉRALE : STATUS & CONFIG ---
st.sidebar.header("🛡️ État du Système")
if TF_AVAILABLE:
    st.sidebar.success("✅ Deep Learning (LSTM) opérationnel")
else:
    st.sidebar.error("⚠️ Note : Deep Learning (LSTM) indisponible sur ce système.")
    st.sidebar.info("L'application utilise les modèles ML classiques (XGBoost/RF).")

st.sidebar.divider()
st.sidebar.header("🔌 Connexion PostgreSQL")
host = st.sidebar.text_input("Host", "localhost")
port = st.sidebar.text_input("Port", "5432")
db = st.sidebar.text_input("Database", "pred_vente")
user = st.sidebar.text_input("User", "postgres")
password = st.sidebar.text_input("Password", type="password")

# Initialisation Session State pour garder les données en mémoire
if 'df' not in st.session_state:
    st.session_state.df = None

# =========================
# SOURCE DES DONNÉES
# =========================
source = st.radio("Source des données", ["Upload CSV/Excel", "Base de données PostgreSQL"])

if "Upload" in source:
    file = st.file_uploader("Importer un fichier", type=["csv","xlsx"])
    if file:
        try:
            if file.name.endswith("csv"):
                st.session_state.df = pd.read_csv(file, sep=None, engine='python')
            else:
                st.session_state.df = pd.read_excel(file)
            st.success("Données chargées depuis le fichier !")
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")

else:
    table_name = st.text_input("Nom de la table SQL (ex: ventes)")
    if st.button("Charger depuis PostgreSQL"):
        if not password:
            st.warning("Veuillez saisir le mot de passe dans la barre latérale.")
        else:
            try:
                # Encodage du mot de passe pour gérer les caractères spéciaux (@, #, etc.)
                encoded_password = urllib.parse.quote_plus(password)
                conn_str = f"postgresql://{user}:{encoded_password}@{host}:{port}/{db}"
                engine = create_engine(conn_str)
                
                # Chargement des données
                st.session_state.df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
                st.success("Données récupérées avec succès !")
            except Exception as e:
                st.error(f"Erreur de connexion : {e}")

# =========================
# ANALYSE ET PRÉVISIONS
# =========================
df = st.session_state.df

if df is not None:
    # Nettoyage standard des colonnes
    df.columns = [c.lower().strip() for c in df.columns]
    required = ['date', 'sales', 'store', 'product']
    
    if all(col in df.columns for col in required):
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        # Agrégation mensuelle
        df_grouped = df.groupby(['year', 'month', 'store', 'product'])['sales'].sum().reset_index()

        st.divider()
        c1, c2 = st.columns(2)
        sel_store = c1.selectbox("Choisir le Magasin", df_grouped['store'].unique())
        sel_prod = c2.selectbox("Choisir le Produit", df_grouped['product'].unique())

        # Filtrage selon la sélection
        df_filtered = df_grouped[(df_grouped['store'] == sel_store) & (df_grouped['product'] == sel_prod)].copy()

        if len(df_filtered) >= 6:
            st.subheader(f"📊 Analyse : {sel_prod} à {sel_store}")
            
            # --- MODÈLES MACHINE LEARNING ---
            X = df_filtered[['year', 'month']]
            y = df_filtered['sales']
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

            models = {
                "Forêt Aléatoire": RandomForestRegressor(n_estimators=100),
                "XGBoost": XGBRegressor()
            }

            best_model = None
            min_rmse = float('inf')
            
            cols = st.columns(len(models))
            for i, (name, model) in enumerate(models.items()):
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                cols[i].metric(name, f"{rmse:.2f} RMSE")
                if rmse < min_rmse:
                    min_rmse, best_model = rmse, model

            # --- SECTION LSTM (GÉRÉE) ---
            st.divider()
            st.subheader("🤖 Deep Learning (LSTM)")
            if TF_AVAILABLE:
                try:
                    series = df_filtered['sales'].values.astype('float32')
                    seq_len = 3
                    if len(series) > seq_len:
                        X_l, y_l = [], []
                        for i in range(len(series)-seq_len):
                            X_l.append(series[i:i+seq_len])
                            y_l.append(series[i+seq_len])
                        X_l = np.array(X_l).reshape((-1, seq_len, 1))
                        
                        model_lstm = Sequential([
                            Input(shape=(seq_len, 1)),
                            LSTM(50, activation='relu'),
                            Dense(1)
                        ])
                        model_lstm.compile(optimizer='adam', loss='mse')
                        with st.spinner('Entraînement du modèle temporel...'):
                            model_lstm.fit(X_l, np.array(y_l), epochs=50, verbose=0)
                        st.success("Modèle LSTM prêt !")
                except Exception as e:
                    st.warning("Le modèle LSTM a rencontré une erreur. Passage en mode dégradé.")
            else:
                st.info("💡 Mode LSTM désactivé pour stabilité. Les modèles XGBoost et RF sont utilisés.")

            # --- GRAPHIQUE HISTORIQUE ---
            st.divider()
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(df_filtered['sales'].values, marker='o', color='#1f77b4', label="Réel")
            ax.set_title("Évolution historique des ventes")
            ax.legend()
            st.pyplot(fig)

            # --- PRÉVISIONS FUTURES ---
            st.subheader("🔮 Prévisions des prochains mois")
            h_slider = st.slider("Nombre de mois à prédire", 1, 6, 3)
            
            last_y = int(df_filtered['year'].iloc[-1])
            last_m = int(df_filtered['month'].iloc[-1])
            
            future_data = []
            for i in range(1, h_slider + 1):
                m_t = (last_m + i - 1) % 12 + 1
                y_t = last_y + (last_m + i - 1) // 12
                pred = best_model.predict([[y_t, m_t]])[0]
                future_data.append({
                    "Mois/Année": f"{m_t}/{y_t}", 
                    "Ventes Prédites": round(max(0, pred), 1)
                })
            
            st.table(pd.DataFrame(future_data))
            
        else:
            st.warning("⚠️ Pas assez de données historiques (minimum 6 mois requis).")
    else:
        st.error(f"Format de table incorrect. Colonnes requises : {required}")
else:
    st.info("💡 En attente de données. Veuillez charger un fichier ou vous connecter à PostgreSQL.")
