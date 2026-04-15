# =========================
# INSTALLATION (requirements.txt)
# =========================
# streamlit
# pandas
# numpy
# scikit-learn
# xgboost
# matplotlib
# seaborn
# openpyxl
# reportlab

# =========================
# app.py
# =========================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Prévision des ventes", layout="wide")

st.title("📊 Application de prédiction des ventes mensuelles")

# =========================
# IMPORT DATA
# =========================
uploaded_file = st.file_uploader("Importer un fichier CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Aperçu des données")
    st.dataframe(df)

    # =========================
    # PREPROCESSING
    # =========================
    st.subheader("Prétraitement des données")

    df = df.dropna()

    # Exemple attendu : date, sales, store, product
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # Agrégation mensuelle
    df_grouped = df.groupby(['year', 'month', 'store', 'product'])['sales'].sum().reset_index()

    st.write("Données après agrégation:")
    st.dataframe(df_grouped)

    # =========================
    # FILTRES
    # =========================
    stores = df_grouped['store'].unique()
    products = df_grouped['product'].unique()

    selected_store = st.selectbox("Choisir un magasin", stores)
    selected_product = st.selectbox("Choisir un produit", products)

    filtered_df = df_grouped[(df_grouped['store'] == selected_store) &
                             (df_grouped['product'] == selected_product)]

    # =========================
    # FEATURES
    # =========================
    X = filtered_df[['year', 'month']]
    y = filtered_df['sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # =========================
    # MODELS
    # =========================
    st.subheader("Choisir un modèle")
    model_name = st.selectbox("Modèle", ["Linear Regression", "Random Forest", "XGBoost"])

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor()
    else:
        model = XGBRegressor()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # =========================
    # VISUALIZATION
    # =========================
    st.subheader("Visualisation")

    fig, ax = plt.subplots()
    ax.plot(y_test.values, label='Réel')
    ax.plot(predictions, label='Prédiction')
    ax.legend()

    st.pyplot(fig)

    # =========================
    # FUTURE PREDICTION
    # =========================
    st.subheader("Prévision future")

    future_year = st.number_input("Année future", value=2026)
    future_month = st.slider("Mois", 1, 12, 1)

    future_pred = model.predict([[future_year, future_month]])

    st.success(f"Prévision des ventes : {future_pred[0]:.2f}")

    # =========================
    # EXPORT CSV
    # =========================
    result_df = pd.DataFrame({
        'réel': y_test.values,
        'prédiction': predictions
    })

    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger CSV", csv, "predictions.csv", "text/csv")

    # =========================
    # EXPORT PDF
    # =========================
    def create_pdf():
        doc = SimpleDocTemplate("report.pdf")
        styles = getSampleStyleSheet()
        content = []

        content.append(Paragraph("Rapport de prévision des ventes", styles['Title']))
        content.append(Paragraph(f"Magasin: {selected_store}", styles['Normal']))
        content.append(Paragraph(f"Produit: {selected_product}", styles['Normal']))
        content.append(Paragraph(f"Prévision: {future_pred[0]:.2f}", styles['Normal']))

        doc.build(content)

    if st.button("Générer PDF"):
        create_pdf()
        with open("report.pdf", "rb") as f:
            st.download_button("Télécharger PDF", f, "report.pdf")

else:
    st.info("Veuillez importer un fichier pour commencer.")

# =========================
# RUN
# streamlit run app.py
# =========================
