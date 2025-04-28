import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Dashboard Crédit", layout="centered")
st.title("📊 Dashboard - Décision de crédit")

# Choisir l'environnement
env = st.sidebar.selectbox("Sélectionner l'environnement :", ["Local", "Cloud"])

if env == "Local":
    API_URL = "http://localhost:5000"
else:
    API_URL = "https://shap-credit-api-mamdou-0a39fd6254f1.herokuapp.com"

# 🔁 Récupérer la liste des IDs depuis l'API
try:
    id_response = requests.get(f"{API_URL}/api/ids")
    id_response.raise_for_status()
    ids = id_response.json().get("ids", [])
    client_id = st.selectbox("Sélectionnez un identifiant client :", ids)
except Exception as e:
    st.error(f"Erreur lors de la récupération des IDs : {e}")
    st.stop()

if st.button("Obtenir la prédiction via API"):
    try:
        response = requests.post(f"{API_URL}/api/predict", json={"id_client": int(client_id)})
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            proba = result["probability"]

            if prediction == 1:
                st.error("❌ Prêt NON accordé")
            else:
                st.success("✅ Prêt accordé")

            st.metric(label="Probabilité de défaut", value=f"{proba*100:.2f} %")

            st.subheader("🧒 Comparaison client vs moyenne (10 variables clés)")
            selected_features = list(result["shap_values"].keys())

            df_compare = pd.DataFrame({
                "Valeur client": {feat: result["features"][feat] for feat in selected_features},
                "Moyenne globale": {feat: result["global_means"][feat] for feat in selected_features}
            })
            st.dataframe(df_compare)

            st.subheader("📉 Visualisation comparative")
            fig, ax = plt.subplots(figsize=(8, 4))
            df_compare.plot(kind="bar", ax=ax)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("🔍 Top 10 variables impactant la prédiction")
            shap_df = pd.DataFrame.from_dict(result["shap_values"], orient="index", columns=["SHAP value"])

            # Trier par valeur absolue décroissante et prendre les 10 premiers
            shap_df = shap_df.reindex(shap_df["SHAP value"].abs().sort_values(ascending=False).index)
            shap_df_top10 = shap_df.head(10)

            # Ajouter une petite valeur epsilon pour éviter que des SHAP nuls cachent le graphique
            epsilon = 1e-6
            shap_df_top10["SHAP value"] = shap_df_top10["SHAP value"].apply(lambda x: x if abs(x) > epsilon else epsilon)

            fig2, ax2 = plt.subplots()
            shap_df_top10.plot(kind="barh", legend=False, ax=ax2, color="skyblue")
            ax2.set_title("Top 10 variables impactant la prédiction")
            plt.tight_layout()
            st.pyplot(fig2)

        else:
            st.warning(f"Erreur API : {response.status_code}")
            st.write(response.json())
    except Exception as e:
        st.error(f"Erreur lors de la connexion à l'API : {e}")
