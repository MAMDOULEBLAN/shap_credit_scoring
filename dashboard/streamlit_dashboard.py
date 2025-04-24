
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="Dashboard Scoring Crédit", layout="wide")
st.title("📊 Tableau de bord - Décision de crédit")

api_url = "https://shap-credit-api-mamdou.herokuapp.com"

try:
    id_response = requests.get(f"{api_url}/api/ids")
    id_response.raise_for_status()
    ids = id_response.json().get("ids", [])
    client_id = st.selectbox("Sélectionnez un identifiant client :", ids)
except Exception as e:
    st.error(f"Erreur lors de la récupération des IDs : {e}")
    st.stop()

if st.button("Obtenir la prédiction via API"):
    try:
        response = requests.post(f"{api_url}/api/predict", json={"id": client_id})
        response.raise_for_status()
        result = response.json()

        prediction = result["prediction"]
        probability = result["probability"]
        shap_values = result.get("shap_values", [])
        features = result.get("features", [])

        if prediction == 0:
            st.success(f"✅ Prêt accordé")
        else:
            st.error(f"❌ Prêt refusé")
        st.write(f"**Probabilité de défaut : {round(probability * 100, 2)} %**")

        if shap_values and features:
            st.subheader("🔍 Explication SHAP des variables principales")
            shap_df = pd.DataFrame({"Feature": features, "SHAP value": shap_values})
            shap_df = shap_df.sort_values(by="SHAP value", key=abs, ascending=False).head(5)

            fig, ax = plt.subplots()
            shap_df.plot(kind="barh", x="Feature", y="SHAP value", ax=ax, legend=False)
            ax.set_title("Impact des variables sur la prédiction")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur lors de la connexion à l’API : {e}")
