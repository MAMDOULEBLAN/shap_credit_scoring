from flask import Flask, request, jsonify
import joblib
import shap
import pandas as pd

app = Flask(__name__)

# Charger le modèle et les données
model = joblib.load("model/model.pkl")
data = pd.read_csv("data/data.csv")

# Créer l'explainer SHAP (TreeExplainer pour LightGBM)
explainer = shap.TreeExplainer(model)

@app.route("/api/ids", methods=["GET"])
def get_ids():
    ids = data["SK_ID_CURR"].tolist()
    return jsonify({"ids": ids})

@app.route("/api/predict", methods=["POST"])
def predict():
    input_data = request.get_json()
    client_id = input_data.get("id_client")

    if client_id not in data["SK_ID_CURR"].values:
        return jsonify({"error": "Identifiant client introuvable"}), 404

    # Extraire les features du client
    X_top = data[data["SK_ID_CURR"] == client_id].drop(columns=["SK_ID_CURR"])

    # Faire la prédiction
    probability = model.predict_proba(X_top)[:, 1][0]
    prediction = int(probability >= 0.5)

    # Calculer les valeurs SHAP
    shap_values = explainer(X_top)

    # Récupérer les SHAP values du client
    client_shap = shap_values.values[0]

    # Construire un DataFrame pour trier les SHAP values
    shap_df = pd.DataFrame({
        'feature': X_top.columns,
        'shap_value': client_shap
    })

    # Trier par impact absolu
    shap_df_sorted = shap_df.reindex(shap_df['shap_value'].abs().sort_values(ascending=False).index)

    # Prendre les 10 variables les plus impactantes
    shap_df_top = shap_df_sorted.head(10)

    # Créer un dictionnaire {feature: shap_value}
    shap_dict = dict(zip(shap_df_top['feature'], shap_df_top['shap_value']))

    # Renvoyer la réponse
    return jsonify({
        "prediction": prediction,
        "probability": probability,
        "features": X_top.iloc[0].to_dict(),
        "global_means": data.drop(columns=["SK_ID_CURR"]).mean().to_dict(),
        "shap_values": shap_dict
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)