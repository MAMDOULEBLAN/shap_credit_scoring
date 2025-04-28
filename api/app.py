from flask import Flask, request, jsonify
import pandas as pd
import shap
import pickle
import os

app = Flask(__name__)

# Charger le modèle et les données avec chemins absolus compatibles Heroku
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'best_model.pickle'))
model = pickle.load(open(model_path, "rb"))

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_full.csv'))
data = pd.read_csv(data_path)

# Garder toutes les colonnes sauf SK_ID_CURR
feature_columns = [col for col in data.columns if col != "SK_ID_CURR"]

# Créer l'explainer SHAP
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

    # Extraire les données du client
    X_client = data[data["SK_ID_CURR"] == client_id][feature_columns]

    # Prédiction
    proba = model.predict_proba(X_client)[:, 1][0]
    prediction = int(proba >= 0.5)

    # Calculer les SHAP values pour ce client
    shap_values = explainer.shap_values(X_client)
    client_shap = shap_values[0]

    # Construire un DataFrame pour trier les variables impactantes
    shap_df = pd.DataFrame({
        'feature': feature_columns,
        'shap_value': client_shap
    })
    shap_df_sorted = shap_df.reindex(shap_df['shap_value'].abs().sort_values(ascending=False).index)

    # Prendre les 10 plus importantes
    shap_df_top10 = shap_df_sorted.head(10)

    # Créer un dictionnaire {feature: shap_value}
    shap_dict = dict(zip(shap_df_top10['feature'], shap_df_top10['shap_value']))

    # Retourner la réponse
    return jsonify({
        "prediction": prediction,
        "probability": proba,
        "features": X_client.iloc[0].to_dict(),
        "global_means": data[feature_columns].mean().to_dict(),
        "shap_values": shap_dict
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
