
import os
import pandas as pd
import pickle
import shap
from flask import Flask, request, jsonify

app = Flask(__name__)

# Chargement du modèle
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model", "best_model.pickle"))
print("Chargement du modèle depuis :", model_path)
model = pickle.load(open(model_path, "rb"))
print("Modèle chargé avec succès.")

# Chargement des données
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "sample_full.csv"))
print("Chargement des données depuis :", data_path)
df = pd.read_csv(data_path)
print(f"{len(df)} lignes chargées.")

# Initialisation de SHAP
explainer = shap.TreeExplainer(model)
features = ['EXT_SOURCE_2', 'DAYS_BIRTH', 'CREDIT_TO_ANNUITY_RATIO', 'EXT_SOURCE_3', 'PAYMENT_RATE']

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        id_client = input_data.get('id_client')
        print("Demande de prédiction pour ID client :", id_client)

        if id_client not in df['SK_ID_CURR'].values:
            return jsonify({"erreur": "Client non trouvé"}), 404

        X = df[df['SK_ID_CURR'] == id_client][features]
        proba = model.predict_proba(X)[:, 1][0]
        prediction = model.predict(X)[0]
        shap_values = explainer.shap_values(X, check_additivity=False)

        result = {
            "prediction": int(prediction),
            "probabilite": round(proba, 4),
            "shap_values": {feature: float(shap_values[1][0][i]) for i, feature in enumerate(features)}
        }
        return jsonify(result)
    except Exception as e:
        print("Erreur lors de la prédiction :", str(e))
        return jsonify({"erreur": str(e)}), 500

@app.route('/api/ids', methods=['GET'])
def get_ids():
    try:
        ids = sorted(df['SK_ID_CURR'].unique().tolist())
        return jsonify({"ids": ids})
    except Exception as e:
        print("Erreur lors de la récupération des IDs :", str(e))
        return jsonify({"erreur": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
