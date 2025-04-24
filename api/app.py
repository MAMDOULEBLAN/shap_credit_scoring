import os
import pandas as pd
import shap
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Chargement du modèle et des données
model_path = os.path.join("model", "best_model.pickle")
data_path = os.path.join("data", "sample_full.csv")

model = pickle.load(open(model_path, "rb"))
df = pd.read_csv(data_path)

@app.route('/api/ids', methods=['GET'])
def get_ids():
    ids = sorted(df['SK_ID_CURR'].unique().tolist())
    return jsonify({"ids": ids})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        content = request.json
        client_id = int(content['id'])
        client_data = df[df['SK_ID_CURR'] == client_id].drop(columns=["SK_ID_CURR"])

        proba = model.predict_proba(client_data)[0][1]
        prediction = model.predict(client_data)[0]

        explainer = shap.Explainer(model)
        shap_values = explainer(client_data)
        top_features = sorted(
            zip(client_data.columns, shap_values.values[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        top_features_list = [{"feature": f, "shap_value": float(v)} for f, v in top_features]

        return jsonify({
            "prediction": int(prediction),
            "proba": round(proba, 4),
            "shap_values": top_features_list
        })
    except Exception as e:
        return jsonify({"erreur": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)