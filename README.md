# Scoring Crédit avec Interprétation SHAP

## 🔍 Description
Ce projet a pour objectif de prédire l'acceptation ou le refus d'une demande de crédit à partir de données client. Il comprend :
- Une API Flask pour les prédictions
- Un dashboard Streamlit interactif
- Une interprétation des résultats à l'aide de SHAP

## 🗂️ Structure du projet
```
shap_credit_project/
├── api/                  # Code de l'API Flask (app.py, model, data...)
├── dashboard/            # Dashboard Streamlit (streamlit_dashboard.py)
├── requirements.txt      # Librairies requises
```

## ⚙️ Exécution en local

### 1. Cloner le projet
```bash
git clone https://github.com/MAMDOULEBLAN/shap_credit_scoring.git
cd shap_credit_scoring
```

### 2. Créer un environnement et installer les dépendances
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Lancer l'API Flask
```bash
cd api
python app.py
```
L'API tourne sur `http://127.0.0.1:5000`

### 4. Lancer le dashboard Streamlit
```bash
cd ..
streamlit run dashboard/streamlit_dashboard.py
```
Le dashboard est accessible sur `http://localhost:8501`

## ☁️ Déploiement Web

### API en ligne (Heroku)
```
https://shap-credit-api-mamdou.herokuapp.com/api/predict
```

### Dashboard en ligne (Streamlit Cloud)
```
https://mamdouleblan-shap-credit-scoring.streamlit.app
```

## 🔌 Fonctionnement de l'API

### ✉️ POST /api/predict
- Envoie un identifiant client
- Reçoit : la prédiction (accepté/refusé), la probabilité et les valeurs SHAP

### 🔢 GET /api/ids
- Retourne la liste triée des identifiants clients disponibles

## 📊 Fonctionnement du dashboard
- Sélection d'un ID client (dropdown)
- Récupération de la prédiction via l'API Heroku
- Affichage de la probabilité de défaut
- Affichage des 5 variables principales influençant la décision (SHAP)

## 👤 Auteur
**Mamadou LEBLAN**

Projet réalisé dans le cadre de la formation Data Scientist OpenClassrooms 🎓
