# Scoring Crédit avec Interprétation SHAP

[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-brightgreen)](https://TON-LIEN-STREAMLIT)
[![API](https://img.shields.io/badge/API-Heroku-blue)](https://shap-credit-api-mamdou-0a39fd6254f1.herokuapp.com/api/predict)

## 🔍 Description

Ce projet a pour objectif de prédire l’acceptation ou le refus d’une demande de crédit.  
Il comprend :

- ✅ Un **modèle de scoring** basé sur LightGBM
- 🌐 Une **API Flask** déployée sur Heroku pour les prédictions
- 📊 Un **dashboard Streamlit** pour visualiser les résultats et l’explication des décisions
- 📌 Une **interprétation des décisions avec SHAP** pour assurer la transparence

## 📦 Fonctionnalités

- Affichage du score client
- Visualisation des variables influentes (SHAP)
- Comparaison des données client à la moyenne de la population
- Requête dynamique à l’API pour obtenir les résultats

## 🚀 Liens utiles

- **Dashboard Streamlit** : [Accéder au dashboard](https://TON-LIEN-STREAMLIT)
- **API Flask (POST)** : [`/api/predict`](https://shap-credit-api-mamdou-0a39fd6254f1.herokuapp.com/api/predict)
- **API Flask (GET)** : [`/api/ids`](https://shap-credit-api-mamdou-0a39fd6254f1.herokuapp.com/api/ids)

## 🗂️ Organisation du projet

```
shap_credit_project/
│
├── api/
│   ├── app.py              # Code de l’API Flask
│   ├── data/               # Données CSV nécessaires
│   └── model/              # Fichier du modèle entraîné (.pkl)
│
├── dashboard/
│   └── streamlit_dashboard.py  # Dashboard interactif
│
├── requirements.txt        # Dépendances du projet
├── Procfile                # Commande de lancement pour Heroku
└── README.md               # Ce fichier
```
