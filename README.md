# Projet de scoring de crédit – OpenClassrooms

Ce projet a pour objectif de construire un modèle de scoring de crédit permettant à une entreprise comme "Prêt à dépenser" de prédire si un client est éligible à un prêt.

## 🚀 Fonctionnalités

- 🔍 Prédiction de probabilité de défaut
- 📊 Dashboard interactif (Streamlit)
- 🧠 Interprétabilité avec SHAP
- 🔗 API Flask connectée au dashboard
- 📁 Projet organisé en modules : `api/`, `dashboard/`, `data/`, `model/`

## 🛠️ Lancer le projet en local

### 1. Se placer dans le dossier racine
```bash
cd shap_credit_project_final
```

### 2. Créer et activer l'environnement (facultatif si conda utilisé)
```bash
conda create -n shap_test python=3.10
conda activate shap_test
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Lancer l'API
```bash
python api/app.py
```

### 5. Lancer le dashboard
```bash
streamlit run dashboard/streamlit_dashboard.py
```

## 🔑 Tester un identifiant client

Exemples valides :
```
100001, 100005, 100021
```

## 📦 Arborescence du projet

```
shap_credit_project_final/
│
├── api/
│   ├── app.py
│   ├── model/
│   │   └── best_model.pickle
│   └── data/
│       └── sample_full.csv
│
├── dashboard/
│   └── streamlit_dashboard.py
│
├── requirements.txt
└── README.md
```

## 👤 Auteur

Projet réalisé dans le cadre de la formation **Data Scientist – OpenClassrooms**
