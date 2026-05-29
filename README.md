# POC SOC Triage — Priorisation ML + XAI

## Vue d'ensemble

Proof-of-concept de priorisation automatique des alertes de sécurité pour analystes N1, combinant Machine Learning supervisé (XGBoost) et non-supervisé (Isolation Forest) avec des explications XAI (SHAP + LIME).

**Dataset** : Microsoft GUIDE — Security Incident Prediction (Kaggle)
**Cible** : classer chaque incident en TP / BenignPositive / FP et calculer un score de priorisation 0-100

---

## Structure du projet

```
poc-soc-triage/
├── 01_extract_and_simulate.py   # Extraction dataset + génération données simulées
├── 02_feature_engineering.ipynb # Construction des features ML
├── 03_ml_models.ipynb           # Isolation Forest + XGBoost + évaluation
├── 04_xai.ipynb                 # SHAP global + local + LIME + counterfactuals
├── 05_demo_app.py               # Dashboard Streamlit analyste N1
├── config/                      # Fichiers de configuration (requis par 01_)
│   ├── eol_os.csv               # OS en fin de support (sources officielles)
│   ├── ti_config.csv            # Paramètres TI simulée
│   ├── cmdb_config.csv          # Paramètres CMDB (criticité, MFA, zones réseau…)
│   └── sandbox_config.csv       # Paramètres comportements sandbox
├── data/                        # Généré automatiquement
│   ├── incidents_dataset.csv             # 01_ — dataset principal
│   ├── TI_database.csv                   # 01_ — Threat Intel simulée (air-gap)
│   ├── CMDB.csv                          # 01_ — CMDB simulée (air-gap)
│   ├── Sandbox.csv                       # 01_ — résultats sandbox simulés (air-gap)
│   ├── crit_ratios.csv                   # 01_ — ratios de criticité
│   ├── features_ml.csv                   # 02_ — dataset ML
│   ├── features_ml.parquet               # 02_ — dataset ML (format ML, dtypes préservés)
│   ├── encoder_alert_title.{csv,parquet} # 02_ — table de taux par titre d'alerte
│   ├── encoder_category.{csv,parquet}    # 02_ — table de taux par catégorie
│   ├── encoder_detector.{csv,parquet}    # 02_ — table de taux par détecteur
│   ├── global_rates.pkl                  # 02_ — taux globaux de fallback
│   ├── predictions_sample.csv            # 03_ — 500 incidents pour la démo
│   ├── explanations_sample.csv           # 04_ — textes d'explication XAI
│   ├── shap_values_sample.pkl            # 04_ — valeurs SHAP
│   └── lime_values_sample.pkl            # 04_ — valeurs LIME
└── models/
    ├── isolation_forest.pkl     # 03_ — modèle non-supervisé + scaler
    ├── xgboost_booster.json     # 03_ — booster XGBoost natif (chargé par 04_ et 05_)
    └── inference_config.pkl     # 03_ — FEATURE_COLS + config d'inférence
```

---

## Installation

```bash
bash setup.sh
```

Le script vérifie Python 3.10+, crée un virtualenv `.venv`, installe les dépendances depuis `requirements.txt` et initialise la structure de dossiers.

---

## Étapes d'exécution

### 1. Récupérer le dataset

Deux options au choix :

**A) Téléchargement manuel** depuis [kaggle.com](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction) — récupérer `GUIDE_Train.csv` et le placer dans `data/`.

**B) Via la CLI Kaggle :**

```bash
# Configurer l'API Kaggle : kaggle.com > Account > Create New Token > kaggle.json dans ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d Microsoft/microsoft-security-incident-prediction
unzip microsoft-security-incident-prediction.zip -d data/
```

### 2. Extraire un échantillon représentatif + générer les données simulées

```bash
python 01_extract_and_simulate.py --input data/GUIDE_Train.csv --n_incidents 30000 --no-expand
# Durée : ~10-15 min selon la machine
# Output : ./data/*.csv
```

> ⚠️ Les fichiers `config/` (eol_os.csv, ti_config.csv, cmdb_config.csv, sandbox_config.csv) sont requis avant de lancer ce script. Sans eux, le script s'arrête avec une `FileNotFoundError`.

**Option Parquet** (recommandé si plusieurs runs) :

```bash
python 01_extract_and_simulate.py --input data/GUIDE_Train.csv --n_incidents 30000 --no-expand --to-parquet
# Les runs suivants passent directement : --input data/guide_train.parquet
```

### 3. Feature engineering

Ouvrir et exécuter `02_feature_engineering.ipynb` cellule par cellule.

**Familles de features construites :**
| Famille | Features clés |
|---------|--------------|
| Alerte brute | alert_title_tp_rate, alert_title_fp_rate, category_fp_rate, nb_alerts, nb_detectors |
| Temporelle | hour_sin/cos, is_weekend, is_after_midnight, alert_rate_per_hour |
| Threat Intelligence | ip_ti_score, hash_ti_score, any_ioc_in_blocklist, ti_actor_* |
| CMDB / Asset | asset_criticality_score, asset_risk_score, asset_x_ti |
| CMDB / User | user_criticality_score, MFA_enabled, nb_failed_logins_7d, login_country_mismatch |
| Sandbox | sandbox_malware_score, sandbox_c2_beaconing, sandbox_evasion |
| Historique SOC | detector_fp_rate, detector_tp_rate, alert_title_fp_rate ⬅ feature SHAP n°1 |
| MITRE | mitre_initial_access, mitre_lateral_movement, mitre_impact… |

### 4. Entraînement ML

Exécuter `03_ml_models.ipynb` :
- **Isolation Forest** (non-supervisé) : score anomalie sans labels
- **XGBoost** (supervisé) : classification TP/BP/FP avec `sample_weight` + évaluation F2-score
- **Simulation feedback loop** : amélioration du modèle avec les validations analyste

### 5. XAI

Exécuter `04_xai.ipynb` :
- SHAP summary plot global
- Waterfall SHAP par incident
- LIME — approximation linéaire locale par incident
- Analyse de convergence SHAP/LIME
- Counterfactuals (sensibilité aux features)
- Génération des textes d'explication pour Streamlit

### 6. Démo Streamlit

```bash
streamlit run 05_demo_app.py
```

---

## Architecture ML

### Isolation Forest (non-supervisé)
- Entraîné SANS labels → applicable dès le jour 1 en production
- Score d'anomalie 0-100 : 100 = incident le plus atypique statistiquement
- Évaluation : AUC-ROC (TP vs reste) et précision dans le top 20% des scores

### XGBoost (supervisé)
- Labels GUIDE utilisés : TP=2, BenignPositive=1, FP=0
- Rééquilibrage des classes via `sample_weight` (`compute_sample_weight('balanced')`),
  pas de SMOTE : en multiclasse avec BP majoritaire, SMOTE génère trop de BP
  synthétiques qui noient le signal TP et font chuter le F2-score
- Split stratifié par OrgId (StratifiedGroupKFold) — évite le data leakage organisationnel
- Métriques : F2-score (macro), AUC-ROC multi-classe, Precision-Recall

### Score de priorisation (0-100)
Basé sur P(TP) du modèle XGBoost, avec seuils :
| Score | Sévérité | Décision suggérée |
|-------|----------|-------------------|
| ≥ 75  | CRITIQUE | Action urgente (isolation) |
| 45–74 | HAUTE    | Escalade N2/N3 |
| 20–44 | MOYENNE  | Investigation N1 |
| < 20  | FAIBLE   | Clôture FP probable |

---

## XAI — SHAP et LIME

Deux méthodes d'explication complémentaires sont utilisées pour chaque incident :

**SHAP (TreeExplainer)** calcule la contribution exacte de chaque feature au score, en s'appuyant sur la structure interne du modèle XGBoost. Il donne une explication fidèle mais spécifique à ce type de modèle.

**LIME** perturbe les valeurs de l'incident et entraîne un modèle linéaire local pour approximer XGBoost dans son voisinage. Il est model-agnostic et apporte une perspective indépendante.

La convergence SHAP/LIME (part de features communes dans le top 8) est calculée par incident et par type de décision ; un seuil de 60% est utilisé comme repère de cohérence. Une convergence faible sur certains incidents est normale — le modèle hésite dans ces zones non-linéaires, ce qui est précisément l'information utile pour l'analyste.

---

## Données simulées (compatible air-gap / banque)

Les fichiers TI_database.csv, CMDB.csv et Sandbox.csv sont générés localement à partir des valeurs réelles du dataset GUIDE (IPs, hashes, noms de machines). En production bancaire, ces fichiers seraient remplacés par des connecteurs internes :

| Fichier simulé | Source réelle en banque |
|----------------|------------------------|
| TI_database.csv | MISP interne, Threat Intel feed offline |
| CMDB.csv | ServiceNow, IBM MAXIMO, ou CMDB propriétaire |
| Sandbox.csv | Cuckoo/Cape en réseau isolé, CrowdStrike Falcon |

Aucune donnée ne sort du réseau interne.

---

## Feedback loop (théorique)

Chaque décision de l'analyste (confirmation ou correction) constitue un nouveau label de ground truth. En production :
1. Les décisions sont stockées en base de données interne
2. Un batch hebdomadaire réentraîne XGBoost avec les nouveaux labels
3. Le modèle s'adapte aux spécificités organisationnelles (assets, détecteurs, utilisateurs)

La simulation dans `03_ml_models.ipynb` montre l'amélioration attendue du F2-score sur 3 cycles de feedback.

---

## Points clés pour la présentation

1. **Le non-supervisé démarre sans labels** : Isolation Forest opérationnel dès J1 en banque
2. **Le supervisé apprend des décisions passées** : amélioration continue via feedback loop
3. **L'analyste N1 ne cherche plus** : tout le contexte (TI, CMDB, sandbox, timeline) est pré-agrégé
4. **XAI = confiance** : SHAP explique POURQUOI le score est élevé ; LIME confirme indépendamment
5. **Air-gap compatible** : toutes les données simulées sont générées et consultées localement
