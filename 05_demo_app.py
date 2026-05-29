"""
POC SOC Triage — 05 Démo Streamlit
====================================

Tableau de bord analyste N1 — matérialisation opérationnelle du POC :
  - File d'incidents priorisée par score 0-100 (zones ACTION_URGENTE / 
    INVESTIGATION / CLÔTURE_FP).
  - Contexte CMDB et TI pré-agrégé par incident.
  - Panneau d'explication XAI (SHAP local + LIME) déroulable.
  - Interface de validation analyste (TP/BP/FP) alimentant la boucle de
    feedback simulée.

Inputs (data/) :
    predictions_sample.csv      — scores des 500 incidents (depuis 03_)
    explanations_sample.csv     — textes XAI (depuis 04_)
    CMDB.csv                    — enrichissement contextuel (depuis 01_)
    features_ml.parquet         — features ML pour la narrative dynamique (depuis 02_)
    shap_values_sample.pkl      — valeurs SHAP locales des 500 incidents (depuis 04_)
    lime_values_sample.pkl      — coefficients LIME des 500 incidents (depuis 04_)
    synthetic_predictions.csv   — incidents synthétiques optionnels (démo, facultatif)
    synthetic_explanations.csv  — textes XAI synthétiques optionnels (démo, facultatif)
    Figures du 04_ (shap_*.png, counterfactuals.png, isolation_forest_scores.png)

Lancement :
    streamlit run 05_demo_app.py

Dépendances :
    pip install streamlit pandas numpy matplotlib shap lime joblib
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from pathlib import Path

# ─── Config page ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SOC Triage Assistant — POC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

# ─── Styles ───────────────────────────────────────────────────────────────────

# CSS global : multiselect tags plus larges + file scrollable
st.markdown("""
<style>
/* Multiselect : tags sur plusieurs lignes, texte complet */
[data-baseweb="tag"] { white-space: normal !important; height: auto !important; }
[data-baseweb="tag"] span { white-space: normal !important; }

/* File d'incidents scrollable */
[data-testid="stVerticalBlock"] .queue-scroll {
    max-height: 600px; overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

# ─── Chargement des données ────────────────────────────────────────────────────

@st.cache_data
def load_data():
    pred = pd.read_csv(DATA_DIR / "predictions_sample.csv")
    expl = pd.read_csv(DATA_DIR / "explanations_sample.csv")
    pred = pred.merge(expl, on="IncidentId", how="left")
    pred["first_seen"] = pd.to_datetime(pred["first_seen"], errors="coerce")

    # Enrichissement CMDB
    cmdb_devices = None
    try:
        cmdb = pd.read_csv(DATA_DIR / "CMDB.csv")
        cmdb_cols = ["device_id", "device_name", "asset_criticality", "asset_type",
                     "business_unit", "os_family", "is_internet_exposed", "patch_level"]
        cmdb_devices = (cmdb[cmdb["device_id"].notna()][cmdb_cols]
                        .drop_duplicates(subset=["device_id"]))
        pred = pred.merge(cmdb_devices, left_on="sample_device_id",
                          right_on="device_id", how="left")
    except Exception:
        pass  # CMDB optionnel

    # Merge incidents synthétiques si disponibles
    try:
        syn_pred = pd.read_csv(DATA_DIR / "synthetic_predictions.csv")
        syn_expl = pd.read_csv(DATA_DIR / "synthetic_explanations.csv")
        syn = syn_pred.merge(syn_expl, on="IncidentId", how="left")
        syn["first_seen"] = pd.to_datetime(syn["first_seen"], errors="coerce")
        if cmdb_devices is not None:
            syn = syn.merge(cmdb_devices, left_on="sample_device_id",
                            right_on="device_id", how="left")
        pred = pd.concat([pred, syn], ignore_index=True)
    except Exception:
        pass  # synthétiques optionnels

    # Flag signaux faibles + colonnes pour la narrative dynamique
    # On lit depuis features_ml.parquet qui contient toutes les features ML
    FEAT_COLS_NEEDED = [
        "IncidentId", "ti_score_combined", "ti_score_max",
        "nb_unique_ips", "nb_unique_devices", "nb_unique_accounts", "spread_score",
        "detector_tp_rate",
        "sandbox_c2_beaconing", "sandbox_process_injection", "sandbox_malware_score",
        "sandbox_evasion", "sandbox_lateral_movement", "sandbox_privilege_escalation",
        "mitre_lateral_movement", "mitre_credential_access",
        "mitre_command_and_control", "mitre_exfiltration", "mitre_persistence",
        "is_privileged_account", "user_criticality_score", "account_inactive",
        "MFA_enabled", "nb_failed_logins_7d", "login_country_mismatch",
        "asset_x_ti", "asset_x_alerts", "privileged_x_failed_logins",
        "alert_rate_per_hour", "incident_duration_min",
        "has_impacted_entity", "nb_impacted_entities",
    ]
    try:
        parquet_df = pd.read_parquet(DATA_DIR / "features_ml.parquet")
        available = [c for c in FEAT_COLS_NEEDED if c in parquet_df.columns]
        feat = parquet_df[available]
        pred = pred.merge(feat, on="IncidentId", how="left", suffixes=("", "_feat"))
        ti_col = "ti_score_combined"
    except Exception:
        ti_col = None

    if ti_col and ti_col in pred.columns:
        pred["signaux_faibles"] = (
            (pred["priority_score"] >= 75) &
            (pred[ti_col].fillna(0) < 15) &
            (pred["nb_alerts"] <= 10)
        )
    else:
        # Fallback : lire depuis le texte d'explication
        import re
        def _ti_val(text):
            if pd.isna(text): return 0.0
            m = re.search(r'ti_score_combined = ([\d.]+)', text)
            return float(m.group(1)) if m else 0.0
        pred["_ti_in_expl"] = pred["explanation_text"].apply(_ti_val)
        pred["signaux_faibles"] = (
            (pred["priority_score"] >= 75) &
            (pred["_ti_in_expl"] < 20) &
            (pred["nb_alerts"] <= 10)
        )
        pred = pred.drop(columns=["_ti_in_expl"])

    return pred

@st.cache_resource
def load_shap():
    try:
        data = joblib.load(DATA_DIR / "shap_values_sample.pkl")
        sv = data.get("shap_values")
        if sv is not None and not isinstance(sv, list):
            data["shap_values"] = [sv[:, :, i] for i in range(sv.shape[2])]
        return data
    except Exception:
        return None

@st.cache_resource
def load_lime():
    try:
        return joblib.load(DATA_DIR / "lime_values_sample.pkl")
    except Exception:
        return None

df = load_data()
shap_data = load_shap()
lime_data  = load_lime()

# ─── Header ──────────────────────────────────────────────────────────────────

col_logo, col_title = st.columns([1, 7])
with col_logo:
    st.markdown("## 🛡️")
with col_title:
    st.markdown("### SOC Triage Assistant — POC")
    st.caption("Priorisation ML + Explications XAI pour analyste N1")

# ─── Métriques ROI ───────────────────────────────────────────────────────────
# Calculées dynamiquement sur les données chargées

total = len(df[df["is_synthetic"].fillna(False) == False]) if "is_synthetic" in df.columns else len(df)
action_urgente = df[df["decision_suggested"] == "ACTION_URGENTE"]
n_au = len(action_urgente)
tp_in_au = (action_urgente["grade_reel_label"] == "TruePositive").sum()
total_tp = (df["grade_reel_label"] == "TruePositive").sum()
pct_volume = n_au / len(df) * 100 if len(df) > 0 else 0
pct_tp_captured = tp_in_au / total_tp * 100 if total_tp > 0 else 0
precision_au = tp_in_au / n_au * 100 if n_au > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Incidents", f"{len(df):,}")
c2.metric("🔴 Action urgente", f"{n_au}",
          help="Incidents scorés ≥ 75/100 par le ML")
c3.metric("Précision ACTION_URGENTE", f"{precision_au:.0f}%",
          help="Part de vrais positifs parmi les incidents ACTION_URGENTE")
c4.metric("TP capturés", f"{pct_tp_captured:.0f}%",
          help=f"Part des vrais incidents (TP) détectés dans les {pct_volume:.0f}% prioritaires")
c5.metric("Volume priorisé", f"{pct_volume:.0f}%",
          help=f"L'analyste traite {pct_volume:.0f}% du volume pour capturer {pct_tp_captured:.0f}% des menaces")

st.markdown(
    f"<div style='background:#f0fdf4; border-left:4px solid #1d9e75; "
    f"border-radius:6px; padding:8px 14px; font-size:0.82rem; color:#065f46; margin-top:4px;'>"
    f"💡 <strong>Apport ML</strong> — Sans priorisation, l'analyste traite 500 incidents dans l'ordre d'arrivée. "
    f"Avec le ML, il se concentre sur <strong>{n_au} incidents ({pct_volume:.0f}%)</strong> "
    f"qui contiennent <strong>{pct_tp_captured:.0f}% des menaces réelles</strong> "
    f"avec une précision de <strong>{precision_au:.0f}%</strong>."
    f"</div>",
    unsafe_allow_html=True
)

st.divider()

# ─── Filtres ─────────────────────────────────────────────────────────────────

col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
with col_f1:
    st.markdown("**Niveau de priorité**")
    NIVEAU_MAP = {
        "CRITIQUE": ("ACTION_URGENTE",   "🔴 CRITIQUE",  "#e24b4a", "#fde8e8"),
        "HAUTE":    ("ESCALADE_N2",      "🟠 HAUTE",     "#ba7517", "#fef3c7"),
        "MOYENNE":  ("INVESTIGATION_N1", "🔵 MOYENNE",   "#3b8bd4", "#dbeafe"),
        "FAIBLE":   ("CLOTURE_FP",       "🟢 FAIBLE",    "#1d9e75", "#d1fae5"),
    }
    selected_sevs, selected_decs = [], []
    niveaux = list(NIVEAU_MAP.items())
    ca, cb = st.columns(2)
    for col, (sev_key, (dec, label, color, bg)) in zip([ca, cb], niveaux[:2]):
        with col:
            if st.checkbox(label, value=True, key=f"cb_{sev_key}"):
                selected_sevs.append(sev_key)
                selected_decs.append(dec)
    ca2, cb2 = st.columns(2)
    for col, (sev_key, (dec, label, color, bg)) in zip([ca2, cb2], niveaux[2:]):
        with col:
            if st.checkbox(label, value=True, key=f"cb_{sev_key}"):
                selected_sevs.append(sev_key)
                selected_decs.append(dec)

with col_f2:
    score_range = st.slider("Score de priorisation", 0, 100, (0, 100))

with col_f3:
    search_id = st.text_input("🔍 Rechercher un incident (ID)", placeholder="ex: 3603")

df_filtered = df[
    df["severity"].isin(selected_sevs) &
    df["priority_score"].between(score_range[0], score_range[1])
].sort_values("priority_score", ascending=False).reset_index(drop=True)

if search_id.strip():
    try:
        sid = int(search_id.strip())
        df_filtered = df_filtered[df_filtered["IncidentId"] == sid].reset_index(drop=True)
    except ValueError:
        pass

# ─── Layout principal : file gauche + détail droite ───────────────────────────

col_queue, col_detail = st.columns([1, 2], gap="medium")

# ─── File d'incidents ───────────────────────────────────────────────────────────

# ─── Dictionnaires métier ─────────────────────────────────────────────────────

CATEGORY_FR = {
    "Exfiltration":        "Exfiltration de données",
    "InitialAccess":       "Accès initial",
    "SuspiciousActivity":  "Activité suspecte",
    "CommandAndControl":   "Commande & Contrôle (C2)",
    "CredentialAccess":    "Vol de credentials",
    "Impact":              "Impact / Destruction",
    "Execution":           "Exécution de code",
    "Discovery":           "Reconnaissance interne",
    "Persistence":         "Persistance",
    "LateralMovement":     "Mouvement latéral",
    "Malware":             "Malware",
    "DefenseEvasion":      "Contournement défense",
    "Exploit":             "Exploitation de vulnérabilité",
    "Ransomware":          "Ransomware",
}

CATEGORY_DESC = {
    "Exfiltration":        "Des données semblent avoir été transférées hors du réseau.",
    "InitialAccess":       "Une tentative d'entrée dans le réseau a été détectée.",
    "SuspiciousActivity":  "Un comportement inhabituel a été observé sur un poste ou compte.",
    "CommandAndControl":   "Le poste communique avec un serveur de contrôle externe suspect.",
    "CredentialAccess":    "Des identifiants utilisateur ont pu être compromis ou volés.",
    "Impact":              "Une action destructrice (chiffrement, suppression) a été détectée.",
    "Execution":           "Un code ou script suspect a été exécuté sur le poste.",
    "Discovery":           "Le poste explore activement le réseau interne.",
    "Persistence":         "Un mécanisme de persistance a été installé sur le poste.",
    "LateralMovement":     "Une tentative de propagation vers d'autres machines est en cours.",
    "Malware":             "Un fichier ou processus malveillant connu a été identifié.",
    "DefenseEvasion":      "Des techniques pour contourner les protections ont été détectées.",
    "Exploit":             "Une vulnérabilité logicielle a été exploitée.",
    "Ransomware":          "Un comportement de chiffrement de fichiers est en cours.",
}

FEATURE_LABELS = {
    # ── Historique SOC ────────────────────────────────────────────────────
    "detector_tp_rate":         "Taux de vrais positifs (TP) historique de ce détecteur",
    "detector_fp_rate":         "Taux de faux positifs (FP) de ce détecteur",
    "detector_bp_rate":         "Taux de bénins positifs (BP) de ce détecteur",
    "detector_nb_incidents":    "Nombre d'incidents historiques sur ce détecteur",
    # ── Volume et structure alerte ────────────────────────────────────────
    "nb_alerts":                "Nombre d'alertes agrégées dans cet incident",
    "nb_evidences":             "Nombre total d'évidences brutes dans l'incident",
    "nb_detectors":             "Nombre de détecteurs différents déclenchés",
    "nb_categories":            "Nombre de catégories MITRE distinctes dans l'incident",
    "nb_entity_types":          "Nombre de types d'entités impliquées (IP, device, compte...)",
    "alert_rate_per_hour":      "Densité d'alertes par heure (burst attack)",
    "alert_title_tp_rate":      "Taux de vrais positifs (TP) historique pour ce titre d'alerte",
    "alert_title_fp_rate":      "Taux de faux positifs (FP) pour ce titre d'alerte",
    "alert_title_bp_rate":      "Taux de bénins positifs (BP) pour ce titre d'alerte",
    "category_tp_rate":         "Taux de vrais positifs (TP) historique pour cette catégorie MITRE",
    "category_fp_rate":         "Taux de faux positifs (FP) pour cette catégorie MITRE",
    "category_bp_rate":         "Taux de bénins positifs (BP) pour cette catégorie MITRE",
    # ── Threat Intelligence (TI) ──────────────────────────────────────────
    "ti_score_max":             "Score TI maximum — meilleur signal disponible (IP ou hash)",
    "ti_score_combined":        "Score TI combiné — somme réputation IP + hash",
    "ti_ip_score":              "Score TI de l'adresse IP source",
    "ip_ti_score":              "Score TI de l'adresse IP source",
    "ti_hash_score":            "Score TI du hash fichier",
    "hash_ti_score":            "Score TI du hash fichier",
    "ip_in_blocklist":          "IP présente dans une liste de blocage active (blocklist)",
    "hash_in_blocklist":        "Hash présent dans une liste de blocage active (blocklist)",
    "any_ioc_in_blocklist":     "Au moins un IOC (IP ou hash) en blocklist",
    "hash_malicious":           "Hash de fichier connu comme malveillant en TI",
    "hash_nb_sources":          "Nombre de sources TI signalant ce hash",
    "ip_nb_sources":            "Nombre de sources TI signalant cette IP",
    "blocklist_hit":            "IOC présent dans une liste de blocage (blocklist)",
    # ── Sandbox ───────────────────────────────────────────────────────────
    "sandbox_malware_score":        "Score malveillant sandbox (comportement fichier analysé)",
    "sandbox_score":                "Score malveillant sandbox (comportement fichier analysé)",
    "sandbox_network_conns":        "Connexions réseau suspectes détectées en sandbox",
    "sandbox_file_drops":           "Fichiers déposés par le processus analysé en sandbox",
    "sandbox_process_injection":    "Injection de code dans un processus légitime (sandbox)",
    "sandbox_c2_beaconing":         "Communication C2 (Command & Control) détectée en sandbox",
    "sandbox_evasion":              "Évasion sandbox détectée — malware qui contourne l'analyse",
    "sandbox_lateral_movement":     "Mouvement latéral détecté en sandbox (T1021)",
    "sandbox_privilege_escalation": "Élévation de privilège détectée en sandbox (T1068)",
    "sandbox_dns_count":            "Requêtes DNS anormalement nombreuses (signal C2-over-DNS)",
    "has_sandbox_analysis":         "Fichier analysé en sandbox (0 = hash inconnu)",
    # ── CMDB / Asset ──────────────────────────────────────────────────────
    "asset_criticality":        "Criticité de l'asset ciblé (DC, serveur prod, poste...)",
    "asset_criticality_score":  "Score de criticité asset (1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL)",
    "asset_sensitivity_score":  "Score de sensibilité de l'asset (données PII, secrets...)",
    "asset_internet_exposed":   "Asset directement exposé sur Internet",
    "asset_patch_risk":         "Risque lié au niveau de patching (1=OK, 2=retard, 3=critique)",
    "asset_risk_score":         "Score de risque global de l'asset",
    # ── Temporel ──────────────────────────────────────────────────────────
    "hour_sin":                 "Heure de l'incident — composante sin (encodage cyclique)",
    "hour_cos":                 "Heure de l'incident — composante cos (encodage cyclique)",
    "day_sin":                  "Jour de la semaine — composante sin (encodage cyclique)",
    "day_cos":                  "Jour de la semaine — composante cos (encodage cyclique)",
    "hour":                     "Heure de l'incident (horaires atypiques = plus suspect)",
    "day":                      "Jour de la semaine",
    "is_weekend":               "Incident survenu le week-end",
    "is_business_hours":        "Incident durant les heures ouvrées (8h-18h)",
    "incident_duration_min":    "Durée de l'incident en minutes",
    "weekend":                  "Incident survenu en dehors des heures ouvrées",
    "duration":                 "Durée de l'activité suspecte",
    # ── Ratios et propagation ─────────────────────────────────────────────
    "accounts_per_device":      "Ratio comptes / machines (signal de mouvement latéral)",
    "ips_per_hour":             "Densité d'IPs distinctes par heure (scan ou propagation rapide)",
    "alerts_per_detector":      "Alertes moyennes par détecteur (faible = convergence multi-détecteurs)",
    "spread_score":             "Score de propagation pondéré (2×devices + 1.5×comptes + 1×IPs)",
    # ── User risk ─────────────────────────────────────────────────────────
    "is_privileged_account":       "Compte administrateur ou à privilèges ciblé",
    "user_criticality_score":      "Criticité du compte (exec, admin, analyste...)",
    "account_inactive":            "Compte inactif ou suspendu réactivé (pattern APT)",
    "MFA_enabled":                 "Authentification multi-facteur activée sur ce compte",
    "nb_failed_logins_7d":         "Tentatives de login échouées sur 7 jours (credential stuffing)",
    "login_country_mismatch":      "Connexion depuis un pays inhabituel (impossible travel T1078)",
    # ── Interactions CMDB × incident ─────────────────────────────────────
    "asset_x_ti":                  "Asset critique × score TI — double signal convergent",
    "asset_x_alerts":              "Asset critique × volume d'alertes — attaque sur cible de valeur",
    "asset_x_sandbox":             "Asset critique × malware score — impact potentiel élevé",
    "privileged_x_failed_logins":  "Compte privilégié × tentatives login — credential stuffing admin",
    # ── Propagation enrichie ──────────────────────────────────────────────
    "has_impacted_entity":         "Au moins une entité directement impactée dans l'incident",
    "nb_impacted_entities":        "Nombre d'entités directement impactées (EvidenceRole=Impacted)",
    # ── TI enrichie ───────────────────────────────────────────────────────
    "ti_actor_APT":                "IOC attribué à un groupe APT (espionnage étatique)",
    "ti_actor_cybercrime":         "IOC attribué à un groupe cybercriminel organisé",
    "ti_actor_hacktivist":         "IOC attribué à un groupe hacktivist",
    "ip_ti_confidence":            "Niveau de confiance du score TI pour cette IP",
    "hash_ti_confidence":          "Niveau de confiance du score TI pour ce hash",
    # ── Contexte incident ─────────────────────────────────────────────────
    "has_ip":                   "Incident implique une adresse IP identifiée",
    "has_file":                 "Incident implique un fichier (hash SHA256)",
    "has_account":              "Incident implique un compte utilisateur",
    "has_device":               "Incident implique un device/machine identifié",
    "has_email":                "Incident implique un email (NetworkMessageId)",
    "has_threat_family":        "Famille de malware connue identifiée (ex: Emotet, Cobalt Strike)",
    "suspicion_level_encoded":  "Niveau de suspicion (Suspicious=1, Incriminated=2)",
    "is_windows":               "Incident sur un système Windows",
    # ── MITRE ATT&CK ──────────────────────────────────────────────────────
    "mitre_initial_access":     "MITRE ATT&CK — Accès initial détecté (T1078, T1190...)",
    "mitre_execution":          "MITRE ATT&CK — Exécution de code (T1059, T1053...)",
    "mitre_persistence":        "MITRE ATT&CK — Mécanisme de persistance (T1547, T1543...)",
    "mitre_credential_access":  "MITRE ATT&CK — Vol de credentials (T1003, T1110...)",
    "mitre_discovery":          "MITRE ATT&CK — Reconnaissance interne (T1082, T1087...)",
    "mitre_lateral_movement":   "MITRE ATT&CK — Mouvement latéral (T1021, T1550...)",
    "mitre_command_and_control":"MITRE ATT&CK — Communication C2 (Command & Control) (T1071, T1095...)",
    "mitre_exfiltration":       "MITRE ATT&CK — Exfiltration de données (T1041, T1048...)",
    "mitre_impact":             "MITRE ATT&CK — Impact / destruction (T1485, T1486...)",
}

def feature_label(feat_name):
    """Retourne un label métier lisible pour une feature SHAP.
    Teste les clés les plus longues en premier pour éviter qu'une clé générique
    (ex: 'hour') ne capte 'hour_sin' avant la clé spécifique.
    """
    sorted_keys = sorted(FEATURE_LABELS.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in feat_name:
            return FEATURE_LABELS[key]
    return feat_name  # fallback : nom brut

def humanize_explanation(raw_text):
    """Remplace les noms de features techniques par des labels métier lisibles.
    Traite les clés les plus longues en premier pour éviter les substitutions partielles
    (ex: 'hour' ne doit pas remplacer 'hour_sin' avant que 'hour_sin' soit traité).
    """
    if not raw_text or (isinstance(raw_text, float)):
        return None
    result = raw_text
    for key in sorted(FEATURE_LABELS.keys(), key=len, reverse=True):
        result = result.replace(key, FEATURE_LABELS[key])
    return result

SEVERITY_COLORS = {
    "CRITIQUE": "#fde8e8",
    "HAUTE": "#fef3c7",
    "MOYENNE": "#dbeafe",
    "FAIBLE": "#d1fae5",
}
SEVERITY_BORDER = {
    "CRITIQUE": "#e24b4a",
    "HAUTE": "#ba7517",
    "MOYENNE": "#3b8bd4",
    "FAIBLE": "#1d9e75",
}

with col_queue:
    st.markdown(f"**File priorisée** — {len(df_filtered)} incidents")

    if "selected_idx" not in st.session_state:
        st.session_state.selected_idx = 0

    # Scroll CSS natif Streamlit
    with st.container(height=600):
        for i, row in df_filtered.iterrows():
            sev = row.get("severity", "MOYENNE")
            score = row.get("priority_score", 0)
            cat_raw = str(row.get("top_category", ""))
            cat_fr = CATEGORY_FR.get(cat_raw, cat_raw)
            decision = row.get("decision_suggested", "")
            device_name = row.get("device_name", "")
            device_display = f"🖥️ {device_name}" if pd.notna(device_name) and str(device_name).strip() not in ("", "nan") else "—"
            inc_id = row.get("IncidentId", "")

            decision_emoji = {
                "ACTION_URGENTE": "🔴",
                "ESCALADE_N2": "🟠",
                "INVESTIGATION_N1": "🔵",
                "CLOTURE_FP": "🟢",
            }.get(decision, "⚪")

            bg = SEVERITY_COLORS.get(sev, "#f8f9fa")
            border = SEVERITY_BORDER.get(sev, "#888")
            is_selected = (st.session_state.selected_idx == i)
            border_width = "4px" if is_selected else "1px"
            sf_badge = ""
            if row.get("signaux_faibles", False):
                sf_badge = "<span style='background:#7c3aed; color:white; border-radius:3px; padding:1px 5px; font-size:0.68rem; font-weight:700; margin-left:4px;'>⚡ signaux faibles</span>"
            syn_badge = ""
            if row.get("is_synthetic", False):
                syn_badge = "<span style='background:#0369a1; color:white; border-radius:3px; padding:1px 5px; font-size:0.68rem; font-weight:600; margin-left:4px;'>🔬 synthétique</span>"

            col_card, col_btn = st.columns([5, 1])
            with col_card:
                st.markdown(f"""
                <div style='background:{bg}; border-left:{border_width} solid {border};
                            border-radius:6px; padding:6px 10px; margin-bottom:4px;'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='font-weight:600; font-size:0.82rem;'>{cat_fr}{sf_badge}{syn_badge}</span>
                        <span style='font-weight:700; color:{border};'>{score:.0f}</span>
                    </div>
                    <div style='font-size:0.72rem; color:#555;'>
                        {decision_emoji} #{inc_id} &nbsp;|&nbsp; {device_display}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_btn:
                st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
                if st.button("▶" if not is_selected else "✓",
                             key=f"sel_{i}_{inc_id}",
                             use_container_width=True,
                             type="primary" if is_selected else "secondary"):
                    st.session_state.selected_idx = i
                    st.rerun()

# ─── Détail de l'incident ─────────────────────────────────────────────────────

with col_detail:
    if len(df_filtered) == 0:
        st.info("Aucun incident dans les filtres sélectionnés.")
    else:
        idx = min(st.session_state.selected_idx, len(df_filtered) - 1)
        incident = df_filtered.iloc[idx]

        # En-tête incident
        sev = incident.get("severity", "MOYENNE")
        score = incident.get("priority_score", 0)
        decision = incident.get("decision_suggested", "INVESTIGATION_N1")
        border_color = SEVERITY_BORDER.get(sev, "#888")

        cat_raw = str(incident.get("top_category", ""))
        cat_fr = CATEGORY_FR.get(cat_raw, cat_raw)
        cat_desc = CATEGORY_DESC.get(cat_raw, "")
        first_seen_str = str(incident.get("first_seen", ""))[:16] or "—"
        is_sf = incident.get("signaux_faibles", False)
        is_syn = bool(incident.get("is_synthetic", False))

        syn_banner = ""
        if is_syn:
            scenario_ref = incident.get("scenario_ref", "")
            syn_banner = f"""
            <div style='background:#e0f2fe; border-left:4px solid #0369a1;
                        border-radius:6px; padding:8px 12px; margin-bottom:10px;
                        font-size:0.82rem; color:#0c4a6e;'>
                🔬 <strong>Incident synthétique</strong> — généré pour illustrer un cas réel documenté.<br>
                <span style='font-size:0.78rem; color:#075985;'>{scenario_ref}</span>
            </div>
            """

        sf_banner = ""
        if is_sf:
            sf_banner = """
            <div style='background:#f5f3ff; border-left:4px solid #7c3aed;
                        border-radius:6px; padding:8px 12px; margin-bottom:10px;
                        font-size:0.85rem; color:#4c1d95;'>
                ⚡ <strong>Signaux faibles détectés</strong> — Cet incident a été remonté par le ML
                sans signal Threat Intelligence ni volume d'alertes élevé.
                Un analyste humain aurait pu le manquer.
            </div>
            """

        st.markdown(f"""
        {syn_banner}{sf_banner}
        <div style='background:{SEVERITY_COLORS.get(sev,"#f8f9fa")};
                    border-left:5px solid {border_color};
                    border-radius:8px; padding:12px 16px; margin-bottom:12px;'>
            <div style='font-size:1.0rem; font-weight:700;'>
                {cat_fr}
            </div>
            <div style='font-size:0.85rem; color:#444; margin-top:4px;'>{cat_desc}</div>
            <div style='font-size:0.8rem; color:#555; margin-top:6px;'>
                Incident #{incident.get("IncidentId","")} &nbsp;|&nbsp; {first_seen_str}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Score + métriques
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Score ML", f"{score:.0f}/100")
        niveau_info = {
            "ACTION_URGENTE":    ("🔴", "Action urgente",  "#e24b4a", "#fde8e8"),
            "ESCALADE_N2":       ("🟠", "Escalade N2",     "#ba7517", "#fef3c7"),
            "INVESTIGATION_N1":  ("🔵", "Investigation N1","#3b8bd4", "#dbeafe"),
            "CLOTURE_FP":        ("🟢", "Clôture FP",      "#1d9e75", "#d1fae5"),
        }.get(decision, ("⚪", sev, "#888", "#f8f9fa"))
        with m2:
            st.markdown("**Priorité**")
            st.markdown(
                f"<div style='background:{niveau_info[3]}; border-left:3px solid {niveau_info[2]}; "
                f"border-radius:4px; padding:4px 8px; font-size:0.82rem; font-weight:600; color:{niveau_info[2]};'>"
                f"{niveau_info[0]} {niveau_info[1]}</div>",
                unsafe_allow_html=True
            )
        m3.metric("P(TP)", f"{incident.get('proba_tp', 0):.0f}%")
        m4.metric("P(FP)", f"{incident.get('proba_fp', 0):.0f}%")

        st.divider()

        # Tabs : Contexte | XAI | Décision
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Contexte enrichi", "🧠 Explications XAI", "⚡ Décision", "🌐 Vue globale ML"])

        # ── Tab 1 : Contexte ─────────────────────────────────────────────────
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Entités impliquées**")
                device_name = incident.get("device_name", "")
                device_id = incident.get("sample_device_id", "")
                sample_ip = incident.get("sample_ip", "")
                sample_account = incident.get("sample_account", "")
                sample_sha256 = incident.get("sample_sha256", "")

                if pd.notna(device_name) and str(device_name).strip():
                    st.markdown(f"- 🖥️ **Poste / Serveur** : `{device_name}`")
                    if pd.notna(device_id):
                        st.caption(f"   ID interne : {int(device_id)}")
                else:
                    st.markdown("- 🖥️ Poste : *non identifié*")

                if pd.notna(sample_ip):
                    st.markdown(f"- 🌐 **IP** : `{int(sample_ip)}`")
                if pd.notna(sample_account):
                    # Récupérer le libellé du compte si disponible dans la CMDB
                    user_name_val = incident.get("user_name", "")
                    user_role_val = incident.get("user_role", "")
                    user_dept_val = incident.get("department", "")
                    if pd.notna(user_name_val) and str(user_name_val).strip() not in ("", "nan"):
                        account_label = str(user_name_val)
                        if pd.notna(user_role_val) and str(user_role_val).strip() not in ("", "nan"):
                            account_label += f" ({user_role_val}"
                            if pd.notna(user_dept_val) and str(user_dept_val).strip() not in ("", "nan"):
                                account_label += f" — {user_dept_val}"
                            account_label += ")"
                        st.markdown(f"- 👤 **Compte** : `{account_label}`")
                        st.caption(f"   ID interne : {int(float(sample_account))}")
                    else:
                        st.markdown(f"- 👤 **Compte** : `{int(float(sample_account))}`")
                if pd.notna(sample_sha256):
                    st.markdown(f"- 📁 **Hash** : `{int(sample_sha256)}`")

                st.caption("⚠️ Valeurs anonymisées (dataset POC) — en production : IP, compte et hash réels.")

                nb = incident.get("nb_alerts", "—")
                st.markdown(f"- 🔔 **{nb} alertes** agrégées dans cet incident")
                st.markdown(f"- 🗂️ **Catégorie** : {cat_fr}")

                # Contexte CMDB enrichi
                asset_crit = incident.get("asset_criticality", "")
                asset_type = incident.get("asset_type", "")
                business_unit = incident.get("business_unit", "")
                os_family = incident.get("os_family", "")
                internet_exposed = incident.get("is_internet_exposed", "")
                patch_level = incident.get("patch_level", "")

                cmdb_available = any(pd.notna(v) and str(v).strip() not in ("", "nan")
                                     for v in [asset_crit, asset_type, business_unit])
                if cmdb_available:
                    st.markdown("---")
                    st.markdown("**Contexte CMDB**")
                    crit_colors = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}
                    crit_icon = crit_colors.get(str(asset_crit).upper(), "⚪")
                    if pd.notna(asset_crit):
                        st.markdown(f"- {crit_icon} **Criticité** : `{asset_crit}`")
                    if pd.notna(asset_type):
                        st.markdown(f"- 🖧 **Type** : `{asset_type}`")
                    if pd.notna(business_unit):
                        st.markdown(f"- 🏢 **Business Unit** : `{business_unit}`")
                    if pd.notna(os_family):
                        st.markdown(f"- 💻 **OS** : `{os_family}`")
                    if pd.notna(internet_exposed):
                        exposed = int(internet_exposed) == 1
                        icon = "⚠️" if exposed else "✅"
                        st.markdown(f"- {icon} **Exposé Internet** : {'Oui' if exposed else 'Non'}")
                    if pd.notna(patch_level) and str(patch_level).strip():
                        patch_icon = "⚠️" if "lag" in str(patch_level).lower() else "✅"
                        st.markdown(f"- {patch_icon} **Patch level** : `{patch_level}`")

            with c2:
                st.markdown("**Scores de risque**")
                proba_tp = incident.get("proba_tp", 0)
                iso = incident.get("iso_anomaly_score", 50)
                st.progress(min(100, int(proba_tp)) / 100,
                            text=f"Probabilité d'incident réel (TP) : {proba_tp:.0f}%")
                st.progress(min(100, int(iso)) / 100,
                            text=f"Anomalie comportementale : {iso:.0f}%")
                st.caption(
                    "Le score d'anomalie mesure à quel point cet incident est "
                    "statistiquement atypique par rapport à la masse des alertes observées."
                )

        # ── Tab 2 : XAI ──────────────────────────────────────────────────────
        with tab2:
            explanation_text = incident.get("explanation_text", "")
            explanation_human = humanize_explanation(explanation_text)
            if explanation_human:
                st.markdown(explanation_human)
            else:
                st.info("Explication non disponible pour cet incident.")

            # Section signaux faibles
            if is_sf:
                nb_alerts   = incident.get("nb_alerts", "?")
                score       = incident.get("priority_score", 0)
                ti_val      = float(incident.get("ti_score_combined", 0) or 0)
                ti_max      = float(incident.get("ti_score_max", 0) or 0)
                sample_ip   = incident.get("sample_ip", None)
                sample_sha  = incident.get("sample_sha256", None)
                device      = incident.get("sample_device", None)
                nb_ips      = float(incident.get("nb_unique_ips", 1) or 1)
                nb_devices  = float(incident.get("nb_unique_devices", 1) or 1)
                nb_accounts = float(incident.get("nb_unique_accounts", 1) or 1)
                sandbox_c2  = float(incident.get("sandbox_c2_beaconing", 0) or 0)
                sandbox_inj = float(incident.get("sandbox_process_injection", 0) or 0)
                mitre_lat   = float(incident.get("mitre_lateral_movement", 0) or 0)
                mitre_cred  = float(incident.get("mitre_credential_access", 0) or 0)
                mitre_c2    = float(incident.get("mitre_command_and_control", 0) or 0)
                mitre_exfil = float(incident.get("mitre_exfiltration", 0) or 0)
                is_priv     = float(incident.get("is_privileged_account", 0) or 0)
                spread      = float(incident.get("spread_score", 0) or 0)

                # ── Construction des signaux spécifiques à cet incident ──────
                signaux = []

                # Signal TI
                if ti_val < 5:
                    signaux.append("aucun signal Threat Intelligence (IP et hash absents des bases TI)")
                elif ti_max > 0:
                    signaux.append(f"score TI modéré ({ti_max:.0f}/100) — IP ou hash partiellement référencé")

                # Volume alertes
                signaux.append(f"seulement {int(nb_alerts)} alerte(s) agrégée(s)")

                # Propagation
                if spread > 5 or nb_ips > 3:
                    signaux.append(
                        f"propagation détectée sur {int(nb_ips)} IP(s) distincte(s)"
                        + (f", {int(nb_devices)} machine(s)" if nb_devices > 1 else "")
                        + (f", {int(nb_accounts)} compte(s)" if nb_accounts > 1 else "")
                    )

                # Sandbox enrichi
                if sandbox_c2 > 0:
                    signaux.append("communication C2 (Command & Control) confirmée en sandbox")
                if sandbox_inj > 0:
                    signaux.append("injection de processus détectée en sandbox")
                sandbox_evasion = float(incident.get("sandbox_evasion", 0) or 0)
                sandbox_lat     = float(incident.get("sandbox_lateral_movement", 0) or 0)
                if sandbox_evasion > 0:
                    signaux.append("évasion sandbox détectée — malware qui contourne l'analyse")
                if sandbox_lat > 0:
                    signaux.append("mouvement latéral détecté en sandbox")

                # MITRE
                mitre_detected = []
                if mitre_lat:   mitre_detected.append("mouvement latéral")
                if mitre_cred:  mitre_detected.append("vol de credentials")
                if mitre_c2:    mitre_detected.append("C2")
                if mitre_exfil: mitre_detected.append("exfiltration")
                if mitre_detected:
                    signaux.append(f"tactiques MITRE ATT&CK : {', '.join(mitre_detected)}")

                # Compte privilégié + nouveaux signaux user risk
                if is_priv:
                    signaux.append("compte à privilèges ciblé")
                nb_failed = float(incident.get("nb_failed_logins_7d", 0) or 0)
                mismatch  = float(incident.get("login_country_mismatch", 0) or 0)
                priv_x_logins = float(incident.get("privileged_x_failed_logins", 0) or 0)
                if nb_failed > 5:
                    signaux.append(f"{int(nb_failed)} tentatives de login échouées en 7 jours")
                if mismatch > 0:
                    signaux.append("connexion depuis un pays inhabituel (impossible travel)")
                if priv_x_logins > 0:
                    signaux.append("attaque sur compte admin détectée (credential stuffing)")

                # ── Drivers qui ont permis la détection ─────────────────────
                drivers = []
                if mitre_detected:
                    drivers.append(f"tactiques ATT&CK convergentes ({', '.join(mitre_detected)})")
                if sandbox_c2 or sandbox_inj:
                    drivers.append("analyse sandbox comportementale")
                if nb_ips > 2 or spread > 4:
                    drivers.append("score de propagation multi-entités")
                if not drivers:
                    drivers = ["historique du détecteur", "pattern temporel", "contexte organisationnel"]

                signaux_txt = " — ".join(signaux)
                drivers_txt = ", ".join(drivers)

                st.markdown("---")
                st.markdown(f"""
                <div style='background:#f5f3ff; border-left:4px solid #7c3aed;
                            border-radius:8px; padding:12px 16px;'>
                    <div style='font-size:0.95rem; font-weight:700; color:#4c1d95; margin-bottom:8px;'>
                        ⚡ Pourquoi cet incident est remarquable
                    </div>
                    <div style='font-size:0.85rem; color:#3b0764;'>
                        <p>Cet incident a été scoré <strong>{score:.0f}/100</strong> malgré :
                        <em>{signaux_txt}</em>.</p>
                        <p>Un analyste triant uniquement par volume d'alertes ou réputation TI
                        <strong>aurait pu manquer cet incident</strong>. Le modèle l'a détecté
                        grâce à : <strong>{drivers_txt}</strong>.</p>
                        <p>C'est précisément le cas d'usage pour lequel le ML apporte
                        de la valeur au-delà des règles et seuils classiques.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("")

            # ── Comparaison XGBoost vs Isolation Forest ───────────────
            st.markdown("---")
            st.markdown("**⚖️ Comparaison XGBoost (supervisé) vs Isolation Forest (non-supervisé)**")
            st.caption(
                "XGBoost apprend à partir des grades historiques des analystes. "
                "Isolation Forest détecte les comportements statistiquement anormaux "
                "sans jamais avoir vu de labels. La convergence ou divergence des deux "
                "scores donne un signal de confiance supplémentaire."
            )

            xgb_score = float(incident.get("priority_score", 0) or 0)
            iso_score  = float(incident.get("iso_anomaly_score", 0) or 0)

            col_xgb, col_iso, col_interp = st.columns([1, 1, 2])

            with col_xgb:
                xgb_color = (
                    "#e24b4a" if xgb_score >= 75 else
                    "#ba7517" if xgb_score >= 45 else
                    "#3b8bd4" if xgb_score >= 20 else "#1d9e75"
                )
                st.markdown(
                    f"<div style='text-align:center; background:#f8f9fa; "
                    f"border:2px solid {xgb_color}; border-radius:8px; padding:12px;'>"
                    f"<div style='font-size:0.8rem; color:#666;'>XGBoost (supervisé)</div>"
                    f"<div style='font-size:2rem; font-weight:700; color:{xgb_color};'>"
                    f"{xgb_score:.0f}<span style='font-size:1rem'>/100</span></div>"
                    f"<div style='font-size:0.75rem; color:#666;'>Similarité aux TP historiques</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with col_iso:
                iso_color = (
                    "#e24b4a" if iso_score >= 70 else
                    "#ba7517" if iso_score >= 40 else "#1d9e75"
                )
                st.markdown(
                    f"<div style='text-align:center; background:#f8f9fa; "
                    f"border:2px solid {iso_color}; border-radius:8px; padding:12px;'>"
                    f"<div style='font-size:0.8rem; color:#666;'>Isolation Forest (non-supervisé)</div>"
                    f"<div style='font-size:2rem; font-weight:700; color:{iso_color};'>"
                    f"{iso_score:.0f}<span style='font-size:1rem'>/100</span></div>"
                    f"<div style='font-size:0.75rem; color:#666;'>Score d'anomalie comportementale</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with col_interp:
                # Interpréter les 4 cas de convergence/divergence
                xgb_high = xgb_score >= 60
                iso_high  = iso_score  >= 50
                if xgb_high and iso_high:
                    st.success(
                        "**🟢 Convergence forte** — Les deux modèles s'accordent. "
                        "L'incident ressemble aux attaques historiques ET présente un "
                        "comportement statistiquement anormal. Signal le plus fiable pour une action urgente."
                    )
                elif xgb_high and not iso_high:
                    st.warning(
                        "**🟡 Pattern connu, comportement ordinaire** — XGBoost reconnaît "
                        "un titre d'alerte historiquement TP, mais Isolation Forest ne voit "
                        "pas de comportement vraiment anormal. Peut indiquer un incident "
                        "classique bien référencé mais de faible impact réel."
                    )
                elif not xgb_high and iso_high:
                    st.error(
                        "**🔴 Comportement nouveau et anormal** — Isolation Forest détecte "
                        "une anomalie forte mais XGBoost ne reconnaît pas de pattern connu. "
                        "**Cas typique d'une nouvelle technique d'attaque** non vue à "
                        "l'entraînement. À investiguer en priorité malgré le score XGBoost faible."
                    )
                else:
                    st.info(
                        "**⚪ Convergence sur FP probable** — Les deux modèles s'accordent "
                        "sur l'absence de signal. Clôture du FP probable, mais vérifier "
                        "les logs si doute subsiste."
                    )

            # SHAP waterfall si disponible
            if shap_data is not None:
                incident_ids_shap = shap_data.get("incident_ids", [])
                inc_id = incident.get("IncidentId")
                matches = np.where(incident_ids_shap == inc_id)[0]

                if len(matches) > 0:
                    shap_idx = matches[0]
                    shap_tp_vals = shap_data["shap_values"][2][shap_idx]
                    feature_names = shap_data["feature_names"]
                    x_vals = shap_data["X_sample"][shap_idx]

                    contrib = pd.Series(shap_tp_vals, index=feature_names)
                    top_contrib = contrib.abs().nlargest(8).index
                    contrib_top = contrib[top_contrib].sort_values()

                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors_bar = ['#e24b4a' if v > 0 else '#3b8bd4'
                                  for v in contrib_top.values]
                    ax.barh(range(len(contrib_top)), contrib_top.values,
                            color=colors_bar, height=0.6)
                    ax.set_yticks(range(len(contrib_top)))
                    feat_labels = []
                    for f in contrib_top.index:
                        fi = feature_names.index(f) if f in feature_names else -1
                        fv = x_vals[fi] if fi >= 0 else 0
                        feat_labels.append(f"{feature_label(f)}\n= {fv:.2f}")
                    ax.set_yticklabels(feat_labels, fontsize=9)
                    ax.axvline(0, color="gray", lw=0.8)
                    ax.set_xlabel("Contribution SHAP vers TP")
                    ax.set_title("Waterfall SHAP — Pourquoi ce score ?")
                    red_patch = mpatches.Patch(color='#e24b4a', label='↑ Augmente le score TP')
                    blue_patch = mpatches.Patch(color='#3b8bd4', label='↓ Réduit le score TP')
                    ax.legend(handles=[red_patch, blue_patch], fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.caption("Graphique SHAP : incident non trouvé dans l'index SHAP.")

            # ── LIME dynamique par incident ───────────────────────────────
            st.markdown("---")
            st.markdown("**🔬 LIME — Explication alternative (modèle linéaire local)**")
            st.caption(
                "LIME perturbe les valeurs de l'incident et entraîne un modèle linéaire "
                "local pour approximer XGBoost dans ce voisinage. "
                "Rouge = pousse vers TP | Bleu = tire vers FP. "
                "Si LIME et SHAP identifient les mêmes features → explication fiable."
            )

            if lime_data is not None and shap_data is not None:
                inc_id = incident.get("IncidentId")
                incident_ids_lime = lime_data.get("incident_ids", [])
                matches_lime = np.where(incident_ids_lime == inc_id)[0]

                if len(matches_lime) > 0:
                    lime_idx  = matches_lime[0]
                    lime_contribs = lime_data["lime_top_features"][lime_idx]
                    lime_contrib_full = lime_data["lime_contributions"][lime_idx]

                    # Reconstruire les paires (feature, poids) dans l'ordre
                    lime_pairs = [(f, lime_contrib_full.get(f, 0)) for f in lime_contribs]
                    lime_feats   = [feature_label(f) for f, _ in lime_pairs]
                    lime_weights = [w for _, w in lime_pairs]

                    # Convergence avec SHAP pour cet incident
                    incident_ids_shap = shap_data.get("incident_ids", [])
                    matches_shap_l = np.where(incident_ids_shap == inc_id)[0]
                    convergence_txt = ""
                    if len(matches_shap_l) > 0:
                        shap_idx_l   = matches_shap_l[0]
                        shap_tp_l    = shap_data["shap_values"][2][shap_idx_l]
                        feat_names_l = shap_data["feature_names"]
                        shap_top_l   = (pd.Series(np.abs(shap_tp_l), index=feat_names_l)
                                        .nlargest(8).index.tolist())
                        overlap_l    = set(lime_contribs) & set(shap_top_l)
                        pct_conv     = len(overlap_l) / 8
                        if pct_conv >= 0.75:
                            convergence_txt = f"🟢 Convergence SHAP/LIME : **{pct_conv:.0%}** — explication fiable"
                        elif pct_conv >= 0.50:
                            convergence_txt = f"🟡 Convergence SHAP/LIME : **{pct_conv:.0%}** — explication partielle"
                        else:
                            convergence_txt = f"🔴 Convergence SHAP/LIME : **{pct_conv:.0%}** — zone complexe, investiguer manuellement"

                    # Graphique LIME dynamique
                    fig_lime, ax_lime = plt.subplots(figsize=(8, 4))
                    colors_lime = ['#e24b4a' if w > 0 else '#3b8bd4' for w in lime_weights]
                    ax_lime.barh(range(len(lime_weights)), lime_weights,
                                 color=colors_lime, height=0.6)
                    ax_lime.set_yticks(range(len(lime_feats)))
                    ax_lime.set_yticklabels(lime_feats, fontsize=9)
                    ax_lime.invert_yaxis()
                    ax_lime.axvline(0, color='gray', lw=0.8)
                    ax_lime.set_xlabel("Coefficient LIME (→ TP)")
                    ax_lime.set_title("LIME — Approximation linéaire locale")
                    red_p  = mpatches.Patch(color='#e24b4a', label='↑ Augmente le score TP')
                    blue_p = mpatches.Patch(color='#3b8bd4', label='↓ Réduit le score TP')
                    ax_lime.legend(handles=[red_p, blue_p], fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig_lime)

                    if convergence_txt:
                        st.markdown(convergence_txt)
                else:
                    st.caption("Incident non trouvé dans l'index LIME — relancer 04_xai.ipynb.")
            else:
                st.info("Relancer 04_xai.ipynb (section LIME batch) pour activer les explications LIME.")

            # Counterfactual textuel — basé sur les vraies causes du score
            st.markdown("---")
            st.markdown("**Qu'est-ce qui changerait la décision ?**")

            expl_raw = incident.get("explanation_text", "")
            proba_tp_val = incident.get("proba_tp", 0)

            # Déterminer le vrai driver principal depuis l'explanation_text
            ti_is_zero = "ti_score_combined = 0.00" in str(expl_raw) or \
                         ("ti_score_combined" not in str(expl_raw))
            ti_is_driver = "ti_score_combined" in str(expl_raw) and \
                           "Raisons principales" in str(expl_raw) and \
                           "ti_score_combined" in str(expl_raw).split("Éléments qui réduisent")[0]
            org_tp_is_driver = False  # org_tp_rate supprimé du modèle
            mitre_is_driver = any(f"mitre_{t}" in str(expl_raw).split("Éléments qui réduisent")[0]
                                  for t in ["lateral_movement", "credential_access", "exfiltration",
                                            "command_and_control", "persistence"])

            if proba_tp_val >= 75:
                if ti_is_zero:
                    # TI est nul ET c'est déjà le cas → le driver réel est ailleurs
                    if mitre_is_driver:
                        st.warning(
                            "Le score est élevé en raison de **techniques MITRE ATT&CK critiques** "
                            "détectées (mouvement latéral, vol de credentials, C2...). "
                            "Si ces techniques n'avaient pas été détectées, le score passerait "
                            "probablement sous le seuil d'action urgente."
                        )
                    elif org_tp_is_driver:
                        st.warning(
                            "Le score est élevé principalement parce que **cette organisation a "
                            "un historique élevé d'incidents réels** pour ce type d'alerte. "
                            "Si ce taux historique baissait (moins d'incidents confirmés), "
                            "le score pourrait descendre sous le seuil d'action urgente."
                        )
                    else:
                        st.warning(
                            "Le score est élevé malgré l'absence de signal Threat Intelligence. "
                            "C'est le **volume et la nature des alertes** qui justifient l'urgence. "
                            "Si le nombre d'alertes était inférieur à 10, le score passerait "
                            "probablement sous le seuil d'action urgente."
                        )
                elif ti_is_driver:
                    st.warning(
                        "Si le score Threat Intelligence de l'IP ou du hash était nul : "
                        "le score passerait probablement sous le seuil d'action urgente → escalade N2 uniquement."
                    )
                else:
                    st.warning(
                        "Le score est élevé en raison de plusieurs signaux convergents. "
                        "Supprimer l'un d'eux seul ne suffirait pas à changer la décision — "
                        "il faudrait que plusieurs facteurs soient absents simultanément."
                    )
            elif proba_tp_val >= 45:
                if ti_is_zero:
                    st.info(
                        "Aucun signal Threat Intelligence n'est présent pour cet incident. "
                        "Si l'IP ou le hash apparaissait dans une liste TI, le score monterait "
                        "vers le seuil d'action urgente."
                    )
                else:
                    st.info(
                        "Si l'asset était classé critique dans le CMDB (serveur de production, DC) : "
                        "le score monterait vers le seuil d'action urgente."
                    )
            else:
                if ti_is_zero:
                    st.success(
                        "Aucun signal Threat Intelligence détecté. Si l'IP ou le hash était "
                        "référencé dans une liste TI, le score pourrait franchir le seuil "
                        "d'investigation N1."
                    )
                else:
                    st.success(
                        "Si le volume d'alertes augmentait significativement ou si un hash "
                        "malveillant était détecté, le score pourrait franchir le seuil "
                        "d'investigation N1."
                    )

        # ── Tab 3 : Décision ─────────────────────────────────────────────────
        with tab3:
            st.markdown("**Décision suggérée par le système :**")

            decision_map = {
                "ACTION_URGENTE": ("🔴 Action urgente", "#fde8e8", "#e24b4a",
                                   "Isoler le poste / sous-réseau concerné. Alerter le CISO."),
                "ESCALADE_N2": ("🟠 Escalade N2/N3", "#fef3c7", "#ba7517",
                                "Transférer à l'équipe N2 avec le contexte complet."),
                "INVESTIGATION_N1": ("🔵 Investigation N1", "#dbeafe", "#3b8bd4",
                                     "Approfondir l'analyse. Vérifier les logs complémentaires."),
                "CLOTURE_FP": ("🟢 Clôture FP probable", "#d1fae5", "#1d9e75",
                               "Fermer l'incident. Documenter la raison."),
            }

            label, bg, border, description = decision_map.get(
                decision, ("⚪ Inconnu", "#f8f9fa", "#888", "")
            )

            st.markdown(f"""
            <div style='background:{bg}; border:2px solid {border};
                        border-radius:8px; padding:16px; margin-bottom:12px;'>
                <div style='font-size:1.2rem; font-weight:700; color:{border};'>{label}</div>
                <div style='font-size:0.9rem; color:#333; margin-top:6px;'>{description}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**L'analyste confirme ou corrige :**")
            col_a, col_b, col_c, col_d = st.columns(4)

            grade_reel = incident.get("grade_reel_label", "?")
            with col_a:
                if st.button("✅ Confirmer", use_container_width=True, type="primary"):
                    st.success(f"Décision confirmée ! (Grade réel dataset : **{grade_reel}**)")
            with col_b:
                if st.button("🔴 Isoler poste", use_container_width=True):
                    st.error(f"Action d'isolation enregistrée. (Grade réel : **{grade_reel}**)")
            with col_c:
                if st.button("🟠 Escalader N2", use_container_width=True):
                    st.warning(f"Escalade enregistrée. (Grade réel : **{grade_reel}**)")
            with col_d:
                if st.button("🟢 Clôturer FP", use_container_width=True):
                    st.success(f"Incident clôturé. (Grade réel : **{grade_reel}**)")

            st.divider()
            st.markdown("**Feedback loop** *(théorique dans ce POC)*")
            st.caption(
                "En production, chaque décision de l'analyste est enregistrée comme nouveau label "
                "et intégrée au prochain cycle d'entraînement (batch hebdomadaire). "
                "La simulation dans le notebook 03 montre une amélioration du F2-score "
                "d'environ +2,5 points dès le premier cycle de feedback (0,792 → 0,817), "
                "le gain se stabilisant ensuite."
            )

        # ── Tab 4 : Vue globale ML ────────────────────────────────────────────
        with tab4:
            st.markdown("**Importance globale des features — calculée sur les 30 000 incidents**")

            # Distribution Isolation Forest par grade
            iso_img = DATA_DIR / "isolation_forest_scores.png"
            if iso_img.exists():
                st.image(str(iso_img),
                         caption="Isolation Forest — Distribution des scores d'anomalie par grade réel")
                st.caption(
                    "Si les TP ont un score significativement plus élevé que les FP, "
                    "Isolation Forest est complémentaire à XGBoost pour détecter les "
                    "incidents anormaux sans supervision."
                )
                st.markdown("---")
            
            st.caption(
                "Ces graphiques montrent ce que le modèle a appris sur l'ensemble du dataset, "
                "pas seulement cet incident. Ils permettent de comprendre quelles features "
                "sont structurellement les plus discriminantes pour détecter les vrais incidents."
            )

            g1, g2 = st.columns(2)
            with g1:
                shap_bar = DATA_DIR / "shap_importance_bar.png"
                if shap_bar.exists():
                    st.image(str(shap_bar),
                             caption="Top 15 features — Importance SHAP moyenne |ϕ|")
                else:
                    st.info("Relancer 04_xai.ipynb pour générer ce graphique.")
            with g2:
                shap_summary = DATA_DIR / "shap_summary_tp.png"
                if shap_summary.exists():
                    st.image(str(shap_summary),
                             caption="SHAP Summary Plot — Distribution des contributions vers TP")
                else:
                    st.info("Relancer 04_xai.ipynb pour générer ce graphique.")

            st.markdown("---")
            st.markdown("**Counterfactual — Sensibilité du modèle (exemple)**")
            cf_img = DATA_DIR / "counterfactuals.png"
            if cf_img.exists():
                st.image(str(cf_img),
                         caption="Exemple sur un incident à score élevé : comment la probabilité TP "
                                 "varie quand on modifie ses features clés. L'analyse de sensibilité "
                                 "est ici illustrée sur un incident, non calculée pour les 500.")
            else:
                st.info("Relancer 04_xai.ipynb pour générer ce graphique.")

            st.markdown("---")
            st.markdown("**Convergence SHAP/LIME — Analyse sur les 500 incidents**")
            st.caption(
                "Mesure à quel point SHAP et LIME s'accordent sur les features importantes "
                "par type de décision. ≥ 75% = explication fiable. "
                "Les incidents borderline (ESCALADE_N2) ont typiquement une convergence "
                "plus faible — le modèle hésite dans ces zones non-linéaires."
            )
            conv_img = DATA_DIR / "shap_lime_convergence_analysis.png"
            if conv_img.exists():
                st.image(str(conv_img),
                         caption="Convergence SHAP/LIME par décision et features communes")
                if lime_data is not None and shap_data is not None:
                    try:
                        lime_top_all  = lime_data.get("lime_top_features", [])
                        decisions_arr = lime_data.get("decisions", [])
                        feat_names    = shap_data.get("feature_names", [])
                        shap_tp_all   = shap_data["shap_values"][2]
                        N_TOP = 8
                        rows = []
                        for idx in range(min(len(lime_top_all), len(shap_tp_all))):
                            if not lime_top_all[idx]:
                                continue
                            shap_top = (pd.Series(np.abs(shap_tp_all[idx]), index=feat_names)
                                        .nlargest(N_TOP).index.tolist())
                            overlap  = set(shap_top) & set(lime_top_all[idx])
                            rows.append({
                                'decision':    decisions_arr[idx] if idx < len(decisions_arr) else '?',
                                'convergence': len(overlap) / N_TOP,
                            })
                        if rows:
                            conv_df = pd.DataFrame(rows)
                            st.markdown("**Convergence moyenne par décision :**")
                            dec_order  = ['ACTION_URGENTE','ESCALADE_N2',
                                          'INVESTIGATION_N1','CLOTURE_FP']
                            dec_emoji  = {'ACTION_URGENTE':'🔴','ESCALADE_N2':'🟠',
                                          'INVESTIGATION_N1':'🔵','CLOTURE_FP':'🟢'}
                            cols_conv  = st.columns(4)
                            for col, dec in zip(cols_conv, dec_order):
                                subset = conv_df[conv_df['decision'] == dec]
                                if len(subset) > 0:
                                    col.metric(
                                        f"{dec_emoji.get(dec,'⚪')} {dec.replace('_',' ')}",
                                        f"{subset['convergence'].mean():.0%}",
                                        f"N={len(subset)}"
                                    )
                    except Exception:
                        pass
            else:
                st.info("Relancer 04_xai.ipynb (section LIME batch) pour générer cette analyse.")

# ─── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "POC SOC Triage — Dataset : Microsoft GUIDE (Kaggle) | "
    "Modèles : Isolation Forest (non-supervisé) + XGBoost (supervisé) | "
    "XAI : SHAP TreeExplainer + LIME | "
    "Données simulées : TI, CMDB, Sandbox (air-gap compatible)"
)