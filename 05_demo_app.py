"""
POC SOC Triage — 05 Démo Streamlit
====================================
Lancer avec : streamlit run 05_demo_app.py

Nécessite : pip install streamlit pandas numpy matplotlib shap joblib
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
    try:
        cmdb = pd.read_csv(DATA_DIR / "CMDB.csv")
        cmdb_cols = ["device_id", "device_name", "asset_criticality", "asset_type",
                     "business_unit", "os_family", "is_internet_exposed", "patch_level"]
        # Dédupliquer sur device_id (le CMDB contient aussi des lignes utilisateurs)
        cmdb_devices = (cmdb[cmdb["device_id"].notna()][cmdb_cols]
                        .drop_duplicates(subset=["device_id"]))
        pred = pred.merge(cmdb_devices, left_on="sample_device_id",
                          right_on="device_id", how="left")
    except Exception:
        pass  # CMDB optionnel

    return pred

@st.cache_resource
def load_shap():
    try:
        data = joblib.load(DATA_DIR / "shap_values_sample.pkl")
        # Normalisation : gère les deux formats selon la version de SHAP utilisée lors du 04
        # Ancienne API : liste de 3 arrays (n_samples, n_features)
        # Nouvelle API : array 3D (n_samples, n_features, n_classes)
        sv = data.get("shap_values")
        if sv is not None and not isinstance(sv, list):
            data["shap_values"] = [sv[:, :, i] for i in range(sv.shape[2])]
        return data
    except Exception:
        return None

df = load_data()
shap_data = load_shap()

# ─── Header ──────────────────────────────────────────────────────────────────

col_logo, col_title, col_stats = st.columns([1, 4, 3])
with col_logo:
    st.markdown("## 🛡️")
with col_title:
    st.markdown("### SOC Triage Assistant — POC")
    st.caption("Priorisation ML + Explications XAI pour analyste N1")
with col_stats:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incidents", f"{len(df):,}")
    c2.metric("Critiques", f"{(df['severity']=='CRITIQUE').sum()}")
    c3.metric("À escalader", f"{(df['decision_suggested']=='ESCALADE_N2').sum()}")
    c4.metric("FP probables", f"{(df['decision_suggested']=='CLOTURE_FP').sum()}")

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
    "org_tp_rate":              "Taux historique d'incidents réels dans cette organisation",
    "org_fp_rate":              "Taux historique de faux positifs dans cette organisation",
    "detector_tp_rate":         "Fiabilité historique de ce type de détecteur",
    "detector_fp_rate":         "Taux de faux positifs de ce détecteur",
    "nb_alerts":                "Nombre d'alertes agrégées dans cet incident",
    "nb_detectors":             "Nombre de détecteurs différents déclenchés",
    "alert_rate_per_hour":      "Fréquence d'alertes par heure",
    "ti_score_combined":        "Score Threat Intelligence (réputation IP/hash)",
    "ti_ip_score":              "Réputation de l'adresse IP (listes TI)",
    "ti_hash_score":            "Réputation du hash fichier (listes TI)",
    "hash_malicious":           "Hash de fichier connu comme malveillant",
    "hash_nb_sources":          "Nombre de sources TI signalant ce hash",
    "ip_nb_sources":            "Nombre de sources TI signalant cette IP",
    "blocklist_hit":            "IP ou hash présent dans une liste de blocage",
    "sandbox_score":            "Score d'analyse sandbox (comportement fichier)",
    "sandbox_network_conns":    "Connexions réseau suspectes détectées en sandbox",
    "sandbox_file_drops":       "Fichiers déposés par le processus analysé",
    "asset_criticality":        "Criticité de l'actif (serveur critique, DC, etc.)",
    "asset_internet_exposed":   "Poste ou serveur exposé directement sur Internet",
    "category_tp_rate":         "Taux de vrais positifs historique pour cette catégorie",
    "category_bp_rate":         "Taux de bénins pour cette catégorie d'alerte",
    "hour_sin":                 "Heure de l'incident — composante sin (encodage cyclique)",
    "hour_cos":                 "Heure de l'incident — composante cos (encodage cyclique)",
    "day_sin":                  "Jour de la semaine — composante sin (encodage cyclique)",
    "day_cos":                  "Jour de la semaine — composante cos (encodage cyclique)",
    "hour":                     "Heure de l'incident (horaires atypiques = plus suspect)",
    "day":                      "Jour de la semaine",
    "weekend":                  "Incident survenu en dehors des heures ouvrées",
    "duration":                 "Durée de l'activité suspecte",
    "alert_title_tp_rate":      "Fiabilité historique de ce titre d'alerte précis",
    "alert_title_fp_rate":      "Taux de faux positifs pour ce titre d'alerte",
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

            col_card, col_btn = st.columns([5, 1])
            with col_card:
                st.markdown(f"""
                <div style='background:{bg}; border-left:{border_width} solid {border};
                            border-radius:6px; padding:6px 10px; margin-bottom:4px;'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='font-weight:600; font-size:0.82rem;'>{cat_fr}</span>
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

        st.markdown(f"""
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
        tab1, tab2, tab3 = st.tabs(["📋 Contexte enrichi", "🧠 Explications XAI", "⚡ Décision"])

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
                    st.markdown(f"- 👤 **Compte** : `{int(sample_account)}`")
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

            # Counterfactual textuel — basé sur les vraies causes du score
            st.markdown("---")
            st.markdown("**Counterfactual — Qu'est-ce qui changerait la décision ?**")

            expl_raw = incident.get("explanation_text", "")
            proba_tp_val = incident.get("proba_tp", 0)

            # Déterminer le vrai driver principal depuis l'explanation_text
            ti_is_zero = "ti_score_combined = 0.00" in str(expl_raw)
            ti_is_driver = ("ti_score_combined" in str(expl_raw)
                            and "↑" in str(expl_raw).split("ti_score_combined")[0][-5:]
                            or ("Raisons principales" in str(expl_raw)
                                and "ti_score_combined" in str(expl_raw).split("Éléments qui réduisent")[0]))
            org_tp_is_driver = "org_tp_rate" in str(expl_raw) and "Raisons principales" in str(expl_raw) and \
                               "org_tp_rate" in str(expl_raw).split("Éléments qui réduisent")[0]

            if proba_tp_val >= 75:
                if ti_is_zero:
                    # TI est nul ET c'est déjà le cas → le driver réel est ailleurs
                    if org_tp_is_driver:
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
                "La simulation dans le notebook 03 montre une amélioration du F2-score de "
                "+3 à +8 points selon le volume de feedbacks collectés."
            )

# ─── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "POC SOC Triage — Dataset : Microsoft GUIDE (Kaggle) | "
    "Modèles : Isolation Forest (non-supervisé) + XGBoost (supervisé) | "
    "XAI : SHAP TreeExplainer | "
    "Données simulées : TI, CMDB, Sandbox (air-gap compatible)"
)