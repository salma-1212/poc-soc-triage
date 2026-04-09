"""
POC SOC Triage — 05 Démo Streamlit
====================================
Lancer avec : streamlit run 05_demo_app.py

Nécessite : pip install streamlit pandas numpy matplotlib shap joblib
"""

import streamlit as st
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

st.markdown("""
<style>
.severity-CRITIQUE { background-color:#fde8e8; color:#991b1b;
    border-left:4px solid #e24b4a; padding:4px 10px; border-radius:4px; font-weight:600; }
.severity-HAUTE { background-color:#fef3c7; color:#92400e;
    border-left:4px solid #ba7517; padding:4px 10px; border-radius:4px; font-weight:600; }
.severity-MOYENNE { background-color:#dbeafe; color:#1e40af;
    border-left:4px solid #3b8bd4; padding:4px 10px; border-radius:4px; font-weight:600; }
.severity-FAIBLE { background-color:#d1fae5; color:#065f46;
    border-left:4px solid #1d9e75; padding:4px 10px; border-radius:4px; font-weight:600; }
.score-badge { font-size:2.2rem; font-weight:700; }
.metric-card { background:#f8f9fa; border-radius:8px; padding:12px; text-align:center; }
.action-box { border-radius:8px; padding:12px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)

# ─── Chargement des données ────────────────────────────────────────────────────

@st.cache_data
def load_data():
    pred = pd.read_csv(DATA_DIR / "predictions_sample.csv")
    expl = pd.read_csv(DATA_DIR / "explanations_sample.csv")
    pred = pred.merge(expl, on="IncidentId", how="left")
    pred["first_seen"] = pd.to_datetime(pred["first_seen"], errors="coerce")
    return pred

@st.cache_resource
def load_shap():
    try:
        return joblib.load(DATA_DIR / "shap_values_sample.pkl")
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
    c1.metric("Alertes", f"{len(df):,}")
    c2.metric("Critiques", f"{(df['severity']=='CRITIQUE').sum()}")
    c3.metric("À escalader", f"{(df['decision_suggested']=='ESCALADE_N2').sum()}")
    c4.metric("FP probables", f"{(df['decision_suggested']=='CLOTURE_FP').sum()}")

st.divider()

# ─── Filtres ─────────────────────────────────────────────────────────────────

col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
with col_f1:
    severity_filter = st.multiselect(
        "Sévérité", ["CRITIQUE", "HAUTE", "MOYENNE", "FAIBLE"],
        default=["CRITIQUE", "HAUTE", "MOYENNE", "FAIBLE"]
    )
with col_f2:
    decision_filter = st.multiselect(
        "Décision suggérée",
        ["ACTION_URGENTE", "ESCALADE_N2", "INVESTIGATION_N1", "CLOTURE_FP"],
        default=["ACTION_URGENTE", "ESCALADE_N2", "INVESTIGATION_N1", "CLOTURE_FP"]
    )
with col_f3:
    score_range = st.slider("Score de priorisation", 0, 100, (0, 100))

df_filtered = df[
    df["severity"].isin(severity_filter) &
    df["decision_suggested"].isin(decision_filter) &
    df["priority_score"].between(score_range[0], score_range[1])
].sort_values("priority_score", ascending=False).reset_index(drop=True)

# ─── Layout principal : file gauche + détail droite ───────────────────────────

col_queue, col_detail = st.columns([1, 2], gap="medium")

# ─── File d'alertes ───────────────────────────────────────────────────────────

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

    for i, row in df_filtered.head(30).iterrows():
        sev = row.get("severity", "MOYENNE")
        score = row.get("priority_score", 0)
        title = str(row.get("top_alert_title", "Alerte inconnue"))[:45]
        decision = row.get("decision_suggested", "")
        device = row.get("sample_device", "")
        ts = row.get("first_seen", "")

        decision_emoji = {
            "ACTION_URGENTE": "🔴",
            "ESCALADE_N2": "🟠",
            "INVESTIGATION_N1": "🔵",
            "CLOTURE_FP": "🟢",
        }.get(decision, "⚪")

        bg = SEVERITY_COLORS.get(sev, "#f8f9fa")
        border = SEVERITY_BORDER.get(sev, "#888")

        is_selected = (st.session_state.selected_idx == i)
        border_width = "3px" if is_selected else "1px"

        card_html = f"""
        <div style='background:{bg}; border-left:{border_width} solid {border};
                    border-radius:6px; padding:8px 12px; margin-bottom:6px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span style='font-weight:600; font-size:0.85rem;'>{title}…</span>
                <span style='font-size:1.1rem; font-weight:700; color:{border};'>{score:.0f}</span>
            </div>
            <div style='font-size:0.75rem; color:#555; margin-top:2px;'>
                {decision_emoji} {decision} &nbsp;|&nbsp; {device or '—'} &nbsp;|&nbsp;
                <span style='color:{border}; font-weight:600;'>{sev}</span>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        if st.button("Sélectionner", key=f"sel_{i}", use_container_width=True):
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

        st.markdown(f"""
        <div style='background:{SEVERITY_COLORS.get(sev,"#f8f9fa")};
                    border-left:5px solid {border_color};
                    border-radius:8px; padding:12px 16px; margin-bottom:12px;'>
            <div style='font-size:1.0rem; font-weight:700;'>
                {incident.get("top_alert_title", "Alerte")}
            </div>
            <div style='font-size:0.8rem; color:#555; margin-top:4px;'>
                Incident #{incident.get("IncidentId","")} &nbsp;|&nbsp;
                {str(incident.get("first_seen",""))[:16]} &nbsp;|&nbsp;
                Catégorie : {incident.get("top_category","—")}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Score + métriques
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Score ML", f"{score:.0f}/100")
        m2.metric("Sévérité", sev)
        m3.metric("Nb alertes", f"{incident.get('nb_alerts', '—')}")
        m4.metric("P(TP)", f"{incident.get('proba_tp', 0):.0f}%")
        m5.metric("P(FP)", f"{incident.get('proba_fp', 0):.0f}%")

        st.divider()

        # Tabs : Contexte | XAI | Décision
        tab1, tab2, tab3 = st.tabs(["📋 Contexte enrichi", "🧠 Explications XAI", "⚡ Décision"])

        # ── Tab 1 : Contexte ─────────────────────────────────────────────────
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Entités**")
                st.markdown(f"- 🖥️ Host : `{incident.get('sample_device','—')}`")
                st.markdown(f"- 👤 Compte : `{incident.get('sample_account','—')}`")
                st.markdown(f"- 🌐 IP : `{incident.get('sample_ip','—')}`")
                st.markdown(f"- 📁 Hash : `{str(incident.get('sample_sha256','—'))[:20]}…`")
                st.markdown(f"- 🌍 Pays : `{incident.get('country','—')}`")

            with c2:
                st.markdown("**Threat Intelligence**")
                proba_tp = incident.get("proba_tp", 0)
                iso = incident.get("iso_anomaly_score", 50)
                st.progress(min(100, int(proba_tp)) / 100,
                            text=f"Score TI estimé : {proba_tp:.0f}%")
                st.progress(min(100, int(iso)) / 100,
                            text=f"Score anomalie (Isolation Forest) : {iso:.0f}%")

                st.markdown("**Contexte CMDB (simulé)**")
                st.info("Asset : Criticité déduite du nom de machine\n"
                        "Patch level : estimé\n"
                        "Exposition Internet : non")

            st.markdown("**Score Isolation Forest (non-supervisé)**")
            iso_score = incident.get("iso_anomaly_score", 50)
            st.progress(min(100, int(iso_score)) / 100)
            st.caption(f"Score anomalie : {iso_score:.0f}/100 — "
                       f"Plus le score est élevé, plus l'incident est statistiquement atypique "
                       f"par rapport à la masse des alertes.")

        # ── Tab 2 : XAI ──────────────────────────────────────────────────────
        with tab2:
            explanation_text = incident.get("explanation_text", "")
            if explanation_text and not pd.isna(explanation_text):
                st.markdown(explanation_text)
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
                        feat_labels.append(f"{f}\n= {fv:.2f}")
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

            # Counterfactual textuel
            st.markdown("---")
            st.markdown("**Counterfactual — Qu'est-ce qui changerait la décision ?**")
            proba_tp_val = incident.get("proba_tp", 0)
            if proba_tp_val >= 75:
                st.warning("Si le score TI de l'IP était nul : le score passerait probablement "
                           "sous le seuil d'action urgente → escalade N2 uniquement.")
            elif proba_tp_val >= 45:
                st.info("Si l'asset était critique (CMDB) : le score monterait vers le seuil "
                        "d'action urgente.")
            else:
                st.success("Si l'IP apparaissait dans une liste TI : le score pourrait franchir "
                           "le seuil d'investigation N1.")

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
