"""
POC SOC Triage - Script 01 : Extraction représentative + Données simulées
=========================================================================
- Lit GUIDE_train.csv par chunks (pas de OOM)
- Extrait ~100k lignes stratifiées par IncidentGrade x OrgId
- Agrège au niveau incident (1 ligne = 1 incident)
- Génère TI_database.csv, CMDB.csv, Sandbox.csv simulés
  calibrés sur les valeurs réelles du dataset

Usage:
    python 01_extract_and_simulate.py --input GUIDE_train.csv --n_incidents 15000
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

# ── Paramètres ────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Colonnes à charger (on ignore les colonnes quasi-vides pour la vitesse)
USECOLS = [
    "Id", "OrgId", "IncidentId", "AlertId", "Timestamp",
    "DetectorId", "AlertTitle", "Category", "MitreTechniques",
    "IncidentGrade",
    "EntityType", "EvidenceRole",
    "DeviceId", "DeviceName",
    "Sha256", "IpAddress", "Url",
    "AccountName", "AccountObjectId",
    "FileName", "FolderPath",
    "ThreatFamily",
    "OSFamily", "OSVersion",
    "SuspicionLevel",
    "ApplicationName",
    "AntispamDirection",
    "NetworkMessageId",
    "CountryCode", "State", "City",
    "LastVerdict",
]

# On tolère les colonnes manquantes (certaines versions du dataset varient)
USECOLS_SAFE = None  # sera filtré après lecture du header

CHUNK_SIZE = 200_000  # lignes par chunk (~200MB RAM max)

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="GUIDE_train.csv", help="Chemin vers GUIDE_train.csv")
    p.add_argument("--n_incidents", type=int, default=15000, help="Nb incidents cibles dans l'extrait")
    p.add_argument("--output_dir", default="data", help="Dossier de sortie")
    return p.parse_args()

# ── Étape 1 : Extraction stratifiée ──────────────────────────────────────────

def extract_stratified(input_path: str, n_incidents: int, output_dir: Path):
    print(f"\n[1/4] Lecture du header pour détecter les colonnes disponibles...")
    header = pd.read_csv(input_path, nrows=0)
    available = [c for c in USECOLS if c in header.columns]
    print(f"  → {len(available)}/{len(USECOLS)} colonnes disponibles : {available}")

    print(f"\n[2/4] Lecture par chunks de {CHUNK_SIZE} lignes...")
    chunks = []
    total_rows = 0
    incident_ids_seen = set()

    # Passe 1 : collecter tous les IncidentId et leur grade (léger)
    print("  Passe 1 : inventaire des incidents...")
    incident_index = {}  # incident_id -> (OrgId, IncidentGrade)

    for chunk in pd.read_csv(input_path, usecols=["OrgId", "IncidentId", "IncidentGrade"],
                              chunksize=CHUNK_SIZE, low_memory=False):
        chunk = chunk.dropna(subset=["IncidentGrade"])
        for _, row in chunk.drop_duplicates("IncidentId").iterrows():
            iid = row["IncidentId"]
            if iid not in incident_index:
                incident_index[iid] = (row["OrgId"], row["IncidentGrade"])
        total_rows += len(chunk)
        print(f"    ...{total_rows:,} lignes lues, {len(incident_index):,} incidents uniques", end="\r")

    print(f"\n  Total : {total_rows:,} lignes, {len(incident_index):,} incidents uniques")

    # Stratification : proportionnelle à la distribution réelle
    index_df = pd.DataFrame.from_dict(incident_index, orient="index",
                                       columns=["OrgId", "IncidentGrade"])
    index_df.index.name = "IncidentId"
    index_df = index_df.reset_index()

    dist = index_df["IncidentGrade"].value_counts(normalize=True)
    print(f"\n  Distribution IncidentGrade dans le dataset complet :")
    for grade, pct in dist.items():
        print(f"    {grade}: {pct:.1%} ({int(pct * len(index_df)):,} incidents)")

    # Échantillon stratifié — même distribution, n_incidents cibles
    sampled = index_df.groupby("IncidentGrade", group_keys=False).apply(
        lambda g: g.sample(frac=min(1.0, n_incidents / len(index_df)), random_state=RANDOM_SEED)
    ).reset_index(drop=True)

    # Cap à n_incidents (arrondi de la stratification)
    sampled = sampled.sample(min(n_incidents, len(sampled)), random_state=RANDOM_SEED)
    selected_ids = set(sampled["IncidentId"])

    print(f"\n  Incidents sélectionnés : {len(selected_ids):,}")
    print(f"  Distribution dans l'extrait :")
    for grade, cnt in sampled["IncidentGrade"].value_counts().items():
        print(f"    {grade}: {cnt:,} ({cnt/len(sampled):.1%})")

    # Passe 2 : extraire les lignes correspondantes
    print(f"\n  Passe 2 : extraction des lignes pour les {len(selected_ids):,} incidents...")
    selected_chunks = []
    for chunk in pd.read_csv(input_path, usecols=available,
                              chunksize=CHUNK_SIZE, low_memory=False):
        mask = chunk["IncidentId"].isin(selected_ids)
        if mask.any():
            selected_chunks.append(chunk[mask])

    df_raw = pd.concat(selected_chunks, ignore_index=True)
    print(f"  Lignes extraites : {len(df_raw):,}")

    # Sauvegarde du brut
    raw_path = output_dir / "guide_extract_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    print(f"  Sauvegardé : {raw_path}")

    return df_raw, sampled

# ── Étape 2 : Agrégation au niveau incident ───────────────────────────────────

def aggregate_to_incidents(df_raw: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    print(f"\n[3/4] Agrégation au niveau incident...")

    df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"], errors="coerce")

    agg = df_raw.groupby("IncidentId").agg(
        OrgId=("OrgId", "first"),
        IncidentGrade=("IncidentGrade", "first"),
        # Temporel
        first_seen=("Timestamp", "min"),
        last_seen=("Timestamp", "max"),
        nb_alerts=("AlertId", "nunique"),
        nb_evidences=("Id", "count"),
        # Catégories et techniques
        nb_categories=("Category", "nunique"),
        top_category=("Category", lambda x: x.mode()[0] if len(x) > 0 else "Unknown"),
        mitre_techniques=("MitreTechniques", lambda x: "|".join(x.dropna().unique()[:5])),
        # Détecteurs
        nb_detectors=("DetectorId", "nunique"),
        top_detector=("DetectorId", lambda x: x.mode()[0] if len(x) > 0 else "Unknown"),
        top_alert_title=("AlertTitle", lambda x: x.mode()[0] if len(x) > 0 else "Unknown"),
        # Entités
        nb_entity_types=("EntityType", "nunique"),
        has_ip=("IpAddress", lambda x: int(x.notna().any())),
        has_file=("FileName", lambda x: int(x.notna().any())),
        has_account=("AccountName", lambda x: int(x.notna().any())),
        has_device=("DeviceId", lambda x: int(x.notna().any())),
        has_email=("NetworkMessageId", lambda x: int(x.notna().any())),
        # Valeurs clés pour enrichissement
        sample_ip=("IpAddress", lambda x: x.dropna().iloc[0] if x.notna().any() else None),
        sample_sha256=("Sha256", lambda x: x.dropna().iloc[0] if x.notna().any() else None),
        sample_filename=("FileName", lambda x: x.dropna().iloc[0] if x.notna().any() else None),
        sample_device=("DeviceName", lambda x: x.dropna().iloc[0] if x.notna().any() else None),
        sample_account=("AccountName", lambda x: x.dropna().iloc[0] if x.notna().any() else None),
        # Géo
        country=("CountryCode", lambda x: x.dropna().iloc[0] if x.notna().any() else "Unknown"),
        # Suspicion
        max_suspicion=("SuspicionLevel", lambda x: x.dropna().iloc[0] if x.notna().any() else "Unknown"),
        # OS
        os_family=("OSFamily", lambda x: x.dropna().iloc[0] if x.notna().any() else "Unknown"),
        # Threat family
        threat_family=("ThreatFamily", lambda x: x.dropna().iloc[0] if x.notna().any() else None),
        # Last verdict
        last_verdict=("LastVerdict", lambda x: x.dropna().iloc[0] if x.notna().any() else "Unknown"),
    ).reset_index()

    # Feature temporelle : durée de l'incident
    agg["incident_duration_minutes"] = (
        (agg["last_seen"] - agg["first_seen"]).dt.total_seconds() / 60
    ).fillna(0).astype(int)

    # Heure du premier événement
    agg["hour_of_day"] = agg["first_seen"].dt.hour
    agg["day_of_week"] = agg["first_seen"].dt.dayofweek
    agg["is_weekend"] = (agg["day_of_week"] >= 5).astype(int)
    agg["is_business_hours"] = ((agg["hour_of_day"] >= 8) & (agg["hour_of_day"] <= 18)).astype(int)

    out_path = output_dir / "incidents_aggregated.csv"
    agg.to_csv(out_path, index=False)
    print(f"  {len(agg):,} incidents agrégés → {out_path}")
    print(f"  Colonnes : {list(agg.columns)}")

    return agg

# ── Étape 3 : Génération des données simulées ─────────────────────────────────

def generate_simulated_data(df_incidents: pd.DataFrame, output_dir: Path):
    print(f"\n[4/4] Génération des données simulées (TI, CMDB, Sandbox)...")

    # ── TI Database ──────────────────────────────────────────────────────────
    # Basée sur les vraies IPs et hashes du dataset

    real_ips = df_incidents["sample_ip"].dropna().unique()
    real_hashes = df_incidents["sample_sha256"].dropna().unique()

    # IPs supplémentaires synthétiques pour compléter
    n_extra_ips = max(0, 500 - len(real_ips))
    extra_ips = [f"185.{np.random.randint(1,254)}.{np.random.randint(1,254)}.{np.random.randint(1,254)}"
                 for _ in range(n_extra_ips)]
    all_ips = list(real_ips) + extra_ips

    ti_ip_records = []
    for ip in all_ips:
        # Les IPs réelles issues d'incidents TP ont plus de chance d'être malveillantes
        matched_incident = df_incidents[df_incidents["sample_ip"] == ip]
        if len(matched_incident) > 0:
            grade = matched_incident["IncidentGrade"].values[0]
            if grade == "TP":
                malicious_prob = np.random.uniform(0.6, 1.0)
            elif grade == "BenignPositive":
                malicious_prob = np.random.uniform(0.05, 0.3)
            else:  # FP
                malicious_prob = np.random.uniform(0.0, 0.1)
        else:
            malicious_prob = np.random.uniform(0.0, 0.4)

        ti_score = int(malicious_prob * 100)
        categories = []
        if ti_score > 70:
            categories = np.random.choice(
                ["Malware-C2", "Botnet", "Phishing", "Ransomware", "APT", "Scanner"],
                size=np.random.randint(1, 3), replace=False
            ).tolist()
        elif ti_score > 40:
            categories = np.random.choice(
                ["Suspicious", "Spam", "Proxy", "Tor-Exit"], 
                size=1
            ).tolist()

        ti_ip_records.append({
            "ioc_value": ip,
            "ioc_type": "ip",
            "ti_score": ti_score,
            "in_blocklist": int(ti_score > 60),
            "nb_sources_flagging": max(0, int(ti_score / 20) + np.random.randint(-1, 2)),
            "threat_categories": "|".join(categories) if categories else "",
            "last_seen_malicious": pd.Timestamp("2024-01-01") + pd.Timedelta(days=np.random.randint(0, 180)) if ti_score > 40 else None,
            "geo_country": np.random.choice(["RU", "CN", "IR", "KP", "US", "DE", "FR", "NL", "BR"],
                                              p=[0.15, 0.15, 0.08, 0.05, 0.12, 0.1, 0.1, 0.1, 0.15]),
        })

    # Hashes
    ti_hash_records = []
    for sha in real_hashes:
        matched_incident = df_incidents[df_incidents["sample_sha256"] == sha]
        if len(matched_incident) > 0:
            grade = matched_incident["IncidentGrade"].values[0]
            if grade == "TP":
                vt_positives = np.random.randint(10, 72)
            elif grade == "BenignPositive":
                vt_positives = np.random.randint(0, 5)
            else:
                vt_positives = np.random.randint(0, 3)
        else:
            vt_positives = np.random.randint(0, 30)

        ti_hash_records.append({
            "ioc_value": sha,
            "ioc_type": "sha256",
            "ti_score": min(100, vt_positives * 100 // 72),
            "in_blocklist": int(vt_positives > 15),
            "nb_sources_flagging": vt_positives,
            "threat_categories": np.random.choice(["Trojan", "Ransomware", "Backdoor", "Dropper", ""]) if vt_positives > 10 else "",
            "last_seen_malicious": pd.Timestamp("2024-01-01") + pd.Timedelta(days=np.random.randint(0, 180)) if vt_positives > 5 else None,
            "geo_country": "",
        })

    ti_df = pd.DataFrame(ti_ip_records + ti_hash_records)
    ti_path = output_dir / "TI_database.csv"
    ti_df.to_csv(ti_path, index=False)
    print(f"  TI_database.csv : {len(ti_df):,} IOCs ({len(ti_ip_records)} IPs, {len(ti_hash_records)} hashes)")

    # ── CMDB ─────────────────────────────────────────────────────────────────
    real_devices = df_incidents["sample_device"].dropna().unique()
    n_extra = max(0, 300 - len(real_devices))
    extra_devices = [f"WS-{np.random.randint(1000,9999)}" for _ in range(n_extra)]
    all_devices = list(real_devices) + extra_devices

    cmdb_records = []
    criticality_map = {
        # Patterns réels issus du dataset
        "DC": "CRITICAL", "AD": "CRITICAL", "EXCHANGE": "CRITICAL",
        "SRV": "HIGH", "SERVER": "HIGH", "SQL": "HIGH", "WEB": "HIGH",
        "LAPTOP": "MEDIUM", "WS": "MEDIUM", "DESKTOP": "MEDIUM",
        "TEST": "LOW", "DEV": "LOW",
    }

    for device in all_devices:
        # Déduction de la criticité depuis le nom
        criticality = "MEDIUM"
        device_upper = str(device).upper()
        for pattern, crit in criticality_map.items():
            if pattern in device_upper:
                criticality = crit
                break

        cmdb_records.append({
            "device_name": device,
            "asset_criticality": criticality,
            "asset_type": np.random.choice(
                ["Workstation", "Server", "Domain Controller", "Laptop", "Network Device"],
                p=[0.4, 0.3, 0.05, 0.2, 0.05]
            ),
            "business_unit": np.random.choice(
                ["Finance", "HR", "IT", "Trading", "Risk", "Compliance", "Operations"],
                p=[0.2, 0.1, 0.2, 0.15, 0.1, 0.1, 0.15]
            ),
            "os_family": np.random.choice(["Windows", "Linux", "MacOS"], p=[0.7, 0.2, 0.1]),
            "is_internet_exposed": int(np.random.random() < 0.15),
            "patch_level": np.random.choice(["Up-to-date", "Minor-lag", "Critical-lag"],
                                              p=[0.5, 0.3, 0.2]),
            "sensitivity_score": {"CRITICAL": 90, "HIGH": 70, "MEDIUM": 40, "LOW": 10}[criticality]
                                  + np.random.randint(-10, 10),
            "owner_team": np.random.choice(["IT-OPS", "SOC", "DevOps", "Business"]),
        })

    cmdb_df = pd.DataFrame(cmdb_records)
    cmdb_path = output_dir / "CMDB.csv"
    cmdb_df.to_csv(cmdb_path, index=False)
    print(f"  CMDB.csv : {len(cmdb_df):,} assets")

    # ── Sandbox ───────────────────────────────────────────────────────────────
    sandbox_records = []
    for sha in real_hashes:
        matched_incident = df_incidents[df_incidents["sample_sha256"] == sha]
        grade = matched_incident["IncidentGrade"].values[0] if len(matched_incident) > 0 else "FP"

        if grade == "TP":
            verdict = np.random.choice(["Malicious", "Suspicious"], p=[0.7, 0.3])
            malware_score = np.random.randint(60, 100)
        elif grade == "BenignPositive":
            verdict = np.random.choice(["Clean", "Suspicious"], p=[0.7, 0.3])
            malware_score = np.random.randint(10, 45)
        else:
            verdict = "Clean"
            malware_score = np.random.randint(0, 20)

        sandbox_records.append({
            "sha256": sha,
            "sandbox_verdict": verdict,
            "malware_score": malware_score,
            "network_connections": np.random.randint(0, 20) if verdict != "Clean" else np.random.randint(0, 3),
            "dropped_files": np.random.randint(0, 5) if verdict == "Malicious" else 0,
            "registry_modifications": np.random.randint(0, 10) if verdict != "Clean" else 0,
            "process_injections": int(verdict == "Malicious" and np.random.random() > 0.5),
            "c2_beaconing": int(verdict == "Malicious" and np.random.random() > 0.4),
            "analysis_duration_s": np.random.randint(60, 300),
        })

    sandbox_df = pd.DataFrame(sandbox_records)
    sandbox_path = output_dir / "Sandbox.csv"
    sandbox_df.to_csv(sandbox_path, index=False)
    print(f"  Sandbox.csv : {len(sandbox_df):,} fichiers analysés")

    # ── Taux FP historique par détecteur (feature clé) ────────────────────────
    # Calculé directement depuis le dataset réel — très précieux pour le ML
    print(f"\n  Calcul du taux FP historique par détecteur...")
    # Ceci sera calculé dans le notebook de feature engineering
    # On sauvegarde juste un placeholder ici
    print(f"  (sera calculé dans 02_feature_engineering.ipynb depuis incidents_aggregated.csv)")

    return ti_df, cmdb_df, sandbox_df

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("POC SOC Triage — Extraction et simulation des données")
    print("=" * 60)

    if not os.path.exists(args.input):
        print(f"\n⚠️  Fichier introuvable : {args.input}")
        print("   Lance d'abord :")
        print("   kaggle datasets download -d Microsoft/microsoft-security-incident-prediction")
        print("   unzip microsoft-security-incident-prediction.zip")
        return

    # Étape 1 + 2
    df_raw, sampled = extract_stratified(args.input, args.n_incidents, output_dir)

    # Étape 3
    df_incidents = aggregate_to_incidents(df_raw, output_dir)

    # Étape 4
    ti_df, cmdb_df, sandbox_df = generate_simulated_data(df_incidents, output_dir)

    print("\n" + "=" * 60)
    print("✅ Extraction terminée. Fichiers générés dans ./data/ :")
    print(f"   guide_extract_raw.csv     — lignes brutes extraites")
    print(f"   incidents_aggregated.csv  — 1 ligne = 1 incident (input ML)")
    print(f"   TI_database.csv           — IOCs simulés (IPs + hashes)")
    print(f"   CMDB.csv                  — Assets simulés")
    print(f"   Sandbox.csv               — Résultats sandbox simulés")
    print("\n→ Prochaine étape : ouvrir 02_feature_engineering.ipynb")
    print("=" * 60)

if __name__ == "__main__":
    main()