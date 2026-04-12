"""
POC SOC Triage - Script 01 : Extraction + Simulation
=====================================================
version 0
- Lecture en chunks (pas de crash RAM sur 2.43 Go)
- Stratification double IncidentGrade x OrgId
- Expansion incidents liés (IP, hash, device, category, threat family, MITRE techniques)
- Features dérivées : alerts_per_min, multi_entity
- Quatre fichiers simulés séparés : TI (IPs + hashes), CMDB (devices + users), Sandbox

Usage:
    python 01_extract_and_simulate.py \
        --input GUIDE_train.csv \
        --output data/ \
        --n_incidents 15000 \
        [--no-expand]
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Colonnes réellement présentes dans GUIDE_train.csv
# (noms exacts — sensibles à la casse)
USECOLS = [
    "Id", "OrgId", "IncidentId", "AlertId", "Timestamp",
    "DetectorId", "AlertTitle", "Category", "MitreTechniques",
    "IncidentGrade",
    "EntityType", "EvidenceRole",
    "DeviceId", "DeviceName",
    "Sha256",           # pas SHA256
    "IpAddress",        # pas IPAddress
    "Url",
    "AccountName",
    "AccountObjectId",  # pas AccountId
    "FileName", "FolderPath",
    "ThreatFamily",
    "OSFamily", "OSVersion",
    "SuspicionLevel",
    "NetworkMessageId",
    "CountryCode",
    "LastVerdict",
]

CHUNK_SIZE = 200_000


# =========================================================
# CLI
# =========================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Chemin vers GUIDE_train.csv")
    p.add_argument("--output", default="data",
                   help="Dossier de sortie")
    p.add_argument("--n_incidents", type=int, default=15000,
                   help="Nb incidents cibles dans l'extrait")
    p.add_argument("--no-expand", action="store_true",
                   help="Désactiver l'expansion des incidents liés")
    return p.parse_args()


# =========================================================
# ÉTAPE 1 : LECTURE EN CHUNKS + AGRÉGATION
# =========================================================

def load_and_aggregate(input_path: str) -> pd.DataFrame:
    """
    Lit GUIDE_train.csv par chunks de 200k lignes.
    Agrège directement chaque chunk au niveau incident,
    puis re-agrège les résultats partiels.
    RAM max consommée : ~400 Mo quelle que soit la taille du fichier.
    """
    print(f"\n[1/5] Lecture par chunks ({CHUNK_SIZE:,} lignes)...")

    # Détecter les colonnes disponibles
    header = pd.read_csv(input_path, nrows=0)
    available = [c for c in USECOLS if c in header.columns]
    missing = [c for c in USECOLS if c not in header.columns]
    if missing:
        print(f"  Colonnes absentes du dataset : {missing}")
    print(f"  {len(available)} colonnes chargées")

    partial_aggs = []
    total_rows = 0

    for chunk in pd.read_csv(
        input_path,
        usecols=available,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):
        total_rows += len(chunk)
        partial_aggs.append(_aggregate_chunk(chunk))
        print(f"  ...{total_rows:,} lignes lues", end="\r")

    print(f"\n  Total lignes lues : {total_rows:,}")
    print("  Re-agrégation des chunks...")

    combined = pd.concat(partial_aggs, ignore_index=True)
    incidents = _reaggregate(combined)

    print(f"  {len(incidents):,} incidents uniques")
    return incidents


def _aggregate_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Agrégation d'un chunk au niveau IncidentId."""
    chunk = chunk.copy()
    chunk["Timestamp"] = pd.to_datetime(chunk["Timestamp"], errors="coerce")

    # Helper pour les colonnes optionnelles
    def first_notnull(col):
        return lambda x: x.dropna().iloc[0] if x.notna().any() else None

    def has_col(col):
        return col in chunk.columns

    agg_dict = {
        "OrgId":             ("OrgId", "first"),
        "IncidentGrade":     ("IncidentGrade", "first"),
        "nb_alerts":         ("AlertId", "nunique"),
        "nb_evidences":      ("Id", "count"),
        "nb_categories":     ("Category", "nunique"),
        "nb_detectors":      ("DetectorId", "nunique"),
        "nb_entity_types":   ("EntityType", "nunique"),
        "nb_unique_ips":     ("IpAddress", "nunique"),
        "nb_unique_devices": ("DeviceId", "nunique"),
        "nb_unique_accounts":("AccountName", "nunique"),
        "top_alert_title":   ("AlertTitle", lambda x: x.mode()[0] if len(x) else "Unknown"),
        "top_category":      ("Category",   lambda x: x.mode()[0] if len(x) else "Unknown"),
        "top_detector":      ("DetectorId", lambda x: x.mode()[0] if len(x) else "Unknown"),
        "mitre_techniques":  ("MitreTechniques", lambda x: "|".join(x.dropna().unique()[:5])),
        "first_seen":        ("Timestamp", "min"),
        "last_seen":         ("Timestamp", "max"),
        "sample_ip":         ("IpAddress", first_notnull("IpAddress")),
        "sample_sha256":     ("Sha256",    first_notnull("Sha256")),
        "sample_device":     ("DeviceName",first_notnull("DeviceName")),
        "sample_account":    ("AccountName", first_notnull("AccountName")),
        "has_ip":            ("IpAddress", lambda x: int(x.notna().any())),
        "has_account":       ("AccountName", lambda x: int(x.notna().any())),
        "has_device":        ("DeviceId",  lambda x: int(x.notna().any())),
        "os_family":         ("OSFamily",  first_notnull("OSFamily")),
        "max_suspicion":     ("SuspicionLevel", first_notnull("SuspicionLevel")),
        "country":           ("CountryCode", first_notnull("CountryCode")),
        "last_verdict":      ("LastVerdict", first_notnull("LastVerdict")),
        "threat_family":     ("ThreatFamily", first_notnull("ThreatFamily")),
    }

    # Colonnes optionnelles selon la version du dataset
    if has_col("FileName"):
        agg_dict["has_file"] = ("FileName", lambda x: int(x.notna().any()))
    if has_col("NetworkMessageId"):
        agg_dict["has_email"] = ("NetworkMessageId", lambda x: int(x.notna().any()))

    return chunk.groupby("IncidentId").agg(**agg_dict).reset_index()


def _reaggregate(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Re-agrège les résultats partiels de chaque chunk.
    Nécessaire car un même IncidentId peut être réparti
    sur plusieurs chunks consécutifs.
    """
    # Colonnes à prendre en "max" (flags booléens)
    max_cols = ["nb_categories", "nb_detectors", "nb_entity_types",
                "nb_unique_ips", "nb_unique_devices", "nb_unique_accounts"]
    if "has_file" in combined.columns:
        max_cols.append("has_file")
    if "has_email" in combined.columns:
        max_cols.append("has_email")

    agg_dict = {
        "OrgId":           ("OrgId", "first"),
        "IncidentGrade":   ("IncidentGrade", "first"),
        "nb_alerts":       ("nb_alerts", "sum"),
        "nb_evidences":    ("nb_evidences", "sum"),
        "top_alert_title": ("top_alert_title", "first"),
        "top_category":    ("top_category", "first"),
        "top_detector":    ("top_detector", "first"),
        "mitre_techniques":("mitre_techniques", "first"),
        "first_seen":      ("first_seen", "min"),
        "last_seen":       ("last_seen", "max"),
        "sample_ip":       ("sample_ip", "first"),
        "sample_sha256":   ("sample_sha256", "first"),
        "sample_device":   ("sample_device", "first"),
        "sample_account":  ("sample_account", "first"),
        "has_ip":          ("has_ip", "max"),
        "has_account":     ("has_account", "max"),
        "has_device":      ("has_device", "max"),
        "os_family":       ("os_family", "first"),
        "max_suspicion":   ("max_suspicion", "first"),
        "country":         ("country", "first"),
        "last_verdict":    ("last_verdict", "first"),
        "threat_family":   ("threat_family", "first"),
    }
    for col in max_cols:
        if col in combined.columns:
            agg_dict[col] = (col, "max")

    agg = combined.groupby("IncidentId").agg(**agg_dict).reset_index()

    # Features temporelles
    agg["first_seen"] = pd.to_datetime(agg["first_seen"])
    agg["last_seen"]  = pd.to_datetime(agg["last_seen"])
    agg["incident_duration_minutes"] = (
        (agg["last_seen"] - agg["first_seen"]).dt.total_seconds() / 60
    ).fillna(0).clip(upper=10080)

    agg["hour_of_day"]      = agg["first_seen"].dt.hour
    agg["day_of_week"]      = agg["first_seen"].dt.dayofweek
    agg["is_weekend"]       = (agg["day_of_week"] >= 5).astype(int)
    agg["is_business_hours"]= (
        (agg["hour_of_day"] >= 8) & (agg["hour_of_day"] <= 18)
    ).astype(int)

    # Densité d'alertes — +1 pour éviter division par zéro
    agg["alerts_per_min"] = (
        agg["nb_alerts"] / (agg["incident_duration_minutes"] + 1)
    ).clip(upper=1000)

    # Incident multi-entité : signal fort de propagation latérale
    nb_ips      = agg.get("nb_unique_ips",      pd.Series(0, index=agg.index))
    nb_devices  = agg.get("nb_unique_devices",  pd.Series(0, index=agg.index))
    nb_accounts = agg.get("nb_unique_accounts", pd.Series(0, index=agg.index))

    agg["nb_multi_entity_score"] = (
        (nb_ips      > 1).astype(int) +
        (nb_devices  > 1).astype(int) +
        (nb_accounts > 1).astype(int)
    )
    agg["multi_entity"] = (agg["nb_multi_entity_score"] >= 2).astype(int)

    return agg


# =========================================================
# ÉTAPE 2 : STRATIFIED SAMPLING
# =========================================================

def stratified_sampling(incidents: pd.DataFrame, n_incidents: int) -> set:
    """
    Échantillon stratifié sur deux axes :
    - IncidentGrade : préserve la distribution TP/BP/FP réelle
    - OrgId         : évite qu'une grosse organisation domine

    Le groupby double garantit que chaque combinaison
    (grade, org) est représentée proportionnellement.
    """
    print(f"\n[2/5] Stratification IncidentGrade x OrgId "
          f"→ {n_incidents:,} incidents...")

    dist = incidents["IncidentGrade"].value_counts()
    print("  Distribution réelle :")
    for grade, cnt in dist.items():
        print(f"    {grade}: {cnt:,} ({cnt/len(incidents):.1%})")

    sampled = (
    incidents
    .groupby(["IncidentGrade","OrgId"], group_keys=False)
    .apply(lambda g: g.sample(
        n=max(1, int(len(g)/len(incidents)*n_incidents)),
        random_state=RANDOM_SEED
    ))
    )

    sampled = sampled.sample(n=min(n_incidents, len(sampled)), random_state=RANDOM_SEED)

    selected = set(sampled["IncidentId"])
    print(f"  {len(selected):,} incidents sélectionnés")
    return selected


# =========================================================
# ÉTAPE 3 : EXPANSION DES INCIDENTS LIÉS
# =========================================================
def has_common_techniques(series, pivot_set):
    if not pivot_set:
        return pd.Series([False] * len(series), index=series.index)
    pattern = "|".join(pivot_set)
    return series.fillna("").str.contains(pattern, regex=True)

def expand_with_related_incidents(
    selected_ids: set,
    incidents: pd.DataFrame
) -> set:
    """
    Étend la sélection aux incidents partageant une entité clé
    (IP, hash, device, category, threat family, MITRE techniques)
    avec les incidents déjà sélectionnés.

    Limite : plafond à 2x n_incidents pour éviter l'explosion
    quand une IP est très partagée (ex : scanner de vulnérabilités).
    """
    print(f"\n[3/5] Expansion des incidents liés...")

    selected_rows = incidents[incidents["IncidentId"].isin(selected_ids)]

    pivot_ips     = set(selected_rows["sample_ip"].dropna())
    pivot_hashes  = set(selected_rows["sample_sha256"].dropna())
    pivot_devices = set(selected_rows["sample_device"].dropna())
    pivot_categories = set(selected_rows["top_category"].dropna())
    pivot_families = set(selected_rows["threat_family"].dropna())
    pivot_techniques = set()
    for techs in selected_rows["mitre_techniques"].dropna():
        if techs:
            pivot_techniques.update(techs.split("|"))

    print(f"  Entités pivots : {len(pivot_ips)} IPs, "
          f"{len(pivot_hashes)} hashes, "
          f"{len(pivot_devices)} devices, "
          f"{len(pivot_categories)} categories, "
          f"{len(pivot_families)} families, "
          f"{len(pivot_techniques)} techniques")

    # Compute mask once (vectorized)
    mask_tech = has_common_techniques(
    incidents["mitre_techniques"], pivot_techniques
    )

    related = incidents[
    incidents["sample_ip"].isin(pivot_ips)
    | incidents["sample_sha256"].isin(pivot_hashes)
    | incidents["sample_device"].isin(pivot_devices)
    | incidents["top_category"].isin(pivot_categories)
    | incidents["threat_family"].isin(pivot_families)
    | mask_tech
    ]["IncidentId"]

    related_ids = set(related) - selected_ids

    # Plafond : pas plus du double de la sélection initiale
    max_add = len(selected_ids)
    if len(related_ids) > max_add:
        related_ids = set(list(related_ids)[:max_add])
        print(f"  Expansion plafonnée à {max_add:,} incidents supplémentaires")

    expanded = selected_ids | related_ids
    print(f"  {len(related_ids):,} incidents liés ajoutés "
          f"(total : {len(expanded):,})")
    return expanded

# =========================================================
# ÉTAPE 4 : DONNÉES SIMULÉES (VERSION AVANCÉE)
# =========================================================

# =========================================================
# CALIBRATION DATA-DRIVEN (TI)
# =========================================================
def compute_ti_stats_from_data(incidents: pd.DataFrame) -> dict:
    """
    Calcule les paramètres (mean, std) du score TI
    directement depuis les colonnes du dataset GUIDE.

    Sources utilisées :
    - SuspicionLevel  : proxy de la suspicion détecteur
    - LastVerdict     : verdict final de l'alerte
    - threat_family   : présence d'une famille de malware connue
    """
    # Encodage numérique de SuspicionLevel
    suspicion_map = {"High": 1.0, "Medium": 0.5, "Low": 0.1, "Unknown": 0.0}
    incidents["_susp_num"] = (
        incidents["max_suspicion"]
        .map(suspicion_map)
        .fillna(0.0)
    )

    # Encodage numérique de LastVerdict
    verdict_map = {
        "Malicious":  1.0,
        "Suspicious": 0.6,
        "Clean":      0.0,
        "Unknown":    0.2,
    }
    incidents["_verdict_num"] = (
        incidents["last_verdict"]
        .map(verdict_map)
        .fillna(0.2)
    )

    # Présence d'une ThreatFamily connue
    incidents["_has_threat_family"] = (
        incidents["threat_family"].notna()
    ).astype(float)

    # Score composite 0-1 (moyenne pondérée des trois signaux) - Réduire le poids du verdict, suspicion = signal primaire
    _raw_ti = (
    0.25 * incidents["_verdict_num"] +
    0.45 * incidents["_susp_num"] +
    0.30 * incidents["_has_threat_family"]
    )

    # Calcul mean et std par grade — normalisés en 0-100
    stats = {}
    for grade in ["TruePositive", "BenignPositive", "FalsePositive"]:
        subset = incidents.loc[
            incidents["IncidentGrade"] == grade, "_raw_ti"
        ]
        if len(subset) < 10:
            # Fallback si grade trop rare dans l'extrait
            stats[grade] = (30, 15)
            continue
        mean = float(subset.mean() * 100)
        std  = float(subset.std()  * 100)
        # std minimum de 5 pour éviter une distribution trop piquée
        stats[grade] = (round(mean, 1), max(5.0, round(std, 1)))

    # Nettoyage colonnes temporaires
    incidents.drop(
        columns=["_susp_num", "_verdict_num",
                 "_has_threat_family", "_raw_ti"],
        inplace=True
    )

    print("  Paramètres TI calculés depuis le dataset :")
    for grade, (mean, std) in stats.items():
        print(f"    {grade}: mean={mean:.1f}, std={std:.1f}")

    return stats


def generate_simulated_data(incidents, output_dir):

    ti_stats = compute_ti_stats_from_data(incidents)

    global_mean = np.mean([m for m, s in ti_stats.values()])
    global_std  = np.mean([s for m, s in ti_stats.values()])
    global_stats = (global_mean, global_std)

    # IMPORTANT
    hash_scores = _generate_ti(
        incidents, output_dir, ti_stats, global_stats
    )

    _generate_cmdb(incidents, output_dir)

    _generate_sandbox(
        incidents, output_dir,
        ti_stats, global_stats,
        hash_scores
    )


# =========================================================
# SCORE TI (ANTI DATA LEAKAGE)
# =========================================================

def _ti_score_from_grade(grade: str, ti_stats: dict, global_stats: tuple) -> int:
    """
    Score TI réaliste :
    - 80% basé sur la distribution réelle (par grade)
    - 20% bruit global (indépendant du label)
    """
    mean, std = ti_stats.get(grade, (30, 15))
    global_mean, global_std = global_stats

    score_signal = np.random.normal(mean, std)
    score_noise  = np.random.normal(global_mean, global_std)

    score = 0.8 * score_signal + 0.2 * score_noise

    return int(np.clip(score, 0, 100))


# =========================================================
# TI DATABASE
# =========================================================

def _generate_ti(incidents, output_dir, ti_stats, global_stats):

    records = []
    hash_scores = {}

    # ── IPs ───────────────────────────────────────────────
    for _, row in (
        incidents[["sample_ip", "IncidentGrade"]]
        .dropna(subset=["sample_ip"])
        .drop_duplicates("sample_ip")
        .iterrows()
    ):
        score = _ti_score_from_grade(
            row["IncidentGrade"], ti_stats, global_stats
        )
        records.append({
            "ioc_value":             row["sample_ip"],
            "ioc_type":              "ip",
            "ti_score":              score,
            "ti_reputation":         ("malicious" if score > 70 else
                                      "suspicious" if score > 40 else "clean"),
            "in_blocklist":          int(score > 60),
            "nb_sources_flagging":   max(0, int(score / 20) + np.random.randint(-1, 2)),
            "ti_confidence":         round(np.random.uniform(0.4, 0.95), 2),
            "ti_first_seen_days_ago":int(np.random.randint(0, 3650)),
            "ti_trend":              np.random.choice(
                                         ["increasing", "stable", "decreasing"],
                                         p=[0.3, 0.5, 0.2]
                                     ),
            "threat_categories":     (
                                         np.random.choice(["Malware-C2", "Botnet",
                                             "Phishing", "Ransomware", "Scanner"])
                                         if score > 60 else ""
                                     ),
            "geo_country":           np.random.choice(
                                         ["RU", "CN", "IR", "KP", "US",
                                          "DE", "FR", "NL", "BR"],
                                         p=[0.15, 0.15, 0.08, 0.05, 0.12,
                                            0.10, 0.10, 0.10, 0.15]
                                     ),
        })

    # ── Hashes ────────────────────────────────────────────
    for _, row in (
        incidents[["sample_sha256", "IncidentGrade"]]
        .dropna(subset=["sample_sha256"])
        .drop_duplicates("sample_sha256")
        .iterrows()
    ):
        score = _ti_score_from_grade(
            row["IncidentGrade"], ti_stats, global_stats
        )
        sha = row["sample_sha256"]
        hash_scores[sha] = score  # mémorisé pour Sandbox
        vt  = int(score * 72 / 100)

        records.append({
            "ioc_value":             sha,
            "ioc_type":              "sha256",
            "ti_score":              score,
            "ti_reputation":         ("malicious" if score > 70 else
                                      "suspicious" if score > 40 else "clean"),
            "in_blocklist":          int(vt > 15),
            "nb_sources_flagging":   vt,
            "ti_confidence":         round(np.random.uniform(0.5, 0.98), 2),
            "ti_first_seen_days_ago":int(np.random.randint(0, 365)),
            "ti_trend":              np.random.choice(
                                         ["increasing", "stable", "decreasing"],
                                         p=[0.3, 0.5, 0.2]
                                     ),
            "threat_categories":     (
                                         np.random.choice(["Trojan", "Ransomware",
                                             "Backdoor", "Dropper"])
                                         if vt > 15 else ""
                                     ),
            "geo_country":           "",  # non applicable pour les hashes
        })

    ti_df = pd.DataFrame(records)
    ti_df.to_csv(output_dir / "TI_database.csv", index=False)

    print(f"  TI_database.csv : {len(ti_df):,} IOCs "
          f"({(ti_df.ioc_type == 'ip').sum()} IPs, "
          f"{(ti_df.ioc_type == 'sha256').sum()} hashes)")

    return hash_scores


# =========================================================
# CRITICALITY HELPER
# =========================================================

def _criticality_from_name(name: str) -> str:
    """Criticité déduite du nom de machine — convention bancaire."""
    n = str(name).upper()
    for crit, keys in {
        "CRITICAL": ["DC", "AD", "EXCHANGE", "DOMAIN"],
        "HIGH":     ["SRV", "SERVER", "SQL", "WEB", "APP", "PROXY"],
        "MEDIUM":   ["WS", "DESKTOP", "LAPTOP", "PC"],
        "LOW":      ["TEST", "DEV", "STAGING", "DEMO"],
    }.items():
        if any(k in n for k in keys):
            return crit
    return "MEDIUM"


def _criticality_from_account(account: str) -> str:
    """Criticité déduite du nom d'utilisateur."""
    a = str(account).upper()
    for crit, keys in {
        "CRITICAL": ["EXEC", "CEO", "CTO", "CFO", "ADMIN", "ROOT"],
        "HIGH":     ["MANAGER", "SUPERVISOR", "LEAD", "DIRECTOR"],
        "MEDIUM":   ["ANALYST", "ENGINEER", "DEVELOPER"],
        "LOW":      ["USER", "GUEST", "TEMP", "TEST"],
    }.items():
        if any(k in a for k in keys):
            return crit
    return "MEDIUM"


# =========================================================
# CMDB SIMULÉE
# =========================================================

def _generate_cmdb(incidents: pd.DataFrame, output_dir: Path):
    records = []
    seen = set()

    for _, row in (
        incidents[["sample_device"]]
        .dropna(subset=["sample_device"])
        .drop_duplicates("sample_device")
        .iterrows()
    ):
        device = row["sample_device"]
        if device in seen:
            continue
        seen.add(device)

        crit = _criticality_from_name(device)
        base = {"CRITICAL": 90, "HIGH": 70, "MEDIUM": 40, "LOW": 15}[crit]

        records.append({
            "device_name":       device,
            "asset_criticality": crit,
            "sensitivity_score": base + np.random.randint(-10, 10),
            "asset_type":        np.random.choice(
                                     ["Workstation", "Server",
                                      "Domain Controller", "Laptop"],
                                     p=[0.4, 0.3, 0.05, 0.25]
                                 ),
            "business_unit":     np.random.choice(
                                     ["Finance", "IT", "Trading", "Risk"],
                                     p=[0.3, 0.3, 0.2, 0.2]
                                 ),
            "os_family":         np.random.choice(
                                     ["Windows", "Linux", "MacOS"],
                                     p=[0.7, 0.2, 0.1]
                                 ),
            "is_internet_exposed":int(np.random.random() < 0.15),
            "patch_level":       np.random.choice(
                                     ["Up-to-date", "Minor-lag", "Critical-lag"],
                                     p=[0.5, 0.3, 0.2]
                                 ),
            "owner_team":        np.random.choice(
                                     ["IT-OPS", "SOC", "DevOps", "Business"]
                                 ),
        })

    # ── Utilisateurs ──────────────────────────────────────
    user_seen = set()
    for account in incidents["sample_account"].dropna().drop_duplicates():
        if account in user_seen:
            continue
        user_seen.add(account)

        crit = _criticality_from_account(account)
        base_sensitivity = {"CRITICAL": 95, "HIGH": 75, "MEDIUM": 45, "LOW": 20}[crit]

        records.append({
            "user_name":         account,
            "user_criticality":  crit,
            "sensitivity_score": base_sensitivity + np.random.randint(-5, 5),
            "user_role":         np.random.choice(
                                     ["Exec", "Manager", "Analyst", "User"],
                                     p=[0.05, 0.15, 0.4, 0.4]
                                 ),
            "department":        np.random.choice(
                                     ["Finance", "IT", "HR", "Operations", "Legal"],
                                     p=[0.25, 0.25, 0.15, 0.2, 0.15]
                                 ),
            "is_privileged":     int(crit in ["CRITICAL", "HIGH"]),
            "last_login_days":   np.random.randint(0, 30),
            "account_status":    np.random.choice(
                                     ["Active", "Inactive", "Suspended"],
                                     p=[0.9, 0.05, 0.05]
                                 ),
        })

    pd.DataFrame(records).to_csv(output_dir / "CMDB.csv", index=False)
    print(f"  CMDB.csv : {len([r for r in records if 'device_name' in r]):,} assets, "
          f"{len([r for r in records if 'user_name' in r]):,} users")


# =========================================================
# SANDBOX (corrélé avec TI)
# =========================================================

def _generate_sandbox(incidents, output_dir,
                      ti_stats, global_stats, hash_scores):

    records = []

    for _, row in (
        incidents[["sample_sha256", "IncidentGrade"]]
        .dropna(subset=["sample_sha256"])
        .drop_duplicates("sample_sha256")
        .iterrows()
    ):
        sha = row["sample_sha256"]

        # Réutilisation du score TI si disponible — cohérence TI ↔ Sandbox
        score = hash_scores.get(
            sha,
            _ti_score_from_grade(
                row["IncidentGrade"], ti_stats, global_stats
            )
        )

        if score > 70:
            verdict = "Malicious"
        elif score > 40:
            verdict = "Suspicious"
        else:
            verdict = "Clean"

        records.append({
            "sha256":                 sha,
            "sandbox_verdict":        verdict,
            "malware_score":          score,
            "network_connections":    (
                                          np.random.randint(5, 30)
                                          if verdict != "Clean"
                                          else np.random.randint(0, 3)
                                      ),
            "dropped_files":          (
                                          np.random.randint(1, 6)
                                          if verdict == "Malicious" else 0
                                      ),
            "registry_modifications": (
                                          np.random.randint(1, 15)
                                          if verdict != "Clean" else 0
                                      ),
            "process_injections":     int(
                                          verdict == "Malicious"
                                          and np.random.random() > 0.5
                                      ),
            "c2_beaconing":           int(
                                          verdict == "Malicious"
                                          and np.random.random() > 0.4
                                      ),
            "analysis_duration_s":    np.random.randint(60, 300),
        })

    pd.DataFrame(records).to_csv(output_dir / "Sandbox.csv", index=False)
    print(f"  Sandbox.csv : {len(records):,} fichiers analysés")


# =========================================================
# ÉTAPE 5 : SAUVEGARDE
# =========================================================

def save(incidents: pd.DataFrame, output_dir: Path):
    print(f"\n[5/5] Sauvegarde...")
    path = output_dir / "incidents_dataset.csv"
    incidents.to_csv(path, index=False)
    print(f"  incidents_dataset.csv : "
          f"{len(incidents):,} incidents x {incidents.shape[1]} colonnes")
    print("  Distribution finale :")
    for grade, cnt in incidents["IncidentGrade"].value_counts().items():
        print(f"    {grade}: {cnt:,} ({cnt/len(incidents):.1%})")


# =========================================================
# MAIN
# =========================================================

def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 55)
    print(" POC SOC Triage — Extraction + Simulation")
    print("=" * 55)

    if not Path(args.input).exists():
        print(f"\n  Fichier introuvable : {args.input}")
        print("  Lance d'abord :")
        print("  kaggle datasets download -d Microsoft/microsoft-security-incident-prediction")
        print("  unzip microsoft-security-incident-prediction.zip")
        return

    # 1 — Lecture et agrégation en chunks
    incidents = load_and_aggregate(args.input)

    # 2 — Échantillon stratifié IncidentGrade x OrgId
    selected_ids = stratified_sampling(incidents, args.n_incidents)

    # 3 — Expansion des incidents liés (désactivable avec --no-expand)
    if not args.no_expand:
        selected_ids = expand_with_related_incidents(selected_ids, incidents)
    else:
        print(f"\n[3/5] Expansion désactivée (--no-expand)")

    incidents = incidents[
        incidents["IncidentId"].isin(selected_ids)
    ].reset_index(drop=True)

    # 4 — Données simulées
    generate_simulated_data(incidents, output_dir)

    # 5 — Sauvegarde
    save(incidents, output_dir)

    print("\n" + "=" * 55)
    print("Terminé. Fichiers dans", output_dir)
    print("  incidents_dataset.csv")
    print("  TI_database.csv")
    print("  CMDB.csv")
    print("  Sandbox.csv")
    print("\nProchaine étape : 02_feature_engineering.ipynb")
    print("=" * 55)


if __name__ == "__main__":
    main()