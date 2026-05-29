"""
POC SOC Triage - Script 01 : Extraction + Simulation (v2)
==========================================================

Pipeline sur dataset Microsoft GUIDE (Freitas et al., 2024) :

  [1]   Passe 1 — identification des incidents (3 colonnes, lecture par chunks)
  [1.5] Passe pivot — index leger pour expansion (7 colonnes)
        (ignoree si --no-expand)
  [2]   Stratification IncidentGrade x OrgId (random_state=42)
  [3]   Expansion des incidents lies (IP, hash, device, category,
        threat family, MITRE techniques) — plafonnee a 2x n_incidents
        (desactivable avec --no-expand)
  [4]   Passe 2 — agregation complete des incidents selectionnes (34 colonnes)
  [5]   Simulation TI / CMDB / Sandbox calibree sur GUIDE
  [6]   Sauvegarde des fichiers de sortie

Optimisations memoire :
  - Lecture par chunks (CHUNK_SIZE = 500 000 lignes) — evite le crash RAM
    sur 2,4 Go de CSV brut.
  - Support Parquet optionnel (--to-parquet) : filtrage colonnaire pyarrow,
    passe 2 en quelques dizaines de secondes au lieu de de ~260 s en CSV 
    (run standard 30 000 incidents, --no-expand).
  - Passe pivot 7 colonnes : permet une expansion fidele sans charger
    les 34 colonnes du dataset complet.

Calibration des simulations (toutes correlees au grade SOC reel) :
  - TI    : couverture differenciee par grade (TP 30 % high score, BP/FP < 2 %)
           + actor_type (APT/cybercrime/hacktivist), nb_campaigns, fraicheur IOC.
  - CMDB  : criticite asset/user ponderee 60 % grade / 40 % bruit
           + patch lag, MFA, zones reseau, OS EOL (eol_os.csv).
  - Sandbox : evasion / persistence / lateral / privesc / C2 — probabilites
           conditionnees par LastVerdict GUIDE.

Fichiers de configuration (config/) — externalisation totale :
  eol_os.csv         — OS en fin de support (date EOL officielle par editeur)
  ti_config.csv      — Parametres TI (scores, couvertures, actor_type, geo)
  cmdb_config.csv    — Parametres CMDB (criticite, patch lag, MFA, zones)
  sandbox_config.csv — Parametres comportementaux sandbox (probas par grade)

Fichiers de sortie (data/) :
  incidents_dataset.csv  — dataset principal (56 colonnes apres agregation)
  TI_database.csv        — reputation IOC (IP + SHA256)
  CMDB.csv               — assets + users avec criticite
  Sandbox.csv            — analyse comportementale par hash
  crit_ratios.csv        — ratios de criticite utilises

Reproductibilite : RANDOM_SEED = 42 applique a np.random et a toutes les
sources de bruit (echantillonnage, simulation TI/CMDB/Sandbox).

Usage standard (reproduction du POC) :
    python 01_extract_and_simulate.py \\
        --input data/GUIDE_Train.csv \\
        --n_incidents 30000 \\
        --no-expand

Options :
    --output data/          dossier de sortie (defaut : data)
    --config config/        dossier de configuration (defaut : config)
    --to-parquet            convertit le CSV en Parquet (runs suivants plus rapides)
    --no-expand             desactive l'expansion des incidents lies
    --load-existing         recharge data/incidents_dataset.csv sans retraiter GUIDE
"""

import re
import pandas as pd
import numpy as np
import argparse
import hashlib
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Colonnes réellement présentes dans GUIDE_Train.csv
# (noms exacts — sensibles à la casse)
# AccountSid, AccountUpn, State, City, Usage, EmailClusterId,
# ApplicationId, OAuthApplicationId, ResourceIdName délibérément exclus
# (identifiants haute cardinalité ou trop creux sans signal ML)
USECOLS = [
    "Id", "OrgId", "IncidentId", "AlertId", "Timestamp",
    "DetectorId", "AlertTitle", "Category", "MitreTechniques",
    "IncidentGrade",
    "ActionGrouped", "ActionGranular",      # actions SOC — affichage seulement, pas ML
    "EntityType", "EvidenceRole",
    "DeviceId", "DeviceName",
    "Sha256",
    "IpAddress",
    "Url",
    "AccountName",
    "AccountObjectId",
    "FileName", "FolderPath",
    "RegistryKey",                           # T1547/T1112 persistance / registre
    "ApplicationName",                       # OAuth / BEC (T1528)
    "ThreatFamily",
    "OSFamily", "OSVersion",
    "AntispamDirection",                     # Inbound/Outbound — compte compromis
    "ResourceType",                          # Azure resource — cloud threat
    "SuspicionLevel",
    "NetworkMessageId",
    "CountryCode",
    "LastVerdict",
]

CHUNK_SIZE = 500_000  # augmenté de 200k → 500k : réduit le nombre de chunks et l'overhead concat


# =========================================================
# CHARGEMENT DE LA CONFIGURATION
# =========================================================

def _csv_to_raw(csv_path: Path) -> dict:
    """
    Lit un CSV (parametre, sous_cle, valeur) → dict plat {param: {sous_cle: valeur}}.
    Valeurs numériques converties automatiquement.
    Listes encodées avec | (ex: "RU|CN|FR") parsées en list.
    """
    df  = pd.read_csv(csv_path)
    out = {}
    for _, row in df.iterrows():
        param   = row["parametre"]
        sub_key = str(row["sous_cle"])
        val_raw = row["valeur"]
        if param not in out:
            out[param] = {}
        try:
            val = float(val_raw)
            val = int(val) if val == int(val) else val
        except (ValueError, TypeError):
            val = str(val_raw)
        if isinstance(val, str) and "|" in val:
            parts = val.split("|")
            try:
                val = [float(p) if "." in p else p for p in parts]
            except ValueError:
                val = parts
        out[param][sub_key] = val
    return out


def _split2(key: str, sep="_", maxsplit=1):
    """Split sécurisé — retourne (key, None) si sep absent."""
    parts = key.split(sep, maxsplit)
    return (parts[0], parts[1]) if len(parts) == 2 else (key, None)


def _rsplit2(key: str, sep="_"):
    """rsplit sécurisé — retourne (key, None) si sep absent."""
    parts = key.rsplit(sep, 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (key, None)


def load_config(config_dir: Path) -> dict:
    """
    Charge les fichiers de configuration CSV depuis config_dir.

    Fichiers attendus :
        eol_os.csv          — OS en fin de support (os_pattern, official_eol_date, …)
        ti_config.csv       — Paramètres TI simulée
        cmdb_config.csv     — Paramètres CMDB (criticité, patch lag, MFA…)
        sandbox_config.csv  — Paramètres comportements sandbox

    Format CSV commun (ti/cmdb/sandbox) :
        parametre, sous_cle, valeur, source, notes
    La colonne sous_cle encode les niveaux imbriqués avec _ :
        ex. "TruePositive_mean" → {TruePositive: {mean: valeur}}

    Retourne un dict avec les clés :
        cfg["eol_os"]    — liste de patterns OS EOL (minuscules)
        cfg["eol_dates"] — dict pattern → date EOL officielle
        cfg["ti"]        — paramètres TI reconstruits
        cfg["cmdb"]      — paramètres CMDB reconstruits
        cfg["sandbox"]   — paramètres sandbox reconstruits
    """
    cfg = {}

    # ── eol_os.csv ───────────────────────────────────────────────────────
    eol_path = config_dir / "eol_os.csv"
    if not eol_path.exists():
        raise FileNotFoundError(
            f"Fichier manquant : {eol_path}\n"
            "Colonnes attendues : os_pattern, official_eol_date, vendor, notes, source"
        )
    eol_df = pd.read_csv(eol_path)
    cfg["eol_os"]    = eol_df["os_pattern"].str.lower().tolist()
    cfg["eol_dates"] = dict(zip(eol_df["os_pattern"].str.lower(),
                                eol_df["official_eol_date"].fillna("unknown")))

    # ── ti_config.csv ────────────────────────────────────────────────────
    ti_path = config_dir / "ti_config.csv"
    if not ti_path.exists():
        raise FileNotFoundError(f"Fichier manquant : {ti_path}")
    raw = _csv_to_raw(ti_path)
    ti  = {}

    # ti_score_fallback : TruePositive_mean → {TruePositive: {mean: …}}
    ti["ti_score_fallback"] = {}
    for k, v in raw.get("ti_score_fallback", {}).items():
        grade, stat = _rsplit2(k)
        if stat:
            ti["ti_score_fallback"].setdefault(grade, {})[stat] = v

    # Clés plates directes
    for key in ["ti_score_signal_weights", "suspicion_level_numeric",
                "last_verdict_numeric", "ti_score_noise_ratio",
                "geo_country_distribution", "ti_trend_distribution"]:
        ti[key] = raw.get(key, {})

    # ti_coverage_rates : ip_TP_high_score_pct → {ip: {TP_high_score_pct: …}}
    ti["ti_coverage_rates"] = {"ip": {}, "hash": {}}
    for k, v in raw.get("ti_coverage_rates", {}).items():
        ioc, rest = _split2(k)
        if rest and ioc in ti["ti_coverage_rates"]:
            ti["ti_coverage_rates"][ioc][rest] = v

    # ti_score_params : ip_tp_high_mean → {ip: {tp_high: {mean: …}}}
    ti["ti_score_params"] = {"ip": {}, "hash": {}}
    for k, v in raw.get("ti_score_params", {}).items():
        ioc, rest = _split2(k)
        if rest and ioc in ti["ti_score_params"]:
            cat, stat = _rsplit2(rest)
            if stat:
                ti["ti_score_params"][ioc].setdefault(cat, {})[stat] = v

    # actor_type_by_grade : TruePositive_APT → {TruePositive: {APT: …}}
    ti["actor_type_by_grade"] = {}
    for k, v in raw.get("actor_type_by_grade", {}).items():
        grade, actor = _split2(k)
        if actor:
            ti["actor_type_by_grade"].setdefault(grade, {})[actor] = v

    cfg["ti"] = ti

    # ── cmdb_config.csv ───────────────────────────────────────────────────
    cmdb_path = config_dir / "cmdb_config.csv"
    if not cmdb_path.exists():
        raise FileNotFoundError(f"Fichier manquant : {cmdb_path}")
    raw  = _csv_to_raw(cmdb_path)
    cmdb = {}

    # Clés plates directes
    for key in ["asset_criticality_ratios", "device_sensitivity_base",
                "device_vuln_base_count", "device_dc_probability",
                "device_asset_type_probs", "device_business_unit_probs",
                "device_os_family_probs", "device_internet_exposed_prob_corp",
                "user_sensitivity_base", "user_mfa_probability",
                "user_login_country_mismatch_prob",
                "user_role_probs", "user_department_probs",
                "user_account_status_probs"]:
        cmdb[key] = raw.get(key, {})

    # grade_criticality_correlation : TruePositive_CRITICAL → {TP: {CRITICAL: …}}
    # grade_weight stocké avec préfixe _ pour ne pas être confondu avec un grade
    cmdb["grade_criticality_correlation"] = {}
    for k, v in raw.get("grade_criticality_correlation", {}).items():
        if k == "grade_weight":
            cmdb["grade_criticality_correlation"]["_grade_weight"] = v
            continue
        grade, crit = _split2(k)
        if crit:
            cmdb["grade_criticality_correlation"].setdefault(grade, {})[crit] = v

    # Clés avec structure {crit/grade: {stat: val}}
    # split2 (premier _) pour les clés où outer = CRITICAL/HIGH/etc. (jamais de _ dans outer)
    # rsplit2 (dernier _) pour les clés où inner = min/max/std (jamais de _ dans inner)
    for cfg_key, raw_key, split_fn in [
        ("device_patch_lag_days",        "device_patch_lag_days",        _split2),   # CRITICAL_std_ratio
        ("device_network_zone_probs",    "device_network_zone_probs",    _split2),   # CRITICAL_ISOLATED
        ("device_last_incident_days",    "device_last_incident_days",    _split2),   # TruePositive_min
        ("user_failed_logins_by_grade",  "user_failed_logins_by_grade",  _split2),   # TruePositive_mean
    ]:
        cmdb[cfg_key] = {}
        for k, v in raw.get(raw_key, {}).items():
            outer, inner = split_fn(k)
            if inner:
                cmdb[cfg_key].setdefault(outer, {})[inner] = v

    cfg["cmdb"] = cmdb

    # ── sandbox_config.csv ────────────────────────────────────────────────
    sb_path = config_dir / "sandbox_config.csv"
    if not sb_path.exists():
        raise FileNotFoundError(f"Fichier manquant : {sb_path}")
    raw = _csv_to_raw(sb_path)
    sb  = {}

    # behavior_probabilities_by_grade : TruePositive_evasion → {TP: {evasion: …}}
    sb["behavior_probabilities_by_grade"] = {}
    for k, v in raw.get("behavior_probabilities_by_grade", {}).items():
        grade, behavior = _split2(k)
        if behavior:
            sb["behavior_probabilities_by_grade"].setdefault(grade, {})[behavior] = v

    # Clés avec structure {verdict: {min/max: val}}
    for cfg_key in ["dns_requests_count_by_verdict",
                    "network_connections_by_verdict"]:
        sb[cfg_key] = {}
        for k, v in raw.get(cfg_key, {}).items():
            verdict, stat = _rsplit2(k)
            if stat:
                sb[cfg_key].setdefault(verdict, {})[stat] = v

    # Clés plates
    for key in ["dropped_files_by_verdict", "registry_modifications_by_verdict",
                "process_injection_prob", "c2_beaconing_prob",
                "analysis_duration_seconds"]:
        sb[key] = raw.get(key, {})

    cfg["sandbox"] = sb

    print(f"  Config chargée depuis {config_dir} :")
    print(f"    {len(cfg['eol_os'])} patterns OS EOL")
    print(f"    ti_config.csv, cmdb_config.csv, sandbox_config.csv")
    return cfg


# Référence globale — peuplée dans main() et accessible par toutes les fonctions
_CFG: dict = {}

def _clean_dist(d: dict) -> dict:
    """Supprime les clés de métadonnées (_description, _source, _caveat…) d'un dict de distribution."""
    return {k: v for k, v in d.items() if not str(k).startswith("_")}

def _choice_from_cfg(section: str, key: str, default: dict, rng) -> str:
    """Choisit une valeur aléatoire depuis un dict de distribution config (filtre les métadonnées)."""
    d = _clean_dist(_CFG.get(section, {}).get(key, default))
    if not d:
        d = _clean_dist(default)
    return rng.choice(list(d.keys()), p=list(d.values()))




# =========================================================
# CLI
# =========================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",
                   help="Chemin vers GUIDE_Train.csv")
    p.add_argument("--output", default="data",
                   help="Dossier de sortie")
    p.add_argument("--config", default="config",
                   help="Dossier contenant les fichiers de configuration "
                        "(eol_os.csv, ti_config.csv, cmdb_config.csv, sandbox_config.csv)")
    p.add_argument("--n_incidents", type=int, default=30000,
                   help="Nb incidents cibles dans l'extrait")
    p.add_argument("--to-parquet", action="store_true",
                   help="Convertir GUIDE_Train.csv en Parquet optimisé (une seule fois) "
                        "puis continuer le traitement normal")
    p.add_argument("--no-expand", action="store_true",
                   help="Désactiver l'expansion des incidents liés")
    p.add_argument("--load-existing", action="store_true",
                   help="Charger les incidents depuis data/incidents_dataset.csv "
                        "au lieu de traiter GUIDE_Train.csv")
    return p.parse_args()


# =========================================================
# ÉTAPE 1 : LECTURE EN CHUNKS + AGRÉGATION
# =========================================================

def load_and_aggregate(input_path: str) -> pd.DataFrame:
    """
    Passe 1 — identification de tous les incidents (3 colonnes seulement).
    Supporte CSV et Parquet.
    """
    import time
    t1 = time.perf_counter()
    print(f"\n[1/5] Passe 1 — identification des incidents...")

    is_parquet = str(input_path).endswith(".parquet")

    if is_parquet:
        import pyarrow.parquet as pq
        table = pq.read_table(
            input_path,
            columns=["IncidentId", "IncidentGrade", "OrgId"],
        )
        incidents_index = (
            table.to_pandas()
            .drop_duplicates("IncidentId")
            .reset_index(drop=True)
        )
        total_rows = table.num_rows
        n_chunks   = 1
        print(f"  Parquet · {total_rows:,} lignes → {len(incidents_index):,} incidents "
              f"en {time.perf_counter()-t1:.1f}s")
    else:
        partial_ids = []
        total_rows  = 0
        n_chunks    = 0
        for chunk in pd.read_csv(
            input_path,
            usecols=["IncidentId", "IncidentGrade", "OrgId"],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            total_rows += len(chunk)
            n_chunks   += 1
            partial_ids.append(
                chunk.drop_duplicates("IncidentId")[["IncidentId","IncidentGrade","OrgId"]]
            )
            print(f"  ...{total_rows:>10,} lignes | {n_chunks} chunks", end="\r")

        incidents_index = (
            pd.concat(partial_ids, ignore_index=True)
            .drop_duplicates("IncidentId")
            .reset_index(drop=True)
        )
        print(f"\n  Passe 1 : {total_rows:,} lignes → {len(incidents_index):,} incidents "
              f"en {time.perf_counter()-t1:.1f}s ({n_chunks} chunks)")

    dist = incidents_index["IncidentGrade"].value_counts()
    for grade, cnt in dist.items():
        print(f"    {grade}: {cnt:,} ({cnt/len(incidents_index):.1%})")

    return incidents_index, input_path, total_rows


def aggregate_selected(input_path: str, selected_ids: set) -> pd.DataFrame:
    """
    Passe 2 : agrège uniquement les incidents sélectionnés.

    Stratégie automatique selon le format du fichier source :
    - Parquet (.parquet) : filtrage colonnaire pyarrow — O(k) où k = lignes utiles.
      Lecture des seules lignes des incidents cibles sans scanner le reste.
      Typiquement 5-30s au lieu de 450s.
    - CSV : lecture par chunks avec early-stop dès que tous les IDs sont vus.
      Fallback si le Parquet n'a pas encore été généré.

    Pour générer le Parquet (une seule fois) :
        python 01_extract_and_simulate.py --input GUIDE_Train.csv \
               --to-parquet --output data/
    Puis les runs suivants passent --input data/guide_train.parquet
    """
    import time
    t0 = time.perf_counter()

    is_parquet = str(input_path).endswith(".parquet")

    if is_parquet:
        # ── Chemin Parquet : filtrage colonnaire + agrégation par batches ─
        print(f"  Parquet · {len(selected_ids):,} incidents cibles")
        try:
            import pyarrow.parquet as pq

            t_read = time.perf_counter()
            df_raw    = _read_selected_from_parquet(input_path, selected_ids)
            kept_rows = len(df_raw)
            print(f"  Lu : {kept_rows:,} lignes en {time.perf_counter()-t_read:.1f}s "
                  f"(filtrage colonnaire pyarrow)")

            # Chunker par batches d'IncidentIds (pas par position) —
            # chaque batch contient des incidents complets, sans coupure.
            # Taille de batch : CHUNK_SIZE incidents (pas lignes)
            t2         = time.perf_counter()
            all_ids    = df_raw["IncidentId"].unique()
            BATCH_SIZE = CHUNK_SIZE // 10  # ~50k incidents par batch
            partial_aggs = []
            n_batches  = 0
            for i in range(0, len(all_ids), BATCH_SIZE):
                batch_ids = set(all_ids[i:i + BATCH_SIZE])
                batch     = df_raw[df_raw["IncidentId"].isin(batch_ids)].copy()
                partial_aggs.append(_aggregate_chunk(batch))
                n_batches += 1
                pct = min(100, (i + BATCH_SIZE) / len(all_ids) * 100)
                print(f"  agrégation batch {n_batches} | {pct:.0f}%", end="\r")

            combined  = pd.concat(partial_aggs, ignore_index=True)
            incidents = _reaggregate(combined)
            print(f"\n  Agrégation : {time.perf_counter()-t2:.1f}s "
                  f"({n_batches} batches de ~{BATCH_SIZE:,} incidents) — "
                  f"{len(incidents):,} incidents")
            print(f"  Total passe 2 (parquet) : {time.perf_counter()-t0:.1f}s")
            return incidents

        except Exception as e:
            print(f"  ⚠ Erreur Parquet ({e}) — fallback CSV")
            input_path = str(input_path).replace(".parquet", ".csv")

    # ── Chemin CSV : lecture par chunks avec early-stop ───────────────────
    header    = pd.read_csv(input_path, nrows=0)
    available = [c for c in USECOLS if c in header.columns]
    missing   = [c for c in USECOLS if c not in header.columns]
    if missing:
        print(f"  Colonnes absentes : {missing}")
    print(f"  CSV · {len(available)} colonnes · {len(selected_ids):,} incidents cibles")

    partial_aggs   = []
    total_rows     = 0
    kept_rows      = 0
    n_chunks       = 0
    chunk_times    = []
    ids_remaining  = set(selected_ids)
    last_new_chunk = 0
    BUFFER_CHUNKS  = 1

    for chunk in pd.read_csv(
        input_path,
        usecols=available,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        t_chunk = time.perf_counter()
        total_rows += len(chunk)
        n_chunks   += 1

        chunk_filtered = chunk[chunk["IncidentId"].isin(selected_ids)]
        kept_rows += len(chunk_filtered)

        if len(chunk_filtered):
            seen_this_chunk = set(chunk_filtered["IncidentId"].unique())
            newly_seen      = seen_this_chunk & ids_remaining
            if newly_seen:
                ids_remaining  -= newly_seen
                last_new_chunk  = n_chunks
            partial_aggs.append(_aggregate_chunk(chunk_filtered))

        elapsed = time.perf_counter() - t_chunk
        chunk_times.append(elapsed)
        avg     = sum(chunk_times) / len(chunk_times)
        pct_found = (len(selected_ids) - len(ids_remaining)) / len(selected_ids) * 100
        print(f"  chunk {n_chunks:>2} | {total_rows:>10,} lus | {kept_rows:>8,} gardés "
              f"| {pct_found:5.1f}% trouvés | {elapsed:.1f}s (moy {avg:.1f}s)", end="\r")

        if not ids_remaining and n_chunks >= last_new_chunk + BUFFER_CHUNKS:
            print(f"\n  Early-stop chunk {n_chunks}")
            break

    t_read   = time.perf_counter() - t0
    pct_kept = kept_rows / total_rows * 100 if total_rows else 0
    print(f"\n  Lecture : {total_rows:,} lignes → {kept_rows:,} gardées ({pct_kept:.1f}%) "
          f"en {t_read:.1f}s")

    if ids_remaining:
        print(f"  ⚠ {len(ids_remaining)} incidents non trouvés dans le fichier")

    if not partial_aggs:
        raise ValueError("Aucune ligne gardée — vérifier les IncidentIds sélectionnés")

    t2 = time.perf_counter()
    combined  = pd.concat(partial_aggs, ignore_index=True)
    incidents = _reaggregate(combined)
    print(f"  Agrégation : {time.perf_counter()-t2:.1f}s — {len(incidents):,} incidents")
    print(f"  Total passe 2 : {time.perf_counter()-t0:.1f}s")

    return incidents


def _aggregate_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Agrégation d'un chunk au niveau IncidentId.

    Optimisations perf :
    - Timestamp parsé une seule fois avant groupby
    - mode() remplacé par value_counts().index[0] — 3-5× plus rapide sur grands groupes
    - most_frequent remplacé par first après tri par IncidentId+AlertId
      (le tri garantit la stabilité sans le coût du mode)
    """
    chunk = chunk.copy()
    chunk["Timestamp"] = pd.to_datetime(chunk["Timestamp"], errors="coerce")
    # Trier par IncidentId puis AlertId pour que "first" soit déterministe
    # et que les groupes arrivent déjà triés dans le groupby
    if "AlertId" in chunk.columns:
        chunk = chunk.sort_values(["IncidentId", "AlertId"])

    def first_notnull(col):
        return lambda x: x.dropna().iloc[0] if x.notna().any() else None

    def most_frequent(col):
        """value_counts().index[0] : même résultat que mode()[0], 3-5× plus rapide."""
        def _f(x):
            s = x.dropna()
            if s.empty:
                return None
            vc = s.value_counts()
            return vc.index[0] if len(vc) else None
        return _f

    def has_col(col):
        return col in chunk.columns

    agg_dict = {
        "OrgId":              ("OrgId",       "first"),
        "IncidentGrade":      ("IncidentGrade","first"),
        "nb_alerts":          ("AlertId",      "nunique"),
        "nb_evidences":       ("Id",           "count"),
        "nb_categories":      ("Category",     "nunique"),
        "nb_detectors":       ("DetectorId",   "nunique"),
        "nb_entity_types":    ("EntityType",   "nunique"),
        "nb_unique_ips":      ("IpAddress",    "nunique"),
        "nb_unique_devices":  ("DeviceId",     "nunique"),
        "nb_unique_accounts": ("AccountName",  "nunique"),
        "top_alert_title":    ("AlertTitle",   lambda x: x.value_counts().index[0] if len(x) else "Unknown"),
        "top_category":       ("Category",     lambda x: x.value_counts().index[0] if len(x) else "Unknown"),
        "top_detector":       ("DetectorId",   lambda x: x.value_counts().index[0] if len(x) else "Unknown"),
        "mitre_techniques":   ("MitreTechniques",
                               lambda x: "|".join(x.dropna().unique()[:10])),
        "first_seen":         ("Timestamp",    "min"),
        "last_seen":          ("Timestamp",    "max"),
        "sample_ip":          ("IpAddress",    first_notnull("IpAddress")),
        "sample_sha256":      ("Sha256",       first_notnull("Sha256")),
        "sample_device_id":   ("DeviceId",     most_frequent("DeviceId")),
        "sample_device":      ("DeviceName",   most_frequent("DeviceName")),
        "sample_account":     ("AccountName",  first_notnull("AccountName")),
        "has_ip":             ("IpAddress",    lambda x: int(x.notna().any())),
        "has_account":        ("AccountName",  lambda x: int(x.notna().any())),
        "has_device":         ("DeviceId",     lambda x: int(x.notna().any())),
        "os_family":          ("OSFamily",     first_notnull("OSFamily")),
        "max_suspicion":      ("SuspicionLevel", first_notnull("SuspicionLevel")),
        "country":            ("CountryCode",  first_notnull("CountryCode")),
        "last_verdict":       ("LastVerdict",  first_notnull("LastVerdict")),
        "threat_family":      ("ThreatFamily", first_notnull("ThreatFamily")),
    }

    # ── Nouvelles colonnes GUIDE ──────────────────────────────────────────

    # EvidenceRole : entités directement impactées vs connexes
    if has_col("EvidenceRole"):
        agg_dict["has_impacted_entity"] = (
            "EvidenceRole",
            lambda x: int((x == "Impacted").any()),
        )
        agg_dict["nb_impacted_entities"] = (
            "EvidenceRole",
            lambda x: int((x == "Impacted").sum()),
        )

    # RegistryKey : activité registre → persistance / T1547
    if has_col("RegistryKey"):
        agg_dict["has_registry_activity"] = (
            "RegistryKey",
            lambda x: int(x.notna().any()),
        )

    # ApplicationName : application OAuth impliquée → BEC / T1528
    if has_col("ApplicationName"):
        agg_dict["has_oauth_app"] = (
            "ApplicationName",
            lambda x: int(x.notna().any()),
        )

    # Url : URL dans l'incident → phishing, C2 HTTP
    if has_col("Url"):
        agg_dict["has_url"] = (
            "Url",
            lambda x: int(x.notna().any()),
        )

    # OSVersion : détecter les OS en fin de vie
    if has_col("OSVersion"):
        agg_dict["os_version_sample"] = (
            "OSVersion",
            first_notnull("OSVersion"),
        )

    # AntispamDirection : Outbound → compte compromis qui spamme
    if has_col("AntispamDirection"):
        agg_dict["antispam_direction"] = (
            "AntispamDirection",
            lambda x: x.dropna().value_counts().index[0] if x.notna().any() else None,
        )

    # ResourceType : ressource Azure → cloud threat
    if has_col("ResourceType"):
        agg_dict["has_cloud_resource"] = (
            "ResourceType",
            lambda x: int(x.notna().any()),
        )

    # ActionGrouped / ActionGranular : actions SOC
    # Conservés pour affichage démo — JAMAIS inclus dans features_ml.csv
    if has_col("ActionGrouped"):
        agg_dict["action_grouped"] = (
            "ActionGrouped",
            lambda x: x.dropna().value_counts().index[0] if x.notna().any() else None,
        )
        agg_dict["nb_action_types"] = (
            "ActionGrouped",
            lambda x: x.nunique(),
        )
    if has_col("ActionGranular"):
        agg_dict["top_action_granular"] = (
            "ActionGranular",
            lambda x: x.dropna().value_counts().index[0] if x.notna().any() else None,
        )

    # Colonnes optionnelles historiques
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
    # Colonnes à sommer entre chunks
    sum_cols = ["nb_alerts", "nb_evidences"]
    # Colonnes à prendre en max (counts et flags booléens)
    max_cols = [
        "nb_categories", "nb_detectors", "nb_entity_types",
        "nb_unique_ips", "nb_unique_devices", "nb_unique_accounts",
        "has_ip", "has_account", "has_device",
    ]
    for opt in ["has_file", "has_email", "has_impacted_entity",
                "nb_impacted_entities", "has_registry_activity",
                "has_oauth_app", "has_url", "has_cloud_resource",
                "nb_action_types"]:
        if opt in combined.columns:
            max_cols.append(opt)

    agg_dict = {
        "OrgId":           ("OrgId",           "first"),
        "IncidentGrade":   ("IncidentGrade",   "first"),
        "nb_alerts":       ("nb_alerts",        "sum"),
        "nb_evidences":    ("nb_evidences",     "sum"),
        "top_alert_title": ("top_alert_title",  "first"),
        "top_category":    ("top_category",     "first"),
        "top_detector":    ("top_detector",     "first"),
        "mitre_techniques":("mitre_techniques", lambda x: "|".join(
            list(dict.fromkeys(
                t for s in x.dropna() for t in s.split("|") if t
            ))[:10]
        )),
        "first_seen":      ("first_seen",       "min"),
        "last_seen":       ("last_seen",        "max"),
        "sample_ip":       ("sample_ip",        "first"),
        "sample_sha256":   ("sample_sha256",    "first"),
        "sample_device_id":("sample_device_id", lambda x: x.dropna().mode().iloc[0] if x.notna().any() else None),
        "sample_device":   ("sample_device",    lambda x: x.dropna().mode().iloc[0] if x.notna().any() else None),
        "sample_account":  ("sample_account",   "first"),
        "os_family":       ("os_family",        "first"),
        "max_suspicion":   ("max_suspicion",    "first"),
        "country":         ("country",          "first"),
        "last_verdict":    ("last_verdict",     "first"),
        "threat_family":   ("threat_family",    "first"),
    }

    # Optionnelles
    for col in ["os_version_sample", "antispam_direction",
                "action_grouped", "top_action_granular"]:
        if col in combined.columns:
            agg_dict[col] = (col, "first")

    for col in max_cols:
        if col in combined.columns:
            agg_dict[col] = (col, "max")

    agg = combined.groupby("IncidentId").agg(**agg_dict).reset_index()

    # ── Features temporelles ──────────────────────────────────────────────
    agg["first_seen"] = pd.to_datetime(agg["first_seen"])
    agg["last_seen"]  = pd.to_datetime(agg["last_seen"])

    agg["incident_duration_minutes"] = (
        (agg["last_seen"] - agg["first_seen"]).dt.total_seconds() / 60
    ).fillna(0).clip(upper=10080)

    agg["hour_of_day"]       = agg["first_seen"].dt.hour
    agg["day_of_week"]       = agg["first_seen"].dt.dayofweek
    agg["is_weekend"]        = (agg["day_of_week"] >= 5).astype(int)
    agg["is_business_hours"] = (
        (agg["hour_of_day"] >= 8) & (agg["hour_of_day"] <= 18)
    ).astype(int)
    # Connexion nocturne (0h–5h) : signal fort d'attaque planifiée / automatisée
    agg["is_after_midnight"] = (agg["hour_of_day"] < 6).astype(int)

    # Densité d'alertes — +1 pour éviter division par zéro
    agg["alerts_per_min"] = (
        agg["nb_alerts"] / (agg["incident_duration_minutes"] + 1)
    ).clip(upper=1000)

    # ── Score multi-entité (propagation latérale) ─────────────────────────
    nb_ips      = agg.get("nb_unique_ips",      pd.Series(0, index=agg.index))
    nb_devices  = agg.get("nb_unique_devices",  pd.Series(0, index=agg.index))
    nb_accounts = agg.get("nb_unique_accounts", pd.Series(0, index=agg.index))

    agg["nb_multi_entity_score"] = (
        (nb_ips      > 1).astype(int) +
        (nb_devices  > 1).astype(int) +
        (nb_accounts > 1).astype(int)
    )
    agg["multi_entity"] = (agg["nb_multi_entity_score"] >= 2).astype(int)

    # ── Nombre de techniques MITRE distinctes ────────────────────────────
    agg["nb_mitre_techniques"] = (
        agg["mitre_techniques"]
        .fillna("")
        .apply(lambda s: len([t for t in s.split("|") if t]))
    )

    # ── SuspicionLevel numérique ─────────────────────────────────────────
    suspicion_map = {"High": 1.0, "Medium": 0.5, "Low": 0.1, "Unknown": 0.0}
    agg["max_suspicion_num"] = (
        agg["max_suspicion"].map(suspicion_map).fillna(0.0)
    )

    # ── OS en fin de vie ─────────────────────────────────────────────────
    if "os_version_sample" in agg.columns:
        eol_patterns = _CFG.get("eol_os", [])
        agg["os_eol"] = (
            agg["os_version_sample"]
            .fillna("").astype(str)   # cast explicite — colonne peut être mixed-type après reaggregate
            .str.lower()
            .apply(lambda v: int(any(p in v for p in eol_patterns)))
        )
    else:
        agg["os_eol"] = 0

    # ── Direction antispam : is_outbound_spam ────────────────────────────
    if "antispam_direction" in agg.columns:
        agg["is_outbound_spam"] = (
            agg["antispam_direction"]
            .fillna("").astype(str)   # cast explicite — mixed-type possible
            .str.lower()
            .str.contains("outbound")
            .astype(int)
        )
    else:
        agg["is_outbound_spam"] = 0

    # ── Valeurs par défaut pour colonnes optionnelles ─────────────────────
    for col, default in [
        ("has_impacted_entity",  0), ("nb_impacted_entities", 0),
        ("has_registry_activity",0), ("has_oauth_app",         0),
        ("has_url",              0), ("has_cloud_resource",    0),
        ("has_file",             0), ("has_email",             0),
        ("nb_action_types",      0),
    ]:
        if col not in agg.columns:
            agg[col] = default

    # ── Incidents sans entité identifiée → NET- ───────────────────────────
    no_entity = (
        agg["sample_device_id"].isna() &
        agg["sample_ip"].isna() &
        agg["sample_account"].isna() &
        agg["sample_sha256"].isna()
    )
    agg["sample_device_id"] = agg["sample_device_id"].astype(object)
    agg["sample_device"]    = agg["sample_device"].astype(object)
    net_ids = "NET-" + agg.loc[no_entity, "IncidentId"].astype(str)
    agg.loc[no_entity, "sample_device_id"] = net_ids
    agg.loc[no_entity, "sample_device"]    = net_ids
    agg.loc[no_entity, "has_device"]       = 0

    n_net   = no_entity.sum()
    n_total = len(agg)
    if n_net > 0:
        cats = agg.loc[no_entity, "top_category"].value_counts().to_dict()
        print(f"  {n_net} incidents sans entité → NET- ({n_net/n_total:.1%})")
        print(f"  Catégories NET- : {cats}")

    country_coverage = agg["country"].notna().mean()
    print(f"  Couverture CountryCode : {country_coverage:.1%} "
          f"({'suffisant' if country_coverage > 0.3 else 'trop faible — non affiché dans la démo'})")

    # ── Log de couverture des nouvelles colonnes ──────────────────────────
    new_bool_cols = [
        "has_impacted_entity", "has_registry_activity", "has_oauth_app",
        "has_url", "has_cloud_resource", "is_outbound_spam", "os_eol",
    ]
    print("  Couverture nouvelles features :")
    for col in new_bool_cols:
        if col in agg.columns:
            rate = agg[col].mean()
            flag = " ⚠ constante (sera exclue du ML)" if rate in (0.0, 1.0) else ""
            print(f"    {col}: {rate:.1%}{flag}")

    return agg



# Colonnes nécessaires pour l'expansion — 7 colonnes au lieu de 34
# Pas de timestamp, pas de colonnes de features — juste les entités pivot
EXPANSION_COLS = [
    "IncidentId",
    "IpAddress",       # pivot IP
    "Sha256",          # pivot hash
    "DeviceId",        # pivot device
    "Category",        # pivot catégorie
    "ThreatFamily",    # pivot famille
    "MitreTechniques", # pivot techniques
]


def build_expansion_index(input_path: str) -> pd.DataFrame:
    """
    Passe 1.5 — index pivot pour l'expansion (7 colonnes, agrégation 100% vectorisée).

    Stratégie : zéro lambda Python dans le groupby — uniquement des
    agrégations natives pandas (first, nunique) qui s'exécutent en C.

    Pour chaque colonne pivot :
    - IpAddress, Sha256, DeviceId, ThreatFamily → "first" non-null
      (on n'a besoin que d'un représentant par incident pour le pivot)
    - Category → "first" (mode non nécessaire pour l'expansion)
    - MitreTechniques → concaténé par chunk puis jointure finale légère
      (on collecte tous les strings, le set intersection se fait après)

    Gain attendu vs version lambdas : 20-50× sur 466k groupes.
    """
    import time
    t0 = time.perf_counter()

    is_parquet = str(input_path).endswith(".parquet")

    # Agrégations natives — pas de lambda
    NATIVE_AGG = {
        "IpAddress":       ("sample_ip",        "first"),
        "Sha256":          ("sample_sha256",     "first"),
        "DeviceId":        ("sample_device_id",  "first"),
        "Category":        ("top_category",      "first"),
        "ThreatFamily":    ("threat_family",     "first"),
        "MitreTechniques": ("mitre_techniques",  "first"),
    }

    if is_parquet:
        import pyarrow.parquet as pq
        print(f"  Passe 1.5 — Parquet · colonnes pivot · agrégation vectorisée...")
        pivot_cols = ["IncidentId"] + [c for c in EXPANSION_COLS[1:] if c != "IncidentId"]
        table      = pq.read_table(input_path, columns=[
            c for c in pivot_cols if c in pq.read_schema(input_path).names
        ])
        df = table.to_pandas()
        total_rows = len(df)
        n_chunks   = 1
        agg_dict   = {
            out_col: (src_col, agg_fn)
            for src_col, (out_col, agg_fn) in NATIVE_AGG.items()
            if src_col in df.columns
        }
        expansion_index = (
            df.groupby("IncidentId", sort=False)
              .agg(**agg_dict)
              .reset_index()
        )
        t_elapsed = time.perf_counter() - t0
        print(f"  Passe 1.5 : {total_rows:,} lignes → {len(expansion_index):,} incidents "
              f"en {t_elapsed:.1f}s (Parquet)")
        return expansion_index

    # ── CSV ───────────────────────────────────────────────────────────────
    header     = pd.read_csv(input_path, nrows=0)
    pivot_cols = [c for c in EXPANSION_COLS if c in header.columns]
    missing    = [c for c in EXPANSION_COLS if c not in header.columns]
    if missing:
        print(f"  Colonnes pivot absentes : {missing}")
    print(f"  Passe 1.5 — {len(pivot_cols)} colonnes pivot · agrégation vectorisée...")

    partial    = []
    total_rows = 0
    n_chunks   = 0

    for chunk in pd.read_csv(
        input_path,
        usecols=pivot_cols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        total_rows += len(chunk)
        n_chunks   += 1

        agg_dict = {
            out_col: (src_col, agg_fn)
            for src_col, (out_col, agg_fn) in NATIVE_AGG.items()
            if src_col in chunk.columns
        }

        if agg_dict:
            partial.append(
                chunk.groupby("IncidentId", sort=False)
                     .agg(**agg_dict)
                     .reset_index()
            )

        print(f"  ...{total_rows:>10,} lignes | {n_chunks} chunks", end="\r")

    combined = pd.concat(partial, ignore_index=True)
    out_cols  = [v[0] for v in NATIVE_AGG.values() if v[0] in combined.columns]
    expansion_index = (
        combined.groupby("IncidentId", sort=False)
                .first()
                .reset_index()
                [["IncidentId"] + [c for c in out_cols if c in combined.columns]]
    )

    t_elapsed = time.perf_counter() - t0
    print(f"\n  Passe 1.5 : {total_rows:,} lignes → {len(expansion_index):,} incidents "
          f"en {t_elapsed:.1f}s ({n_chunks} chunks)")
    return expansion_index



def convert_to_parquet(csv_path: str, output_dir: Path) -> str:
    """
    Convertit GUIDE_Train.csv en Parquet partitionné par IncidentId.

    À faire UNE SEULE FOIS. Le fichier Parquet résultant est :
    - ~5-8× plus petit que le CSV (compression snappy)
    - ~20-50× plus rapide à lire (colonnaire — on ne lit que les colonnes utiles)
    - Filtrable par IncidentId sans lire toutes les lignes

    Le fichier est sauvegardé dans output_dir/guide_train.parquet
    et réutilisé automatiquement aux runs suivants si présent.
    """
    import time
    parquet_path = output_dir / "guide_train.parquet"

    if parquet_path.exists():
        print(f"  Parquet existant trouvé : {parquet_path}")
        return str(parquet_path)

    print(f"  Conversion CSV → Parquet (une seule fois)...")
    t0 = time.perf_counter()

    header    = pd.read_csv(csv_path, nrows=0)
    available = [c for c in USECOLS if c in header.columns]

    # Lire en chunks et écrire en Parquet incrémental
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        writer = None
        total  = 0
        for chunk in pd.read_csv(
            csv_path, usecols=available,
            chunksize=CHUNK_SIZE, low_memory=False,
        ):
            total += len(chunk)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    parquet_path, table.schema, compression="snappy"
                )
            writer.write_table(table)
            print(f"  ...{total:>10,} lignes converties", end="\r")

        if writer:
            writer.close()

        size_mb = parquet_path.stat().st_size / 1_048_576
        print(f"\n  Parquet généré : {parquet_path} "
              f"({size_mb:.0f} Mo, {time.perf_counter()-t0:.0f}s)")
        return str(parquet_path)

    except ImportError:
        print("  pyarrow non installé — pip install pyarrow")
        print("  Fallback : utilisation du CSV")
        return csv_path


def _read_selected_from_parquet(parquet_path: str, selected_ids: set) -> pd.DataFrame:
    """
    Lit uniquement les lignes des incidents sélectionnés depuis le Parquet.
    Exploite le filtrage colonnaire de pyarrow — pas de scan ligne par ligne.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa

    table = pq.read_table(
        parquet_path,
        filters=[("IncidentId", "in", list(selected_ids))],
    )
    return table.to_pandas()


# =========================================================
# ÉTAPE 2 : STRATIFIED SAMPLING
# =========================================================

def stratified_sampling(incidents: pd.DataFrame, n_incidents: int) -> set:
    """
    Échantillon stratifié sur deux axes :
    - IncidentGrade : préserve la distribution TP/BP/FP réelle
    - OrgId         : évite qu'une grosse organisation domine
    """
    print(f"\n[2/5] Stratification IncidentGrade x OrgId "
          f"→ {n_incidents:,} incidents...")

    dist = incidents["IncidentGrade"].value_counts()
    print("  Distribution réelle :")
    for grade, cnt in dist.items():
        print(f"    {grade}: {cnt:,} ({cnt/len(incidents):.1%})")

    sampled = (
        incidents
        .groupby(["IncidentGrade", "OrgId"], group_keys=False)
        .apply(lambda g: g.sample(
            n=max(1, int(len(g) / len(incidents) * n_incidents)),
            random_state=RANDOM_SEED,
        ), include_groups=False)
    )
    sampled = sampled.sample(
        n=min(n_incidents, len(sampled)), random_state=RANDOM_SEED
    )
    selected = set(sampled["IncidentId"])
    print(f"  {len(selected):,} incidents sélectionnés")
    return selected


# =========================================================
# ÉTAPE 3 : EXPANSION DES INCIDENTS LIÉS
# =========================================================

def has_common_techniques(series, pivot_set):
    """
    Détecte les incidents partageant au moins une technique MITRE avec pivot_set.

    Optimisation v2 : set intersection au lieu de regex.
    Le regex sur 494 techniques génère un pattern de ~4 Ko appliqué sur
    chaque cellule — O(n × |pattern|). L'approche set est O(n × k) avec
    k = nb techniques par incident (typiquement 1-5).
    Gain mesuré : ~10× plus rapide sur 30k incidents avec 494 pivot techniques.
    """
    if not pivot_set:
        return pd.Series([False] * len(series), index=series.index)

    def _has_overlap(cell: str) -> bool:
        if not cell:
            return False
        return not pivot_set.isdisjoint(cell.split("|"))

    return series.fillna("").apply(_has_overlap)


def expand_with_related_incidents(
    selected_ids: set,
    incidents: pd.DataFrame,
) -> set:
    """
    Étend la sélection aux incidents partageant une entité clé
    (IP, hash, device, category, threat family, MITRE techniques).
    Plafond à 2× n_incidents.
    """
    import time
    t0 = time.perf_counter()
    print(f"\n[3/5] Expansion des incidents liés...")

    selected_rows = incidents[incidents["IncidentId"].isin(selected_ids)]

    t1 = time.perf_counter()
    pivot_ips        = set(selected_rows["sample_ip"].dropna())
    pivot_hashes     = set(selected_rows["sample_sha256"].dropna())
    pivot_devices    = set(selected_rows["sample_device_id"].dropna())
    pivot_categories = set(selected_rows["top_category"].dropna())
    pivot_families   = set(selected_rows["threat_family"].dropna())
    pivot_techniques = set()
    for techs in selected_rows["mitre_techniques"].dropna():
        if techs:
            pivot_techniques.update(techs.split("|"))
    print(f"  Calcul pivots : {time.perf_counter()-t1:.2f}s")

    print(f"  Entités pivots : {len(pivot_ips)} IPs, "
          f"{len(pivot_hashes)} hashes, "
          f"{len(pivot_devices)} devices, "
          f"{len(pivot_categories)} categories, "
          f"{len(pivot_families)} families, "
          f"{len(pivot_techniques)} techniques")

    t2 = time.perf_counter()
    mask_tech = has_common_techniques(
        incidents["mitre_techniques"], pivot_techniques
    )
    print(f"  Masque MITRE (set intersection) : {time.perf_counter()-t2:.2f}s")

    t3 = time.perf_counter()
    related = incidents[
        incidents["sample_ip"].isin(pivot_ips)
        | incidents["sample_sha256"].isin(pivot_hashes)
        | incidents["sample_device_id"].isin(pivot_devices)
        | incidents["top_category"].isin(pivot_categories)
        | incidents["threat_family"].isin(pivot_families)
        | mask_tech
    ]["IncidentId"]
    print(f"  Filtrage incidents liés : {time.perf_counter()-t3:.2f}s")

    related_ids = set(related) - selected_ids
    max_add = len(selected_ids)
    if len(related_ids) > max_add:
        related_ids = set(list(related_ids)[:max_add])
        print(f"  Expansion plafonnée à {max_add:,} incidents supplémentaires")

    expanded = selected_ids | related_ids
    print(f"  {len(related_ids):,} incidents liés ajoutés "
          f"(total : {len(expanded):,}) — étape 3 : {time.perf_counter()-t0:.1f}s")
    return expanded


# =========================================================
# ÉTAPE 4 : DONNÉES SIMULÉES
# =========================================================

# ── Calibration TI data-driven ────────────────────────────────────────────

def compute_ti_stats_from_data(incidents: pd.DataFrame) -> dict:
    """
    Calcule les paramètres (mean, std) du score TI depuis les colonnes GUIDE.
    Valeurs de référence réalistes (Verizon DBIR, MSTIC, CrowdStrike) si signal trop faible.
    """
    GRADES = ["TruePositive", "BenignPositive", "FalsePositive"]
    _ti_cfg  = _CFG.get("ti", {})
    _fallback_cfg = _ti_cfg.get("ti_score_fallback", {})
    FALLBACK = {
        g: (
            _fallback_cfg.get(g, {}).get("mean", 30.0),
            _fallback_cfg.get(g, {}).get("std",  15.0),
        )
        for g in GRADES
    }

    suspicion_map = _ti_cfg.get("suspicion_level_numeric",
                                {"High": 1.0, "Medium": 0.5, "Low": 0.1, "Unknown": 0.0})
    incidents["_susp_num"] = (
        incidents["max_suspicion"].map(suspicion_map)
        .infer_objects(copy=False).fillna(0.0)
    )
    verdict_map = _ti_cfg.get("last_verdict_numeric",
                              {"Malicious": 1.0, "Suspicious": 0.6,
                               "NoThreatsFound": 0.0, "Clean": 0.0, "Unknown": 0.1})
    incidents["_verdict_num"] = (
        incidents["last_verdict"].map(verdict_map)
        .infer_objects(copy=False).fillna(0.1)
    )
    incidents["_has_threat_family"] = incidents["threat_family"].notna().astype(float)

    _w = _ti_cfg.get("ti_score_signal_weights", {})
    _wv = _w.get("verdict_weight",       0.25)
    _ws = _w.get("suspicion_weight",     0.45)
    _wf = _w.get("threat_family_weight", 0.30)
    _raw_ti = (
        _wv * incidents["_verdict_num"] +
        _ws * incidents["_susp_num"] +
        _wf * incidents["_has_threat_family"]
    )

    stats = {}
    for grade in GRADES:
        subset = _raw_ti[incidents["IncidentGrade"] == grade]
        fallback_mean, fallback_std = FALLBACK[grade]
        if len(subset) < 10:
            stats[grade] = FALLBACK[grade]
            continue
        mean = float(subset.mean() * 100)
        std  = float(subset.std()  * 100)
        if mean < fallback_mean * 0.5:
            mean = fallback_mean
            std  = fallback_std
            print(f"    {grade}: signal GUIDE trop faible → valeurs réalistes de référence")
        stats[grade] = (round(mean, 1), max(5.0, round(std, 1)))

    incidents.drop(
        columns=["_susp_num", "_verdict_num", "_has_threat_family"],
        inplace=True,
    )
    print("  Paramètres TI finaux :")
    for grade, (mean, std) in stats.items():
        print(f"    {grade}: mean={mean:.1f}, std={std:.1f}")
    return stats


def generate_simulated_data(incidents, output_dir, crit_ratios=None):
    ti_stats = compute_ti_stats_from_data(incidents)

    global_mean  = np.mean([m for m, s in ti_stats.values()])
    global_std   = np.mean([s for m, s in ti_stats.values()])
    global_stats = (global_mean, global_std)

    hash_scores = _generate_ti(incidents, output_dir, ti_stats, global_stats)
    _generate_cmdb(incidents, output_dir, crit_ratios)
    _generate_sandbox(incidents, output_dir, ti_stats, global_stats, hash_scores)


# ── Score TI (anti data leakage) ─────────────────────────────────────────

def _ti_score_from_grade(grade: str, ti_stats: dict, global_stats: tuple) -> int:
    mean, std = ti_stats.get(grade, (30, 15))
    global_mean, global_std = global_stats
    _noise_cfg = _CFG.get("ti", {}).get("ti_score_noise_ratio", {})
    sig = _noise_cfg.get("signal_ratio", 0.95)
    noi = _noise_cfg.get("noise_ratio",  0.05)
    score = sig * np.random.normal(mean, std) + noi * np.random.normal(global_mean, global_std)
    return int(np.clip(score, 0, 100))


# ── TI DATABASE ───────────────────────────────────────────────────────────

def _generate_ti(incidents, output_dir, ti_stats, global_stats):
    """
    Génère TI_database.csv — base de réputation IOC réaliste.

    IPs exclusivement TP : 30% score élevé, 20% score faible, 50% absentes
    IPs BP/FP            : 2% score élevé, 1% score faible, 97% absentes
    Même logique pour les hashes SHA256.

    Enrichissements v1 :
    - actor_type : APT / cybercrime / hacktivist / unknown
    - nb_campaigns : nb campagnes connues liées à l'IOC
    - ti_last_seen_days_ago : fraîcheur de l'IOC
    """
    records    = []
    hash_scores = {}
    rng = np.random.default_rng(RANDOM_SEED)

    # ── IOCs exclusifs par grade ──────────────────────────────────────────
    ip_grade_sets = (
        incidents.dropna(subset=["sample_ip"])
        .groupby("sample_ip")["IncidentGrade"]
        .apply(set)
    )
    hash_grade_sets = (
        incidents.dropna(subset=["sample_sha256"])
        .groupby("sample_sha256")["IncidentGrade"]
        .apply(set)
    )

    ip_excl_tp = sorted([ip for ip, gs in ip_grade_sets.items() if gs == {"TruePositive"}])
    ip_excl_bp = sorted([ip for ip, gs in ip_grade_sets.items() if gs == {"BenignPositive"}])
    ip_excl_fp = sorted([ip for ip, gs in ip_grade_sets.items() if gs == {"FalsePositive"}])

    hash_excl_tp = sorted([h for h, gs in hash_grade_sets.items() if gs == {"TruePositive"}])
    hash_excl_bp = sorted([h for h, gs in hash_grade_sets.items() if gs == {"BenignPositive"}])
    hash_excl_fp = sorted([h for h, gs in hash_grade_sets.items() if gs == {"FalsePositive"}])

    print(f"  IPs exclusives — TP:{len(ip_excl_tp)}, BP:{len(ip_excl_bp)}, FP:{len(ip_excl_fp)}")
    print(f"  Hashes exclusifs — TP:{len(hash_excl_tp)}, BP:{len(hash_excl_bp)}, FP:{len(hash_excl_fp)}")

    def select_pct(lst, pct, rng):
        n   = max(1, int(len(lst) * pct))
        idx = rng.choice(len(lst), size=n, replace=False)
        return [lst[i] for i in sorted(idx)]

    _cov  = _CFG.get("ti", {}).get("ti_coverage_rates", {})
    _cov_ip   = _cov.get("ip",   {})
    _cov_hash = _cov.get("hash", {})

    ip_tp_high = select_pct(ip_excl_tp, _cov_ip.get("TP_high_score_pct", 0.30), rng)
    ip_tp_low  = select_pct([ip for ip in ip_excl_tp if ip not in set(ip_tp_high)],
                             _cov_ip.get("TP_low_score_pct",  0.25), rng)
    ip_bp_high = select_pct(ip_excl_bp, _cov_ip.get("BP_high_score_pct", 0.02), rng)
    ip_bp_low  = select_pct([ip for ip in ip_excl_bp if ip not in set(ip_bp_high)],
                             _cov_ip.get("BP_low_score_pct",  0.01), rng)
    ip_fp_high = select_pct(ip_excl_fp, _cov_ip.get("FP_high_score_pct", 0.02), rng)
    ip_fp_low  = select_pct([ip for ip in ip_excl_fp if ip not in set(ip_fp_high)],
                             _cov_ip.get("FP_low_score_pct",  0.01), rng)

    hash_tp_high = select_pct(hash_excl_tp, _cov_hash.get("TP_high_score_pct", 0.30), rng)
    hash_tp_low  = select_pct([h for h in hash_excl_tp if h not in set(hash_tp_high)],
                               _cov_hash.get("TP_low_score_pct",  0.25), rng)
    hash_bp_high = select_pct(hash_excl_bp, _cov_hash.get("BP_high_score_pct", 0.02), rng)
    hash_bp_low  = select_pct([h for h in hash_excl_bp if h not in set(hash_bp_high)],
                               _cov_hash.get("BP_low_score_pct",  0.01), rng)
    hash_fp_high = select_pct(hash_excl_fp, _cov_hash.get("FP_high_score_pct", 0.02), rng)
    hash_fp_low  = select_pct([h for h in hash_excl_fp if h not in set(hash_fp_high)],
                               _cov_hash.get("FP_low_score_pct",  0.01), rng)

    # ── Scores par catégorie — chargés depuis ti_config.csv ──────────────
    def _build_score_params(cfg_section: dict) -> dict:
        """Convertit {cat: {mean,std,min,max}} → {cat: (mean,std,min,max)}."""
        return {
            cat: (v["mean"], v["std"], v["min"], v["max"])
            for cat, v in cfg_section.items()
        }

    _sp_cfg   = _CFG.get("ti", {}).get("ti_score_params", {})
    SCORE_PARAMS      = _build_score_params(_sp_cfg.get("ip",   {
        "tp_high": {"mean":80,"std":8,"min":60,"max":100},
        "tp_low":  {"mean":47,"std":8,"min":30,"max":60},
        "bp_high": {"mean":70,"std":8,"min":52,"max":85},
        "bp_low":  {"mean":32,"std":8,"min":15,"max":48},
        "fp_high": {"mean":70,"std":8,"min":52,"max":85},
        "fp_low":  {"mean":32,"std":8,"min":15,"max":48},
    }))
    SCORE_PARAMS_HASH = _build_score_params(_sp_cfg.get("hash", {
        "tp_high": {"mean":82,"std":8,"min":62,"max":100},
        "tp_low":  {"mean":48,"std":8,"min":30,"max":62},
        "bp_high": {"mean":70,"std":8,"min":52,"max":85},
        "bp_low":  {"mean":32,"std":8,"min":15,"max":48},
        "fp_high": {"mean":70,"std":8,"min":52,"max":85},
        "fp_low":  {"mean":32,"std":8,"min":15,"max":48},
    }))

    def _score(params, category):
        mean, std, lo, hi = params[category]
        return int(np.clip(rng.normal(mean, std), lo, hi))

    _actor_cfg = _CFG.get("ti", {}).get("actor_type_by_grade", {
        "TruePositive":   {"APT":0.20,"cybercrime":0.55,"hacktivist":0.10,"unknown":0.15},
        "BenignPositive": {"APT":0.02,"cybercrime":0.05,"hacktivist":0.02,"unknown":0.91},
        "FalsePositive":  {"APT":0.01,"cybercrime":0.03,"hacktivist":0.01,"unknown":0.95},
    })
    _fp_actor = _actor_cfg.get("FalsePositive", {"unknown": 1.0})

    def _actor_type(grade: str, rng) -> str:
        probs = _actor_cfg.get(grade, _fp_actor)
        return rng.choice(list(probs.keys()), p=list(probs.values()))

    def _grade_from_category(cat: str) -> str:
        if "tp" in cat:
            return "TruePositive"
        if "bp" in cat:
            return "BenignPositive"
        return "FalsePositive"

    def add_ip_record(ip, category):
        score = _score(SCORE_PARAMS, category)
        grade = _grade_from_category(category)
        actor = _actor_type(grade, rng)
        nb_campaigns = (
            int(rng.integers(1, 6)) if (score > 60 and actor in ("APT", "cybercrime"))
            else (1 if score > 40 else 0)
        )
        records.append({
            "ioc_value":             ip,
            "ioc_type":              "ip",
            "ti_score":              score,
            "ti_reputation":         ("malicious"  if score > 70 else
                                      "suspicious" if score > 40 else "clean"),
            "in_blocklist":          int(score > 60),
            "nb_sources_flagging":   max(0, int(score / 20) + int(rng.integers(-1, 2))),
            "ti_confidence":         round(float(rng.uniform(0.4, 0.95)), 2),
            "ti_first_seen_days_ago":int(rng.integers(0, 3650)),
            "ti_last_seen_days_ago": int(rng.integers(0, 180)) if score > 40 else int(rng.integers(60, 730)),
            "ti_trend":              rng.choice(
                                         _clean_dist(_CFG.get("ti",{}).get("ti_trend_distribution",{})).get("values",["stable"]),
                                         p=_clean_dist(_CFG.get("ti",{}).get("ti_trend_distribution",{})).get("probabilities",[1.0]),
                                     ),
            "threat_categories":     (
                                         rng.choice(["Malware-C2", "Botnet",
                                             "Phishing", "Ransomware", "Scanner"])
                                         if score > 60 else ""
                                     ),
            "geo_country":           rng.choice(
                                         _clean_dist(_CFG.get("ti",{}).get("geo_country_distribution",{})).get("countries",["US"]),
                                         p=_clean_dist(_CFG.get("ti",{}).get("geo_country_distribution",{})).get("probabilities",[1.0]),
                                     ),
            "actor_type":            actor,
            "nb_campaigns":          nb_campaigns,
        })

    def add_hash_record(sha, category):
        score = _score(SCORE_PARAMS_HASH, category)
        grade = _grade_from_category(category)
        actor = _actor_type(grade, rng)
        nb_campaigns = (
            int(rng.integers(1, 4)) if (score > 60 and actor in ("APT", "cybercrime"))
            else 0
        )
        vt = int(score * 72 / 100)
        hash_scores[sha] = score
        records.append({
            "ioc_value":             sha,
            "ioc_type":              "sha256",
            "ti_score":              score,
            "ti_reputation":         ("malicious"  if score > 70 else
                                      "suspicious" if score > 40 else "clean"),
            "in_blocklist":          int(vt > 15),
            "nb_sources_flagging":   vt,
            "ti_confidence":         round(float(rng.uniform(0.5, 0.98)), 2),
            "ti_first_seen_days_ago":int(rng.integers(0, 365)),
            "ti_last_seen_days_ago": int(rng.integers(0, 90)) if score > 40 else int(rng.integers(30, 365)),
            "ti_trend":              rng.choice(
                                         _clean_dist(_CFG.get("ti",{}).get("ti_trend_distribution",{})).get("values",["stable"]),
                                         p=_clean_dist(_CFG.get("ti",{}).get("ti_trend_distribution",{})).get("probabilities",[1.0]),
                                     ),
            "threat_categories":     (
                                         rng.choice(["Trojan", "Ransomware",
                                             "Backdoor", "Dropper"])
                                         if vt > 15 else ""
                                     ),
            "geo_country":           "",
            "actor_type":            actor,
            "nb_campaigns":          nb_campaigns,
        })

    for ip in ip_tp_high:  add_ip_record(ip, "tp_high")
    for ip in ip_tp_low:   add_ip_record(ip, "tp_low")
    for ip in ip_bp_high:  add_ip_record(ip, "bp_high")
    for ip in ip_bp_low:   add_ip_record(ip, "bp_low")
    for ip in ip_fp_high:  add_ip_record(ip, "fp_high")
    for ip in ip_fp_low:   add_ip_record(ip, "fp_low")

    for sha in hash_tp_high:  add_hash_record(sha, "tp_high")
    for sha in hash_tp_low:   add_hash_record(sha, "tp_low")
    for sha in hash_bp_high:  add_hash_record(sha, "bp_high")
    for sha in hash_bp_low:   add_hash_record(sha, "bp_low")
    for sha in hash_fp_high:  add_hash_record(sha, "fp_high")
    for sha in hash_fp_low:   add_hash_record(sha, "fp_low")

    ti_df = pd.DataFrame(records)
    ti_df.to_csv(output_dir / "TI_database.csv", index=False)

    n_ip_ti    = (ti_df.ioc_type == "ip").sum()
    n_hash_ti  = (ti_df.ioc_type == "sha256").sum()
    n_ip_total = len(ip_grade_sets)
    n_hash_total = len(hash_grade_sets)
    print(f"  TI_database.csv : {len(ti_df):,} IOCs")
    print(f"    IPs  : {n_ip_ti}/{n_ip_total} ({n_ip_ti/max(1,n_ip_total):.1%})")
    print(f"    Hash : {n_hash_ti}/{n_hash_total} ({n_hash_ti/max(1,n_hash_total):.1%})")

    merged = incidents[["sample_ip", "IncidentGrade"]].dropna(subset=["sample_ip"]).copy()
    merged = merged.merge(
        ti_df[ti_df["ioc_type"] == "ip"][["ioc_value", "ti_score"]],
        left_on="sample_ip", right_on="ioc_value", how="left",
    )
    print("  Validation TI IP par grade (score=0 si absent) :")
    for g, grp in merged.groupby("IncidentGrade"):
        cov        = grp["ti_score"].notna().mean()
        mean_score = grp["ti_score"].fillna(0).mean()
        print(f"    {g}: couverture={cov:.1%}, score moyen={mean_score:.1f}")

    return hash_scores


# ── CRITICALITY RATIOS ────────────────────────────────────────────────────

def _compute_criticality_ratios(incidents: pd.DataFrame, output_dir: Path) -> dict:
    ratio_path = output_dir / "crit_ratios.csv"
    if ratio_path.exists():
        return pd.read_csv(ratio_path, index_col=0)["ratio"].to_dict()

    _cmdb_cfg   = _CFG.get("cmdb", {})
    _ratio_cfg  = _cmdb_cfg.get("asset_criticality_ratios",
                                {"CRITICAL": 0.10, "HIGH": 0.20,
                                 "MEDIUM": 0.40, "LOW": 0.30})
    crit_ratios = {k: v for k, v in _ratio_cfg.items() if not k.startswith("_")}
    pd.DataFrame({
        "criticality": list(crit_ratios.keys()),
        "ratio":       list(crit_ratios.values()),
    }).set_index("criticality").to_csv(ratio_path)
    return crit_ratios


# ── NOMS SYNTHÉTIQUES (cohérents avec la criticité) ───────────────────────

# Correspondance criticité → préfixes et types
_DEVICE_NAMES = {
    "CRITICAL": {
        "prefixes":    ["PROD", "CORE", "VAULT"],
        "type_codes":  ["DC", "AD", "EXCH", "SQL"],
    },
    "HIGH": {
        "prefixes":    ["TRADE", "BANK", "CORE"],
        "type_codes":  ["SRV", "APP", "SQL", "WEB"],
    },
    "MEDIUM": {
        "prefixes":    ["OFFICE", "ADMIN", "DEPT"],
        "type_codes":  ["WS", "SRV", "PC"],
    },
    "LOW": {
        "prefixes":    ["TEST", "DEV", "LAB", "DEMO"],
        "type_codes":  ["VM", "WS", "TEST", "GUEST"],
    },
}

_USER_NAMES = {
    "CRITICAL": {
        "roles": ["ceo", "cto", "cfo", "head_security"],
        "depts": ["exec", "board"],
    },
    "HIGH": {
        "roles": ["vp", "director", "manager", "lead"],
        "depts": ["trading", "risk", "tech", "security"],
    },
    "MEDIUM": {
        "roles": ["analyst", "engineer", "specialist", "admin"],
        "depts": ["risk", "ops", "infra", "security"],
    },
    "LOW": {
        "roles": ["user", "temp", "guest", "intern"],
        "depts": ["admin", "ops", "support"],
    },
}


def _make_synthetic_device_name(numeric_id: float, crit_level: str) -> str:
    """Nom de device déterministe, cohérent avec asset_criticality."""
    seed       = int(hashlib.md5(str(numeric_id).encode()).hexdigest(), 16) % 10000
    cfg        = _DEVICE_NAMES.get(crit_level, _DEVICE_NAMES["LOW"])
    prefix     = cfg["prefixes"][seed % len(cfg["prefixes"])]
    type_code  = cfg["type_codes"][(seed // len(cfg["prefixes"])) % len(cfg["type_codes"])]
    number     = str(seed % 9999).zfill(3)
    return f"{prefix}-{type_code}-{number}"


def _make_synthetic_user_name(numeric_id: float, crit_level: str) -> str:
    """Login déterministe, cohérent avec user_criticality."""
    seed   = int(hashlib.md5(str(numeric_id).encode()).hexdigest(), 16) % 10000
    cfg    = _USER_NAMES.get(crit_level, _USER_NAMES["LOW"])
    role   = cfg["roles"][seed % len(cfg["roles"])]
    dept   = cfg["depts"][(seed // len(cfg["roles"])) % len(cfg["depts"])]
    number = str(seed % 999).zfill(2)
    return f"{role}.{dept}.{number}"


def _assign_criticality(
    entity_id,
    incident_grade: str,
    crit_ratios: dict,
    rng_seed: int,
    grade_weight: float = 0.60,
) -> str:
    """
    Assigne une criticité corrélée au grade de l'incident.

    60% du signal vient du grade (TP → criticité plus élevée en moyenne),
    40% de bruit pour préserver l'overlap naturel et éviter le data leakage.

    Distribution cible par grade :
      TruePositive   : CRITICAL 25%, HIGH 35%, MEDIUM 30%, LOW 10%
      BenignPositive : CRITICAL 10%, HIGH 20%, MEDIUM 40%, LOW 30%  (= ratios SOC standard)
      FalsePositive  : CRITICAL  5%, HIGH 15%, MEDIUM 40%, LOW 40%  (assets peu critiques)
    """
    _gcc = _CFG.get("cmdb", {}).get("grade_criticality_correlation", {
        "TruePositive":   {"CRITICAL":0.25,"HIGH":0.35,"MEDIUM":0.30,"LOW":0.10},
        "BenignPositive": {"CRITICAL":0.10,"HIGH":0.20,"MEDIUM":0.40,"LOW":0.30},
        "FalsePositive":  {"CRITICAL":0.05,"HIGH":0.15,"MEDIUM":0.40,"LOW":0.40},
    })
    # Exclure les clés de métadonnées (_description, _source…)
    GRADE_CRIT_PROBS = {k: v for k, v in _gcc.items() if not k.startswith("_")}
    _bp_default = GRADE_CRIT_PROBS.get(
        "BenignPositive",
        {"CRITICAL":0.10,"HIGH":0.20,"MEDIUM":0.40,"LOW":0.30}
    )
    grade_probs = GRADE_CRIT_PROBS.get(incident_grade, _bp_default)

    # Bruit : tirage selon ratios SOC standards
    rand_val = int(hashlib.md5(str(entity_id).encode()).hexdigest(), 16) % 1000 / 1000
    if rand_val < crit_ratios["CRITICAL"]:
        noise_crit = "CRITICAL"
    elif rand_val < crit_ratios["CRITICAL"] + crit_ratios["HIGH"]:
        noise_crit = "HIGH"
    elif rand_val < crit_ratios["CRITICAL"] + crit_ratios["HIGH"] + crit_ratios["MEDIUM"]:
        noise_crit = "MEDIUM"
    else:
        noise_crit = "LOW"

    # Mix grade (60%) + bruit (40%) — tirage reproductible par entity_id
    rng_local = np.random.default_rng(rng_seed)
    if rng_local.random() < grade_weight:
        # Tirage pondéré selon le grade de l'incident
        crits  = list(grade_probs.keys())
        probs  = list(grade_probs.values())
        return rng_local.choice(crits, p=probs)
    else:
        return noise_crit


# ── CMDB ──────────────────────────────────────────────────────────────────

def _generate_cmdb(incidents: pd.DataFrame, output_dir: Path, crit_ratios=None):
    """
    Génère CMDB.csv avec :
    - Criticité corrélée au grade de l'incident (60% grade / 40% bruit)
    - Noms assets cohérents avec la criticité
    - Champs enrichis pour XAI : days_since_last_patch, nb_vulnerabilities,
      is_domain_controller, network_zone, last_incident_days_ago (device)
      + MFA_enabled, nb_failed_logins_7d, login_country_mismatch (user)
    """
    if crit_ratios is None:
        crit_ratios = _compute_criticality_ratios(incidents, output_dir)

    incidents = incidents.copy()
    if "sample_device_id" not in incidents.columns and "sample_device" in incidents.columns:
        incidents["sample_device_id"] = incidents["sample_device"]
    if "sample_device" not in incidents.columns and "sample_device_id" in incidents.columns:
        incidents["sample_device"] = incidents["sample_device_id"]

    records = []
    seen    = set()
    rng     = np.random.default_rng(RANDOM_SEED)

    # ── Devices ──────────────────────────────────────────────────────────
    # Joindre le grade de l'incident pour la corrélation criticité
    device_grade = (
        incidents[["sample_device_id", "IncidentGrade"]]
        .dropna(subset=["sample_device_id"])
        .drop_duplicates("sample_device_id")
        .set_index("sample_device_id")["IncidentGrade"]
        .to_dict()
    )

    for _, row in (
        incidents[["sample_device_id", "sample_device"]]
        .dropna(subset=["sample_device_id"])
        .drop_duplicates("sample_device_id")
        .iterrows()
    ):
        device_id = row["sample_device_id"]
        if device_id in seen or str(device_id).startswith("NET-"):
            continue
        seen.add(device_id)

        grade = device_grade.get(device_id, "BenignPositive")
        # Seed déterministe unique par device_id
        device_seed = int(hashlib.md5(str(device_id).encode()).hexdigest(), 16) % (2**31)
        crit = _assign_criticality(device_id, grade, crit_ratios, device_seed)

        device_name = _make_synthetic_device_name(device_id, crit)

        _dc = _CFG.get("cmdb", {})
        base_sensitivity = _dc.get("device_sensitivity_base",
                                   {"CRITICAL":90,"HIGH":70,"MEDIUM":40,"LOW":15})[crit]

        # days_since_last_patch — corrélé à la criticité inverse
        _pl_cfg    = _dc.get("device_patch_lag_days", {}).get(crit,
                             {"mean":60,"std_ratio":0.40,"max_days":730})
        patch_mean = _pl_cfg["mean"]
        days_patch = int(np.clip(
            rng.normal(patch_mean, patch_mean * _pl_cfg.get("std_ratio", 0.40)),
            0, _pl_cfg.get("max_days", 730)
        ))

        # nb_vulnerabilities — corrélé au patch lag
        vuln_base = _dc.get("device_vuln_base_count",
                            {"CRITICAL":2,"HIGH":5,"MEDIUM":10,"LOW":20})[crit]
        nb_vulns  = max(0, int(rng.normal(vuln_base + days_patch // 30, 3)))

        # is_domain_controller
        _dc_prob = _dc.get("device_dc_probability", {}).get("CRITICAL", 0.20)
        is_dc    = int(crit == "CRITICAL" and rng.random() < _dc_prob)

        # network_zone
        _zone_cfg  = _dc.get("device_network_zone_probs", {
            "CRITICAL":{"ISOLATED":0.50,"CORP":0.45,"DMZ":0.05},
            "HIGH":    {"ISOLATED":0.20,"CORP":0.65,"DMZ":0.15},
            "MEDIUM":  {"ISOLATED":0.05,"CORP":0.75,"DMZ":0.20},
            "LOW":     {"ISOLATED":0.05,"CORP":0.60,"DMZ":0.35},
        })
        zone_probs = _zone_cfg.get(crit, {"CORP": 1.0})
        network_zone = rng.choice(
            list(zone_probs.keys()), p=list(zone_probs.values())
        )

        # last_incident_days_ago — récidive plus probable sur assets critiques ciblés
        _lid = _CFG.get("cmdb", {}).get("device_last_incident_days", {
            "TruePositive":   {"min":1,  "max":90},
            "BenignPositive": {"min":30, "max":365},
            "FalsePositive":  {"min":90, "max":730},
        }).get(grade, {"min": 30, "max": 365})
        last_inc = int(rng.integers(_lid["min"], _lid["max"]))

        # is_internet_exposed — corrélé à network_zone
        _corp_exp = _CFG.get("cmdb", {}).get(
            "device_internet_exposed_prob_corp", {}
        ).get("probability", 0.10)
        is_exposed = int(network_zone == "DMZ" or
                         (network_zone == "CORP" and rng.random() < _corp_exp))

        records.append({
            "device_id":              device_id,
            "device_name":            device_name,
            "asset_criticality":      crit,
            "sensitivity_score":      base_sensitivity + int(rng.integers(-10, 10)),
            "asset_type":             _choice_from_cfg("cmdb","device_asset_type_probs",
                                               {"Workstation":0.4,"Server":0.3,"Domain Controller":0.05,"Laptop":0.25},rng),
            "business_unit":          _choice_from_cfg("cmdb","device_business_unit_probs",
                                               {"Finance":0.3,"IT":0.3,"Trading":0.2,"Risk":0.2},rng),
            "os_family":              _choice_from_cfg("cmdb","device_os_family_probs",
                                               {"Windows":0.7,"Linux":0.2,"MacOS":0.1},rng),
            "is_internet_exposed":    is_exposed,
            "patch_level":            (
                                          "Up-to-date" if days_patch < 30 else
                                          "Minor-lag"  if days_patch < 90 else
                                          "Critical-lag"
                                      ),
            "days_since_last_patch":  days_patch,
            "nb_vulnerabilities":     nb_vulns,
            "is_domain_controller":   is_dc,
            "network_zone":           network_zone,
            "last_incident_days_ago": last_inc,
            "owner_team":             rng.choice(["IT-OPS", "SOC", "DevOps", "Business"]),
        })

    # ── Utilisateurs ──────────────────────────────────────────────────────
    user_seen = set()
    user_grade = (
        incidents[["sample_account", "IncidentGrade"]]
        .dropna(subset=["sample_account"])
        .drop_duplicates("sample_account")
        .set_index("sample_account")["IncidentGrade"]
        .to_dict()
    )

    for account_id in incidents["sample_account"].dropna().drop_duplicates():
        if account_id in user_seen:
            continue
        user_seen.add(account_id)

        grade     = user_grade.get(account_id, "BenignPositive")
        user_seed = int(hashlib.md5(str(account_id).encode()).hexdigest(), 16) % (2**31)
        crit      = _assign_criticality(account_id, grade, crit_ratios, user_seed)

        user_name        = _make_synthetic_user_name(account_id, crit)
        _uc = _CFG.get("cmdb", {})
        base_sensitivity = _uc.get("user_sensitivity_base",
                                   {"CRITICAL":95,"HIGH":75,"MEDIUM":45,"LOW":20})[crit]

        # MFA
        mfa_prob    = _uc.get("user_mfa_probability",
                              {"CRITICAL":0.90,"HIGH":0.75,"MEDIUM":0.55,"LOW":0.35})[crit]
        mfa_enabled = int(rng.random() < mfa_prob)

        # Tentatives de login échouées
        _fl_cfg   = _uc.get("user_failed_logins_by_grade", {}).get(
            grade, {"mean": 2, "std": 2, "min": 0, "max": 10}
        )
        nb_failed = int(np.clip(
            rng.normal(_fl_cfg["mean"], _fl_cfg["std"]),
            _fl_cfg["min"], _fl_cfg["max"]
        ))

        # login_country_mismatch
        mismatch_prob  = _uc.get("user_login_country_mismatch_prob",
                                 {"TruePositive":0.35,"BenignPositive":0.08,"FalsePositive":0.03})
        login_mismatch = int(rng.random() < mismatch_prob.get(grade, 0.05))

        records.append({
            "account_id":               account_id,
            "user_name":                user_name,
            "user_criticality":         crit,
            "sensitivity_score":        base_sensitivity + int(rng.integers(-5, 5)),
            "user_role":                _choice_from_cfg("cmdb","user_role_probs",
                                                 {"Exec":0.05,"Manager":0.15,"Analyst":0.4,"User":0.4},rng),
            "department":               _choice_from_cfg("cmdb","user_department_probs",
                                                 {"Finance":0.25,"IT":0.25,"HR":0.15,"Operations":0.2,"Legal":0.15},rng),
            "is_privileged":            int(crit in ["CRITICAL", "HIGH"]),
            "last_login_days":          int(rng.integers(0, 30)),
            "account_status":           _choice_from_cfg("cmdb","user_account_status_probs",
                                                 {"Active":0.9,"Inactive":0.05,"Suspended":0.05},rng),
            "MFA_enabled":              mfa_enabled,
            "nb_failed_logins_7d":      nb_failed,
            "login_country_mismatch":   login_mismatch,
        })

    pd.DataFrame(records).to_csv(output_dir / "CMDB.csv", index=False)

    n_devices = sum(1 for r in records if "device_name" in r)
    n_users   = sum(1 for r in records if "user_name"   in r)
    print(f"  CMDB.csv : {n_devices:,} assets, {n_users:,} users")

    # Validation : distribution criticité par grade
    df_val = pd.DataFrame(records)
    if "asset_criticality" in df_val.columns:
        # Joindre le grade via device_grade
        df_dev = df_val[df_val["asset_criticality"].notna()].copy()
        df_dev["grade"] = df_dev["device_id"].map(device_grade).fillna("unknown")
        print("  Validation CMDB — criticité device par grade :")
        for g in ["TruePositive", "BenignPositive", "FalsePositive"]:
            sub = df_dev[df_dev["grade"] == g]
            if len(sub):
                dist = sub["asset_criticality"].value_counts(normalize=True).to_dict()
                print(f"    {g}: " + ", ".join(f"{k}={v:.0%}" for k, v in dist.items()))


# ── SANDBOX ───────────────────────────────────────────────────────────────

def _generate_sandbox(incidents, output_dir, ti_stats, global_stats, hash_scores):
    """
    Génère Sandbox.csv avec comportements corrélés au grade.

    Nouveaux champs v1 :
    - evasion_detected          : anti-debug / VM-aware (T1497)
    - persistence_mechanism     : clé de registre / tâche planifiée (T1547)
    - lateral_movement_attempt  : tentative de déplacement latéral (T1021)
    - privilege_escalation      : élévation de privilège (T1068)
    - dns_requests_count        : nb requêtes DNS (C2 via DNS — T1071.004)
    """
    rng     = np.random.default_rng(RANDOM_SEED)
    records = []

    # Probabilités comportements — chargées depuis sandbox_config.csv
    _sb_cfg       = _CFG.get("sandbox", {})
    _bhv_raw      = _sb_cfg.get("behavior_probabilities_by_grade", {})
    # Exclure les clés de métadonnées (_description, _source…)
    BEHAVIOR_PROBS = {k: v for k, v in _bhv_raw.items() if not k.startswith("_")}
    if not BEHAVIOR_PROBS:
        BEHAVIOR_PROBS = {
            "TruePositive":   {"evasion":0.45,"persistence":0.60,"lateral":0.35,"privesc":0.40},
            "BenignPositive": {"evasion":0.05,"persistence":0.10,"lateral":0.03,"privesc":0.05},
            "FalsePositive":  {"evasion":0.02,"persistence":0.03,"lateral":0.01,"privesc":0.02},
        }

    for _, row in (
        incidents[["sample_sha256", "IncidentGrade"]]
        .dropna(subset=["sample_sha256"])
        .drop_duplicates("sample_sha256")
        .iterrows()
    ):
        sha   = row["sample_sha256"]
        grade = row["IncidentGrade"]

        score = hash_scores.get(
            sha,
            _ti_score_from_grade(grade, ti_stats, global_stats),
        )

        if score > 40:
            verdict = "Malicious"
        elif score > 20:
            verdict = "Suspicious"
        else:
            verdict = "Clean"

        bprobs = BEHAVIOR_PROBS.get(grade, BEHAVIOR_PROBS["FalsePositive"])
        is_tp  = (grade == "TruePositive")

        evasion     = int(rng.random() < bprobs["evasion"])
        persistence = int(rng.random() < bprobs["persistence"])
        lateral     = int(rng.random() < bprobs["lateral"])
        privesc     = int(rng.random() < bprobs["privesc"])

        # DNS : C2 via DNS génère beaucoup de requêtes inhabituelles
        _dns_cfg = _sb_cfg.get("dns_requests_count_by_verdict", {
            "Malicious":  {"min": 50,  "max": 500},
            "Suspicious": {"min": 5,   "max": 80},
            "Clean":      {"min": 0,   "max": 20},
        })
        if is_tp and verdict == "Malicious":
            _d = _dns_cfg.get("Malicious", {"min":50,"max":500})
        elif verdict == "Suspicious":
            _d = _dns_cfg.get("Suspicious", {"min":5,"max":80})
        else:
            _d = _dns_cfg.get("Clean", {"min":0,"max":20})
        dns_count = int(rng.integers(_d["min"], _d["max"]))

        records.append({
            "sha256":                  sha,
            "sandbox_verdict":         verdict,
            "malware_score":           score,
            "network_connections":     (
                                           int(rng.integers(5, 30))
                                           if verdict != "Clean"
                                           else int(rng.integers(0, 3))
                                       ),
            "dropped_files":           (
                                           int(rng.integers(1, 6))
                                           if verdict == "Malicious" else 0
                                       ),
            "registry_modifications":  (
                                           int(rng.integers(1, 15))
                                           if verdict != "Clean" else 0
                                       ),
            "process_injections":      int(
                                           verdict == "Malicious"
                                           and rng.random() < _sb_cfg.get("process_injection_prob",{}).get("Malicious", 0.50)
                                       ),
            "c2_beaconing":            int(
                                           verdict == "Malicious"
                                           and rng.random() < _sb_cfg.get("c2_beaconing_prob",{}).get("Malicious", 0.60)
                                       ),
            "evasion_detected":        evasion,
            "persistence_mechanism":   persistence,
            "lateral_movement_attempt":lateral,
            "privilege_escalation":    privesc,
            "dns_requests_count":      dns_count,
            "analysis_duration_s":     int(rng.integers(
                                           _sb_cfg.get("analysis_duration_seconds",{}).get("min", 60),
                                           _sb_cfg.get("analysis_duration_seconds",{}).get("max", 300),
                                       )),
        })

    pd.DataFrame(records).to_csv(output_dir / "Sandbox.csv", index=False)
    print(f"  Sandbox.csv : {len(records):,} fichiers analysés")

    # Validation : taux comportements malveillants par grade
    df_val = pd.DataFrame(records)
    df_val["grade"] = (
        incidents[["sample_sha256", "IncidentGrade"]]
        .dropna(subset=["sample_sha256"])
        .drop_duplicates("sample_sha256")
        .set_index("sample_sha256")["IncidentGrade"]
        .reindex(df_val["sha256"])
        .values
    )
    print("  Validation Sandbox — comportements par grade :")
    for g in ["TruePositive", "BenignPositive", "FalsePositive"]:
        sub = df_val[df_val["grade"] == g]
        if len(sub):
            cols = ["evasion_detected", "persistence_mechanism",
                    "lateral_movement_attempt", "privilege_escalation"]
            rates = {c: sub[c].mean() for c in cols if c in sub.columns}
            print(f"    {g}: " + ", ".join(f"{k}={v:.0%}" for k, v in rates.items()))


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

    # Log colonnes exportées
    action_cols = [c for c in incidents.columns
                   if c in ("action_grouped", "top_action_granular", "nb_action_types")]
    if action_cols:
        print(f"  Colonnes actions SOC (affichage uniquement, exclure du ML) : {action_cols}")


# =========================================================
# MAIN
# =========================================================

def main():
    global _CFG

    args       = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print(" POC SOC Triage — Extraction + Simulation  v2")
    print("=" * 60)

    # Chargement de la configuration — doit être fait avant tout le reste
    print(f"\n[0/5] Chargement configuration depuis {args.config}/")
    _CFG = load_config(Path(args.config))

    if args.load_existing:
        incidents_path = output_dir / "incidents_dataset.csv"
        if not incidents_path.exists():
            print(f"\n  Fichier introuvable : {incidents_path}")
            print("  Lance d'abord sans --load-existing pour créer le fichier.")
            return
        print(f"\n[1/5] Chargement depuis {incidents_path}")
        incidents = pd.read_csv(incidents_path)
        # Compatibilité anciens exports
        if "sample_device_id" not in incidents.columns and "sample_device" in incidents.columns:
            incidents["sample_device_id"] = incidents["sample_device"]
        if "sample_device" not in incidents.columns and "sample_device_id" in incidents.columns:
            incidents["sample_device"] = incidents["sample_device_id"]
        print(f"  {len(incidents):,} incidents chargés "
              f"({incidents.shape[1]} colonnes)")
    else:
        if not args.input:
            print("\n  --input requis si --load-existing n'est pas utilisé")
            return
        if not Path(args.input).exists():
            print(f"\n  Fichier introuvable : {args.input}")
            print("  Lance d'abord :")
            print("  kaggle datasets download -d Microsoft/microsoft-security-incident-prediction")
            print("  unzip microsoft-security-incident-prediction.zip")
            return

        import time as _time
        t_total = _time.perf_counter()

        # ── Conversion Parquet optionnelle (--to-parquet) ───────────────
        effective_input = args.input
        if args.to_parquet:
            print(f"\n[0.5/5] Conversion CSV → Parquet...")
            effective_input = convert_to_parquet(args.input, output_dir)

        # ── Passe 1 : identification des incidents (3 colonnes) ──────────
        incidents_index, input_path_ref, total_rows = load_and_aggregate(effective_input)
        crit_ratios = _compute_criticality_ratios(incidents_index, output_dir)

        # ── Stratification sur l'index complet ───────────────────────────
        t_strat = _time.perf_counter()
        selected_ids = stratified_sampling(incidents_index, args.n_incidents)
        print(f"  Stratification : {_time.perf_counter()-t_strat:.2f}s")

        if args.no_expand:
            print(f"\n[3/5] Expansion désactivée (--no-expand)")
        else:
            # ── Passe 1.5 : index pivot pour expansion (7 colonnes) ──────
            # Beaucoup plus léger que la passe 2 complète (34 colonnes)
            # mais couvre tous les incidents → expansion fidèle à l'originale
            print(f"\n[2.5/5] Passe 1.5 — index pivot pour expansion...")
            expansion_index = build_expansion_index(effective_input)

            # ── Expansion sur l'index pivot complet ──────────────────────
            selected_ids_expanded = expand_with_related_incidents(
                selected_ids, expansion_index
            )
            new_ids = selected_ids_expanded - selected_ids
            if new_ids:
                print(f"  +{len(new_ids):,} incidents ajoutés par expansion")
            selected_ids = selected_ids_expanded

        # ── Passe 2 : agrégation complète sur les IDs finaux ─────────────
        print(f"\n  Passe 2 — agrégation des {len(selected_ids):,} incidents sélectionnés "
              f"(34 colonnes)...")
        incidents = aggregate_selected(effective_input, selected_ids)
        print(f"  Total passes 1+1.5+2 : {_time.perf_counter()-t_total:.1f}s")

    # 4 — Données simulées
    import time as _time
    t_sim = _time.perf_counter()
    generate_simulated_data(
        incidents,
        output_dir,
        crit_ratios if not args.load_existing else None,
    )
    print(f"  Simulation TI/CMDB/Sandbox : {_time.perf_counter()-t_sim:.1f}s")

    # 5 — Sauvegarde
    t_save = _time.perf_counter()
    save(incidents, output_dir)
    print(f"  Sauvegarde : {_time.perf_counter()-t_save:.1f}s")

    print("\n" + "=" * 60)
    print("Terminé. Fichiers dans", output_dir)
    print("  incidents_dataset.csv  — dataset principal (56 colonnes)")
    print("  TI_database.csv        — réputation IOC")
    print("  CMDB.csv               — assets + users")
    print("  Sandbox.csv            — analyse comportementale")
    print()
    print("  Fichiers config utilisés (config/) :")
    print("    eol_os.csv, ti_config.csv, cmdb_config.csv, sandbox_config.csv")
    print()
    print("  Colonnes à EXCLURE du ML dans le 02_ :")
    print("    action_grouped, top_action_granular, nb_action_types")
    print("    (actions SOC décidées après connaissance du grade → data leakage)")
    print("\nProchaine étape : 02_feature_engineering.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()