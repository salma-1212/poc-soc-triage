#!/usr/bin/env bash
# ============================================================
# POC SOC Triage — Setup complet
# Usage : bash setup.sh
# Prérequis : Python 3.10+, git installés
# ============================================================

set -e  # stop on first error

PROJECT_NAME="poc-soc-triage"
PYTHON_MIN="3.10"

echo "=================================================="
echo " POC SOC Triage — Setup"
echo "=================================================="

# ── Vérification Python ───────────────────────────────────
echo ""
echo "[1/5] Vérification de Python..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python détecté : $PYTHON_VERSION"
# Vérification version minimale (simple)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    echo "  ⚠️  Python 3.10+ requis. Installe pyenv ou mise à jour Python."
    exit 1
fi
echo "  ✓ Version OK"

# ── Environnement virtuel ─────────────────────────────────
echo ""
echo "[2/5] Création de l'environnement virtuel (.venv)..."
if [ -d ".venv" ]; then
    echo "  .venv existe déjà — skip création"
else
    python3 -m venv .venv
    echo "  ✓ .venv créé"
fi

# Activation
source .venv/bin/activate
echo "  ✓ .venv activé"

# ── Dépendances système (MacOS) ──────────────────────────
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Installation dépendance OpenMP (libomp)..."
  brew install libomp >/dev/null 2>&1 || true
fi

# ── Installation des dépendances ──────────────────────────
echo ""
echo "[3/5] Installation des dépendances..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "  ✓ Packages installés"

# Vérification des imports clés
python3 -c "
import pandas, numpy, sklearn, xgboost, shap, lime, streamlit, joblib, pyarrow, seaborn
print(f'  pandas     {pandas.__version__}')
print(f'  numpy      {numpy.__version__}')
print(f'  sklearn    {sklearn.__version__}')
print(f'  xgboost    {xgboost.__version__}')
print(f'  shap       {shap.__version__}')
print(f'  lime       {lime.__version__}')
print(f'  streamlit  {streamlit.__version__}')
print(f'  seaborn    {seaborn.__version__}')
print(f'  pyarrow    {pyarrow.__version__}')
print('  ✓ Tous les imports OK')
"

# ── Création de la structure de dossiers ─────────────────
echo ""
echo "[4/5] Création de la structure de dossiers..."
mkdir -p data models config
echo "  ✓ data/, models/ et config/ créés"

# ── Vérification config/ ──────────────────────────────────
MISSING_CFG=0
for f in eol_os.csv ti_config.csv cmdb_config.csv sandbox_config.csv; do
    if [ ! -f "config/$f" ]; then
        echo "  ⚠️  config/$f manquant"
        MISSING_CFG=1
    fi
done
if [ "$MISSING_CFG" -eq 1 ]; then
    echo "  → Les fichiers config/ sont requis par 01_extract_and_simulate.py"
    echo "    Sans eux, le script s'arrête avec une FileNotFoundError."
fi

# ── Git ───────────────────────────────────────────────────
echo ""
echo "[5/5] Initialisation Git..."
if [ ! -d ".git" ]; then
    git init
    git add .
    git commit -m "feat: init POC SOC Triage

- Script extraction dataset GUIDE
- Notebooks Feature Engineering, ML, XAI
- Demo Streamlit analyste N1
- Données simulées TI/CMDB/Sandbox (air-gap)
- Requirements et environnement virtuel"
    echo "  ✓ Premier commit créé"
else
    echo "  .git existe déjà — skip init"
fi

echo ""
echo "=================================================="
echo " Setup terminé ✓"
echo "=================================================="
echo ""
echo " Prochaines étapes :"
echo ""
echo "  # Activer l'env à chaque session :"
echo "  source .venv/bin/activate"
echo ""
echo "  # Récupérer le dataset GUIDE (au choix) :"
echo "  #  A) Téléchargement manuel depuis kaggle.com :"
echo "  #     microsoft-security-incident-prediction → placer GUIDE_Train.csv dans data/"
echo "  #  B) Via la CLI Kaggle :"
echo "  #     → Copier kaggle.json dans ~/.kaggle/  puis  chmod 600 ~/.kaggle/kaggle.json"
echo "  kaggle datasets download -d Microsoft/microsoft-security-incident-prediction"
echo "  unzip microsoft-security-incident-prediction.zip -d data/"
echo ""
echo "  # Lancer l'extraction (10-15 min) :"
echo "  python 01_extract_and_simulate.py --input data/GUIDE_Train.csv --n_incidents 30000 --no-expand"
echo ""
echo "  # Puis ouvrir les notebooks dans l'ordre :"
echo "  jupyter lab"
echo ""
echo "  # Lancer la démo Streamlit :"
echo "  streamlit run 05_demo_app.py"
echo "=================================================="
