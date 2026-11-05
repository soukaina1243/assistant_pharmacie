"""Configuration centrale (chemins, seuils)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
DEMO = DATA / "demo"
MODELS = ROOT / "models"

MEDS_CSV = Path(os.getenv("MEDS_CSV", PROCESSED / "medications.csv"))
ALIASES_JSON = DATA / "aliases.json"
PHARMACIES_CSV = Path(os.getenv("PHARMACIES_CSV", DEMO / "pharmacies.csv"))
STOCK_CSV = Path(os.getenv("STOCK_CSV", DEMO / "stock.csv"))
INTENT_MODEL = MODELS / "intent.joblib"

# Seuils de rapprochement médicament (0-100)
MED_ACCEPT = float(os.getenv("MED_ACCEPT", 80))        # >= : match accepté
MED_CANDIDATE = float(os.getenv("MED_CANDIDATE", 65))  # entre les deux : à confirmer
MED_EXTRA = float(os.getenv("MED_EXTRA", 90))          # score min d'un 2e médicament dans la même phrase

# Mode d'intention : rules | ml | hybrid
INTENT_MODE = os.getenv("INTENT_MODE", "hybrid")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# Reconnaissance de boîtes (Challenge 2)
BOX_OK = float(os.getenv("BOX_OK", 0.72))        # confiance >= : reconnu
BOX_LOW = float(os.getenv("BOX_LOW", 0.40))      # entre les deux : à confirmer
BOX_STOP = float(os.getenv("BOX_STOP", 0.85))    # preuve de marque suffisante pour arrêter l'OCR
