"""Nettoie le référentiel des médicaments (xlsx/csv) -> data/processed/medications.csv

Usage : python scripts/prepare_data.py [chemin_fichier]
Fonctionne avec le référentiel CNOPS (colonnes CODE, NOM, DCI1, DOSAGE1, ...).
Le mapping de colonnes est tolérant pour pouvoir réutiliser le fichier data.gov.ma.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "data/raw/ref-des-medicaments-cnops-2014.xlsx"
OUT = ROOT / "data/processed/medications.csv"

COLS = {  # nom canonique -> alias possibles (en MAJUSCULES)
    "code": ["CODE", "CODE_CIP", "CIP"],
    "nom": ["NOM", "NOM_COMMERCIAL", "NOM COMMERCIAL", "SPECIALITE"],
    "dci": ["DCI1", "DCI", "SUBSTANCE ACTIVE", "COMPOSITION"],
    "dosage": ["DOSAGE1", "DOSAGE"],
    "unite": ["UNITE_DOSAGE1", "UNITE", "UNITE_DOSAGE"],
    "forme": ["FORME"],
    "presentation": ["PRESENTATION"],
    "ppv": ["PPV", "PRIX_PUBLIC", "PRIX PUBLIC DE VENTE"],
    "ph": ["PH", "PRIX_HOSPITALIER"],
    "prix_br": ["PRIX_BR", "PRIX BASE REMBOURSEMENT"],
    "type": ["PRINCEPS_GENERIQUE", "TYPE"],
    "taux": ["TAUX_REMBOURSEMENT", "TAUX"],
}
UNIT_TOKENS = {"MG", "ML", "G", "UI", "MCG", "UG", "KG", "L", "%", "MUI", "DOSE", "DOSES", "CM", "MM"}


def pick(df, names):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([None] * len(df), index=df.index)


def brand_of(nom: str) -> str:
    """'ALFACEFAL 125 MG/5 ML' -> 'ALFACEFAL' ; 'DOLIPRANE VITAMINE C' reste entier."""
    out = []
    for t in nom.split():
        if re.search(r"\d", t) or t in UNIT_TOKENS or t.startswith("MG/"):
            break
        out.append(t)
    while out and out[-1] in {"/", "-", "ET", "DE", "DU", "+"}:
        out.pop()
    return " ".join(out) if out else nom


def to_float(x):
    try:
        return float(str(x).replace(",", ".").replace("%", "").strip())
    except Exception:
        return None


def dosage_mg(v, unit):
    if v is None or not unit:
        return None
    u = str(unit).upper().strip()
    return {"MG": v, "G": v * 1000, "MCG": v / 1000, "UG": v / 1000}.get(u)


def main(path: Path):
    suffix = path.suffix.lower()
    raw = pd.read_excel(path, dtype=str) if suffix in {".xlsx", ".xls"} else pd.read_csv(path, dtype=str)
    raw.columns = [c.strip().upper() for c in raw.columns]
    df = pd.DataFrame({k: pick(raw, v) for k, v in COLS.items()})
    for c in ["code", "nom", "dci", "unite", "forme", "presentation", "type"]:
        df[c] = df[c].fillna("").astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
    df["dosage"] = df["dosage"].map(to_float)
    for c in ["ppv", "ph", "prix_br", "taux"]:
        df[c] = df[c].map(to_float)
    df["brand"] = df["nom"].map(brand_of)
    df["dosage_mg"] = [dosage_mg(v, u) for v, u in zip(df["dosage"], df["unite"])]
    df = df[df["nom"] != ""].drop_duplicates()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} produits | {df['brand'].nunique()} marques | {df['dci'].nunique()} DCI -> {OUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN)
