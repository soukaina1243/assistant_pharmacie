"""Chargement des données (médicaments, pharmacies, stock)."""
from __future__ import annotations

import pandas as pd

from . import config
from .textnorm import make_doc


class Store:
    def __init__(self):
        self.meds = pd.read_csv(config.MEDS_CSV).fillna({"dci": "", "forme": "", "presentation": "", "nom": "", "type": ""})
        self.pharmacies = self._load_pharmacies()
        self.stock = self._load_stock()

    def _load_pharmacies(self):
        if not config.PHARMACIES_CSV.exists():
            return pd.DataFrame(columns=["id", "name", "name_ar", "city", "quartier", "address", "phone",
                                         "lat", "lon", "opening_hours", "on_duty", "source"])
        df = pd.read_csv(config.PHARMACIES_CSV).fillna("")
        df["name_norm"] = df["name"].map(lambda s: make_doc(str(s).replace("Pharmacie", "")).norm)
        df["name_ar_norm"] = df.get("name_ar", "").map(lambda s: make_doc(str(s)).norm)
        df["city_norm"] = df["city"].map(lambda s: make_doc(str(s)).norm)
        return df

    def _load_stock(self):
        """{code: {pharmacy_id: quantité}}"""
        if not config.STOCK_CSV.exists():
            return {}
        st = pd.read_csv(config.STOCK_CSV, dtype={"code": str})
        out = {}
        for code, pid, q in zip(st["code"], st["pharmacy_id"], st["quantity"]):
            out.setdefault(code, {})[int(pid)] = int(q)
        return out
