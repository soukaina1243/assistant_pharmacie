"""Génère des pharmacies et un stock SIMULÉS (démonstration) -> data/demo/.

Aucune source publique ne donne le stock des pharmacies : le stock est donc aléatoire
(graine fixe, reproductible). Les pharmacies sont fictives mais réalistes.
À remplacer par scripts/build_pharmacies.py (AMMPS + OpenStreetMap) pour de vraies données.
"""
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/demo"
random.seed(42)

CITIES = {  # ville: (lat, lon, indicatif, quartiers)
    "RABAT": (34.0209, -6.8416, "0537", ["Agdal", "Hay Riad", "Souissi", "Hassan", "Yacoub El Mansour"]),
    "CASABLANCA": (33.5731, -7.5898, "0522", ["Maarif", "Gauthier", "Anfa", "Hay Hassani", "Ain Sebaa", "Bourgogne"]),
    "SALE": (34.0531, -6.7985, "0537", ["Bettana", "Tabriquet", "Hay Salam", "Sala Al Jadida"]),
    "MARRAKECH": (31.6295, -7.9811, "0524", ["Gueliz", "Hivernage", "Medina", "Daoudiate", "Massira"]),
    "FES": (34.0181, -5.0078, "0535", ["Ville Nouvelle", "Narjiss", "Atlas", "Saiss"]),
    "TANGER": (35.7595, -5.8340, "0539", ["Malabata", "Iberia", "Marshane", "Boukhalef"]),
    "MEKNES": (33.8935, -5.5547, "0535", ["Hamria", "Marjane", "Toulal", "Ville Nouvelle"]),
    "AGADIR": (30.4278, -9.5981, "0528", ["Talborjt", "Founty", "Anza", "Hay Mohammadi"]),
}
NAMES = {  # nom latin: nom arabe
    "Ibn Sina": "ابن سينا", "Al Amal": "الأمل", "Centrale": "المركزية", "Ennahda": "النهضة",
    "Hassan II": "الحسن الثاني", "Al Massira": "المسيرة", "Al Andalous": "الأندلس", "Al Fath": "الفتح",
    "La Paix": "السلام", "Du Soleil": "الشمس", "Ibn Rochd": "ابن رشد", "Al Wifaq": "الوفاق",
    "Ennasr": "النصر", "Atlas": "الأطلس", "Al Manar": "المنار", "Al Qods": "القدس",
}


def main():
    rows, pid = [], 1
    for city, (lat, lon, ind, quartiers) in CITIES.items():
        for name in random.sample(list(NAMES), 12):
            q = random.choice(quartiers)
            rows.append({
                "id": pid, "name": f"Pharmacie {name}", "name_ar": f"صيدلية {NAMES[name]}", "city": city,
                "quartier": q, "address": f"{random.randint(1, 180)} rue {random.choice(['Moulay Youssef', 'Al Massira', 'Mohammed V', 'Hassan II', 'Ibn Battouta'])}, {q}",
                "phone": f"{ind[:2]} {ind[2:]} {random.randint(10, 99)} {random.randint(10, 99)} {random.randint(10, 99)}",
                "lat": round(lat + random.uniform(-0.03, 0.03), 5), "lon": round(lon + random.uniform(-0.03, 0.03), 5),
                "opening_hours": random.choice(["09:00-20:00", "08:30-21:00", "09:00-13:00 / 15:00-20:00"]),
                "on_duty": random.random() < 0.15, "source": "synthetic_demo"})
            pid += 1
    ph = pd.DataFrame(rows)
    for city in CITIES:                       # au moins 2 pharmacies de garde par ville (démo)
        idx = ph[ph["city"] == city].sample(2, random_state=1).index
        ph.loc[idx, "on_duty"] = True
    OUT.mkdir(parents=True, exist_ok=True)
    ph.to_csv(OUT / "pharmacies.csv", index=False)
    make_stock(ph, OUT / "stock.csv")


def make_stock(ph, out_path):

    meds = pd.read_csv(ROOT / "data/processed/medications.csv")
    top = meds.groupby("brand").size().sort_values(ascending=False).head(250).index
    must = ["DOLIPRANE", "AUGMENTIN", "EFFERALGAN", "SPASFON", "SMECTA", "VENTOLINE", "VOLTARENE", "GAVISCON", "MAXILASE", "CLAMOXYL", "ZYRTEC"]
    codes = meds[meds["brand"].isin(set(top) | set(must))]["code"].astype(str).tolist()
    stock = []
    for _, p in ph.iterrows():
        for c in codes:
            if random.random() < 0.65:
                stock.append((c, p["id"], random.randint(1, 30)))
    pd.DataFrame(stock, columns=["code", "pharmacy_id", "quantity"]).to_csv(out_path, index=False)
    print(f"{len(ph)} pharmacies, {len(stock)} lignes de stock SIMULÉ -> {out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--stock-only":   # stock simulé pour un vrai fichier pharmacies
        make_stock(pd.read_csv(sys.argv[2]), ROOT / "data/processed/stock.csv")
    else:
        main()
