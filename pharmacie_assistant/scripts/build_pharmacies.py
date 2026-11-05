"""Construit une vraie base de pharmacies : AMMPS (liste officielle) + OpenStreetMap (GPS, téléphone).

    pip install requests beautifulsoup4 lxml
    python scripts/build_pharmacies.py --ammps --osm --merge
    python scripts/make_demo_data.py --stock-only data/processed/pharmacies.csv     # stock simulé
    PHARMACIES_CSV=data/processed/pharmacies.csv STOCK_CSV=data/processed/stock.csv uvicorn app.main:app

ATTENTION : non testé contre les sites réels (pas d'accès réseau lors de la conception).
La structure HTML de l'AMMPS peut différer : adaptez parse_ammps_page(). Respectez les conditions d'usage.
"""
import argparse
import io
import math
import sys
import time
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.lexicon import CITIES  # noqa: E402
from app.textnorm import make_doc  # noqa: E402

AMMPS_URL = "https://www.ammps.gov.ma/bases-de-donnees/pharmacies?page={}"
OVERPASS = "https://overpass-api.de/api/interpreter"
QUERY = """[out:json][timeout:180];
area["ISO3166-1"="MA"][admin_level=2]->.a;
(node["amenity"="pharmacy"](area.a); way["amenity"="pharmacy"](area.a););
out center tags;"""
CENTERS = {"RABAT": (34.0209, -6.8416), "CASABLANCA": (33.5731, -7.5898), "SALE": (34.0531, -6.7985),
           "MARRAKECH": (31.6295, -7.9811), "FES": (34.0181, -5.0078), "TANGER": (35.7595, -5.8340),
           "MEKNES": (33.8935, -5.5547), "AGADIR": (30.4278, -9.5981), "OUJDA": (34.6814, -1.9086),
           "KENITRA": (34.2610, -6.5802), "TETOUAN": (35.5785, -5.3684), "TEMARA": (33.9287, -6.9066)}


def parse_ammps_page(html: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html))
    df = tables[0].iloc[:, :2]
    df.columns = ["name", "city"]
    return df


def scrape_ammps(pages: int, delay: float = 0.5) -> pd.DataFrame:
    import requests
    out = []
    for p in range(pages):
        r = requests.get(AMMPS_URL.format(p), timeout=30)
        r.raise_for_status()
        out.append(parse_ammps_page(r.text))
        time.sleep(delay)
    df = pd.concat(out, ignore_index=True)
    df["city"] = df["city"].map(canon_city)
    df.to_csv(ROOT / "data/processed/pharmacies_ammps.csv", index=False)
    print(len(df), "pharmacies AMMPS")
    return df


def canon_city(c: str) -> str:
    """'SALE - BOUKNADEL' -> 'SALE' (si la ville de base est connue)."""
    norm = make_doc(str(c)).norm
    for canon in CITIES:
        if norm.startswith(make_doc(canon).norm):
            return canon
    return str(c).upper().strip()


def nearest_city(lat, lon, max_km=25):
    best, bd = "", 1e9
    for c, (la, lo) in CENTERS.items():
        d = 111 * math.hypot(lat - la, (lon - lo) * math.cos(math.radians(la)))
        if d < bd:
            best, bd = c, d
    return best if bd <= max_km else ""


def fetch_osm() -> pd.DataFrame:
    import requests
    data = requests.post(OVERPASS, data={"data": QUERY}, timeout=240).json()["elements"]
    rows = []
    for e in data:
        t = e.get("tags", {})
        lat = e.get("lat") or e.get("center", {}).get("lat")
        lon = e.get("lon") or e.get("center", {}).get("lon")
        if lat is None:
            continue
        rows.append({"name": t.get("name") or t.get("name:fr") or "Pharmacie", "name_ar": t.get("name:ar", ""),
                     "city": canon_city(t.get("addr:city", "")) or nearest_city(lat, lon),
                     "quartier": t.get("addr:suburb", ""), "address": " ".join(filter(None, [t.get("addr:housenumber"), t.get("addr:street")])),
                     "phone": t.get("phone") or t.get("contact:phone", ""), "lat": lat, "lon": lon,
                     "opening_hours": t.get("opening_hours", ""), "on_duty": False, "source": "osm"})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/processed/pharmacies_osm.csv", index=False)
    print(len(df), "pharmacies OSM")
    return df


def merge(ammps: pd.DataFrame, osm: pd.DataFrame, threshold: int = 85) -> pd.DataFrame:
    """Base finale = OSM (GPS/téléphone) + pharmacies AMMPS non retrouvées dans OSM (sans GPS)."""
    osm = osm.copy()
    osm["official"] = False
    matched = set()
    for i, o in osm.iterrows():
        cand = ammps[(ammps["city"] == o["city"]) & (~ammps.index.isin(matched))]
        best = max(((fuzz.token_set_ratio(make_doc(o["name"]).norm, make_doc(a["name"]).norm), j) for j, a in cand.iterrows()),
                   default=(0, None))
        if best[0] >= threshold:
            matched.add(best[1])
            osm.at[i, "official"] = True
    rest = ammps[~ammps.index.isin(matched)].copy()
    for c, v in {"name_ar": "", "quartier": "", "address": "", "phone": "", "lat": "", "lon": "", "opening_hours": "",
                 "on_duty": False, "source": "ammps", "official": True}.items():
        rest[c] = v
    out = pd.concat([osm, rest], ignore_index=True)
    out.insert(0, "id", range(1, len(out) + 1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ammps", action="store_true")
    ap.add_argument("--osm", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--pages", type=int, default=448)
    a = ap.parse_args()
    proc = ROOT / "data/processed"
    if a.ammps:
        scrape_ammps(a.pages)
    if a.osm:
        fetch_osm()
    if a.merge:
        res = merge(pd.read_csv(proc / "pharmacies_ammps.csv"), pd.read_csv(proc / "pharmacies_osm.csv").fillna(""))
        res.to_csv(proc / "pharmacies.csv", index=False)
        print(len(res), "pharmacies ->", proc / "pharmacies.csv")
