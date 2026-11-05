"""Réponses aux requêtes à partir des bases (le « fulfillment » : dispo, prix, remboursement, infos pharmacie)."""
from __future__ import annotations

from rapidfuzz import fuzz

from .lexicon import CITY_LOOKUP
from .textnorm import make_doc, skeleton


def _clean(d: dict) -> dict:
    return {k: (None if (isinstance(v, float) and v != v) else v) for k, v in d.items()}


def _pharm_view(r) -> dict:
    return _clean({"id": int(r["id"]), "name": r["name"], "city": r["city"], "quartier": r["quartier"],
                   "address": r["address"], "phone": r["phone"], "lat": r["lat"], "lon": r["lon"],
                   "opening_hours": r["opening_hours"], "on_duty": bool(r["on_duty"]), "source": r["source"]})


def find_pharmacies(store, name=None, city=None, neighborhood=None, duty_only=False, limit=5):
    df = store.pharmacies
    if df.empty:
        return []
    if city:
        df = df[df["city"].str.upper() == city.upper()]
    if neighborhood:
        nb = make_doc(neighborhood).norm
        sub = df[df["quartier"].map(lambda q: make_doc(str(q)).norm).str.contains(nb, regex=False)]
        df = sub if len(sub) else df
    if duty_only:
        df = df[df["on_duty"].astype(str).str.lower().isin(["true", "1"])]
    if name:
        q = make_doc(name).norm
        qs = skeleton(q.replace(" ", ""))
        scores = []
        for _, r in df.iterrows():
            s = max(fuzz.WRatio(q, r["name_norm"]), fuzz.WRatio(q, r["name_ar_norm"]) if r["name_ar_norm"] else 0,
                    fuzz.ratio(qs, skeleton(r["name_norm"].replace(" ", ""))) if qs else 0)
            scores.append(s)
        df = df.assign(_s=scores)
        df = df[df["_s"] >= 80].sort_values("_s", ascending=False)
    return [_pharm_view(r) for _, r in df.head(limit).iterrows()]


def _variants(rows, limit=5):
    out = []
    for _, r in rows.sort_values("ppv").head(limit).iterrows():
        out.append(_clean({"code": r["code"], "label": f"{r['nom']} | {r['forme']} | {r['presentation']}",
                           "ppv_mad": r["ppv"], "reimbursement_pct": r["taux"], "type": r["type"]}))
    return out


def fulfill(store, intent, entities, meds_rows, city):
    """meds_rows : DataFrame des produits du 1er médicament reconnu (déjà filtré dosage/forme)."""
    note = "Prix et taux : référentiel CNOPS 2014 (indicatifs). Stock et pharmacies : données de démonstration."
    if intent in {"price", "reimbursement", "check_availability", "reserve_order"} and (meds_rows is None or meds_rows.empty):
        return None
    if intent == "price":
        return {"type": "price", "variants": _variants(meds_rows),
                "min_ppv_mad": float(meds_rows["ppv"].min()), "max_ppv_mad": float(meds_rows["ppv"].max()), "note": note}
    if intent == "reimbursement":
        rates = sorted({int(x) for x in meds_rows["taux"].dropna()})
        pos = int((meds_rows["taux"].fillna(0) > 0).sum())
        return {"type": "reimbursement", "reimbursed": "all" if pos == len(meds_rows) else ("some" if pos else "none"),
                "rates_pct": rates, "variants": _variants(meds_rows),
                "note": note + " Taux 0 % = non remboursable."}
    if intent in {"check_availability", "reserve_order"}:
        codes = set(meds_rows["code"].astype(str))
        per = {}
        for c in codes:
            for pid, q in store.stock.get(c, {}).items():
                per[pid] = per.get(pid, 0) + q
        res = []
        for pid, q in sorted(per.items(), key=lambda kv: -kv[1]):
            row = store.pharmacies[store.pharmacies["id"] == pid]
            if row.empty or q <= 0:
                continue
            r = row.iloc[0]
            if city and str(r["city"]).upper() != city.upper():
                continue
            res.append({**_pharm_view(r), "quantity_in_stock": q})
        out = {"type": intent, "available": bool(res), "pharmacies": res[:5],
               "scope": city or "toutes les villes (précisez une ville pour filtrer)", "note": note}
        if intent == "reserve_order":
            out["reservation_draft"] = {"quantity": entities.get("quantity") or 1, "status": "draft_not_sent"}
        return out
    return None


def fulfill_pharmacy(store, intent, entities):
    ph = find_pharmacies(store, entities.get("pharmacy_name"), entities.get("city"), entities.get("neighborhood"),
                         duty_only=bool(entities.get("time_reference") and "garde" in entities["time_reference"]))
    if not ph:
        return {"type": intent, "pharmacies": [], "note": "Aucune pharmacie trouvée (base de démonstration)."}
    keep = {"pharmacy_phone": ["id", "name", "city", "phone"],
            "pharmacy_location": ["id", "name", "city", "quartier", "address", "lat", "lon"],
            "pharmacy_hours": ["id", "name", "city", "opening_hours", "on_duty", "phone"]}[intent]
    return {"type": intent, "pharmacies": [{k: p[k] for k in keep} for p in ph],
            "note": "Données pharmacies : démonstration (à remplacer par AMMPS + OSM)."}
