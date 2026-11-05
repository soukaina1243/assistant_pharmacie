"""Évalue la reconnaissance de boîtes sur un jeu d'images étiquetées (voir make_box_dataset.py).

    python scripts/eval_boxes.py --split dev  [--limit N]
    python scripts/eval_boxes.py --split test
    python scripts/eval_boxes.py --labels data/boxes_real/labels.jsonl      # vraies photos (même format)

Métriques par condition : top-1 / top-3 de la MARQUE, top-1 du PRODUIT (marque + dosage), taux de réponses « ok »
(couverture), précision quand le système se dit sûr, et surtout le taux d'ERREURS CONFIANTES (statut ok mais faux).
Le calcul est repris (cache) si le script est interrompu.
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.data import Store  # noqa: E402
from app.matcher import MedicationMatcher  # noqa: E402
from app.vision.recognize import BoxRecognizer  # noqa: E402


def judge(lab, res):
    ms = res["matches"]
    brand_ok = [m["brand"] == lab["brand"] for m in ms]
    prod_ok = [m["brand"] == lab["brand"] and lab["dosage_mg"] is not None and m["dosage_mg"] is not None
               and abs(m["dosage_mg"] - lab["dosage_mg"]) < 1e-6 for m in ms]
    return {"top1": bool(brand_ok[:1] and brand_ok[0]), "top3": any(brand_ok[:3]),
            "prod1": bool(prod_ok[:1] and prod_ok[0]), "prod3": any(prod_ok[:3])}


def summarize(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r); by["ALL"].append(r)
    out = {}
    for c, rs in by.items():
        n = len(rs)
        ok = [r for r in rs if r["status"] == "ok"]
        out[c] = {"n": n, "top1": sum(r["top1"] for r in rs) / n, "top3": sum(r["top3"] for r in rs) / n,
                  "prod1": sum(r["prod1"] for r in rs) / n, "coverage_ok": len(ok) / n,
                  "precision_ok": (sum(r["top1"] for r in ok) / len(ok)) if ok else None,
                  "confident_errors": sum(1 for r in ok if not r["top1"]) / n,
                  "median_s": sorted(r["time_s"] for r in rs)[n // 2], "max_s": max(r["time_s"] for r in rs)}
    return out


def markdown(title, summ):
    order = ["clean", "blur", "dark", "glare", "rot15", "rot90", "partial", "lowres", "hard", "ALL"]
    L = [f"### {title}\n", "| Condition | n | Marque top-1 | Marque top-3 | Produit top-1 (marque+dosage) | Réponses « ok » | Précision quand « ok » | Erreurs confiantes | Temps médian |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in [c for c in order if c in summ] + [c for c in summ if c not in order]:
        s = summ[c]
        pr = "--" if s["precision_ok"] is None else f"{100 * s['precision_ok']:.1f} %"
        L.append(f"| {c} | {s['n']} | {100 * s['top1']:.1f} % | {100 * s['top3']:.1f} % | {100 * s['prod1']:.1f} % | {100 * s['coverage_ok']:.1f} % | {pr} | "
                 f"{100 * s['confident_errors']:.1f} % | {s['median_s']:.1f} s |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--labels")
    ap.add_argument("--limit", type=int, default=0, help="nb de produits max")
    a = ap.parse_args()
    lab_path = Path(a.labels) if a.labels else ROOT / f"data/boxes/{a.split}/labels.jsonl"
    base = lab_path.parent.parent if not a.labels else lab_path.parent
    tag = a.split or lab_path.parent.name
    labels = [json.loads(l) for l in open(lab_path, encoding="utf-8") if l.strip()]
    if a.limit:
        keep = list(dict.fromkeys(l["code"] for l in labels))[:a.limit]
        labels = [l for l in labels if l["code"] in keep]
    cache = ROOT / f"reports/box_eval_{tag}.jsonl"
    cache.parent.mkdir(exist_ok=True)
    done = {}
    if cache.exists():
        for l in open(cache, encoding="utf-8"):
            r = json.loads(l); done[r["file"]] = r
    R = BoxRecognizer(MedicationMatcher(Store().meds))
    with open(cache, "a", encoding="utf-8") as f:
        for i, lab in enumerate(labels, 1):
            if lab["file"] in done:
                continue
            img = cv2.imread(str(base / lab["file"]))
            t0 = time.perf_counter()
            res = R.recognize(img)
            row = {**{k: lab[k] for k in ("file", "condition", "brand", "dosage_mg")}, **judge(lab, res), "status": res["status"],
                   "confidence": res["confidence"], "pred": res["matches"][0]["brand"] if res["matches"] else None,
                   "warnings": res["warnings"], "attempts": res["ocr"]["attempts"], "time_s": round(time.perf_counter() - t0, 2)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            done[lab["file"]] = row
            if i % 20 == 0:
                print(f"{i}/{len(labels)}", flush=True)
    rows = [done[l["file"]] for l in labels]
    summ = summarize(rows)
    md = markdown(f"Reconnaissance de boîtes -- jeu « {tag} » ({len(rows)} images)", summ)
    Path(ROOT / f"reports/BOX_RESULTS_{tag}.md").write_text(md, encoding="utf-8")
    print(md)
