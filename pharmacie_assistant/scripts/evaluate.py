"""Évalue le pipeline sur le corpus annoté.

Usage : python scripts/evaluate.py [--corpus data/corpus_pharmacie.jsonl] [--modes rules ml hybrid] [-v]
Sortie : tableau de métriques + reports/eval_<mode>.json (erreurs détaillées).
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import f1_score  # noqa: E402

from app.data import Store  # noqa: E402
from app.pipeline import Assistant  # noqa: E402


def _eq(f, a, b):
    if f in ("pharmacy_name", "neighborhood") and isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    return a == b


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def evaluate(rows, assistant):
    preds = [assistant.analyze(r["text"], do_fulfill=False) for r in rows]
    m, errors = collections.defaultdict(list), []
    by_lang = collections.defaultdict(list)
    y_true, y_pred = [], []
    for r, p in zip(rows, preds):
        g, e = r["entities"], p["entities"]
        ok_intent = p["intent"]["label"] == r["intent"]
        y_true.append(r["intent"]); y_pred.append(p["intent"]["label"])
        m["intent"].append(ok_intent)
        m["status"].append(p["status"] == r["status"])
        m["lang_primary"].append(p["language"]["primary"] == r["language"]["primary"])
        m["script"].append(p["language"]["script"] == r["language"]["script"])
        by_lang[(r["language"]["primary"], r["language"]["script"])].append(ok_intent)
        if r["status"] != "out_of_scope":
            for f in ["quantity", "pharmacy_name", "city", "neighborhood", "time_reference"]:
                m[f].append(_eq(f, e[f], g[f]))
            gm, pm = g["medications"], e["medications"]
            if gm:
                m["med_top1"].append(bool(pm) and pm[0]["name"] == gm[0]["name"])
                m["med_all"].append([x["name"] for x in pm] == [x["name"] for x in gm])
                m["dosage"].append(bool(pm) and pm[0]["dosage"] == gm[0]["dosage"])
                m["form"].append(bool(pm) and pm[0]["form"] == gm[0]["form"])
            else:
                m["no_med_false_positive"].append(len(pm) == 0)
        bad = [k for k in ("intent",) if not ok_intent]
        bad += ["status"] if p["status"] != r["status"] else []
        if r["status"] != "out_of_scope":
            bad += [f for f in ["quantity", "pharmacy_name", "city", "neighborhood", "time_reference"] if not _eq(f, e[f], g[f])]
            gm, pm = g["medications"], e["medications"]
            if [x["name"] for x in gm] != [x["name"] for x in pm]:
                bad.append("medications")
            elif gm and (pm[0]["dosage"] != gm[0]["dosage"] or pm[0]["form"] != gm[0]["form"]):
                bad.append("dosage/form")
        if bad:
            errors.append({"id": r["id"], "text": r["text"], "wrong": bad,
                           "gold": {"intent": r["intent"], "status": r["status"], **g},
                           "pred": {"intent": p["intent"]["label"], "status": p["status"], **e}})
    res = {k: round(sum(v) / len(v), 3) for k, v in m.items()}
    res["intent_macro_f1"] = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3)
    res["intent_by_language"] = {f"{a}/{b}": round(sum(v) / len(v), 3) for (a, b), v in by_lang.items()}
    res["n"] = len(rows)
    return res, errors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data/corpus_pharmacie.jsonl"))
    ap.add_argument("--modes", nargs="+", default=["rules", "ml", "hybrid"])
    ap.add_argument("-v", action="store_true", help="afficher les erreurs")
    a = ap.parse_args()
    rows, store = load(a.corpus), Store()
    (ROOT / "reports").mkdir(exist_ok=True)
    for mode in a.modes:
        try:
            asst = Assistant(intent_mode=mode, store=store)
        except FileNotFoundError as e:
            print(f"[{mode}] ignoré : {e}"); continue
        res, errs = evaluate(rows, asst)
        json.dump({"metrics": res, "errors": errs}, open(ROOT / f"reports/eval_{mode}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"\n=== {mode.upper()}  (n={res['n']}) ===")
        for k, v in res.items():
            if k not in {"n", "intent_by_language"}:
                print(f"  {k:24s} {v}")
        print("  intent par langue/graphie:", res["intent_by_language"])
        if a.v:
            for e in errs:
                print(f"  ✗ {e['id']} {e['text']}  -> {e['wrong']}\n      gold={e['gold']}\n      pred={e['pred']}")
