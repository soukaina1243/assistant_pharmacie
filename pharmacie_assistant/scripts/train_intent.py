"""Entraîne le classifieur d'intention (TF-IDF caractères + régression logistique).

Entrée : data/synthetic/train.jsonl (+ optionnel : un corpus réel annoté via --extra)
Sortie : models/intent.joblib
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.textnorm import make_doc  # noqa: E402


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def build():
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(C=20, max_iter=3000, class_weight="balanced")),
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "data/synthetic/train.jsonl"))
    ap.add_argument("--extra", nargs="*", default=[], help="corpus annotés supplémentaires (jsonl avec text+intent)")
    a = ap.parse_args()
    rows = load(a.train) + [r for p in a.extra for r in load(p)]
    X = [make_doc(r["text"]).norm for r in rows]
    y = [r["intent"] for r in rows]
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    m = build().fit(Xtr, ytr)
    print("Validation interne (même gabarits -> optimiste) :")
    print(classification_report(yva, m.predict(Xva), digits=3, zero_division=0))
    final = build().fit(X, y)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(final, ROOT / "models/intent.joblib")
    print(f"Modèle sauvegardé : models/intent.joblib ({len(X)} exemples)")
