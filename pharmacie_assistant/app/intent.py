"""Classification d'intention : règles (baseline), ML (TF-IDF caractères + régression logistique), hybride."""
from __future__ import annotations

from . import config
from .lexicon import INTENT_PATTERNS, INTENT_PRIORITY, INTENTS


def rule_scores(norm: str) -> dict:
    return {i: float(sum(w for rx, w in pats if rx.search(norm))) for i, pats in INTENT_PATTERNS.items()}


def _rule_decision(scores: dict):
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], INTENT_PRIORITY.index(kv[0])))
    (best, s1), (_, s2) = ranked[0], ranked[1]
    if s1 <= 0:
        return None, 0.0, 0.0
    if best != "out_of_scope" and scores["out_of_scope"] >= 4 and scores["out_of_scope"] >= s1:
        best, s1 = "out_of_scope", scores["out_of_scope"]
    conf = min(0.97, s1 / (s1 + s2 + 1.0))
    return best, round(conf, 3), s1


class IntentClassifier:
    def __init__(self, mode: str = config.INTENT_MODE, model_path=config.INTENT_MODEL):
        self.mode = mode
        self.model = None
        if mode in {"ml", "hybrid"} and model_path.exists():
            import joblib
            self.model = joblib.load(model_path)
        elif mode == "ml":
            raise FileNotFoundError(f"Modèle introuvable ({model_path}) : lance scripts/train_intent.py")

    def ml_probs(self, norm: str) -> dict:
        probs = self.model.predict_proba([norm])[0]
        return dict(zip(map(str, self.model.classes_), map(float, probs)))

    def predict(self, norm: str):
        """-> (label | None, confidence, source). None = aucun indice (le pipeline décide)."""
        label, conf, strength = _rule_decision(rule_scores(norm))
        if self.mode == "rules" or self.model is None:
            return label, conf, "rules"
        probs = self.ml_probs(norm)
        ml_label, ml_conf = max(probs.items(), key=lambda kv: kv[1])
        if self.mode == "ml":
            return ml_label, round(ml_conf, 3), "ml"
        # hybride : les règles fortes (>=3) priment ; sinon le modèle ML si assez sûr
        if label is not None and strength >= 3:
            if label != ml_label and ml_conf > 0.85 and strength < 4:
                return ml_label, round(ml_conf, 3), "ml"
            return label, conf, "rules"
        if ml_conf >= 0.5:
            return ml_label, round(ml_conf, 3), "ml"
        return label, conf, "rules"
