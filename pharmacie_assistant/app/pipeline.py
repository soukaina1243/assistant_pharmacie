"""Pipeline complet : texte -> langue -> intention -> entités -> rapprochement -> validation -> JSON."""
from __future__ import annotations

import time
import uuid

from . import config
from .data import Store
from .entities import extract_all
from .intent import IntentClassifier
from .lexicon import (DARIJA_MARKERS, FRENCH_MARKERS, MED_INTENTS, MSA_MARKERS, PHARMACY_INTENTS)
from .lookup import fulfill, fulfill_pharmacy
from .matcher import MedicationMatcher
from .respond import build_reply
from .textnorm import detect_script, make_doc

VERSION = "0.1.0"

CLARIFY = {
    "missing_med": {"french": "Quel médicament cherchez-vous ?",
                    "darija_latin": "Chno smiyt d-dwa li kat9lb 3lih ?",
                    "arabic": "شنو سميت الدوا لي كتقلب عليه؟"},
    "unsure_med": {"french": "Vous voulez dire « {x} » ?",
                   "darija_latin": "Wach kat9sd « {x} » ?",
                   "arabic": "واش كتقصد « {x} »؟"},
    "missing_pharmacy": {"french": "De quelle pharmacie ou de quelle ville parlez-vous ?",
                         "darija_latin": "Chno hiya l-pharmacie wla l-mdina ?",
                         "arabic": "أشمن صيدلية ولا أشمن مدينة؟"},
    "missing_place": {"french": "Dans quelle ville ou quel quartier ?",
                      "darija_latin": "F ina mdina wla ina 7ay ?",
                      "arabic": "فأش مدينة ولا أش حي؟"},
}


def detect_language(doc, script: str):
    toks = set(doc.tokens)
    d, f, m = len(toks & DARIJA_MARKERS), len(toks & FRENCH_MARKERS), len(toks & MSA_MARKERS)
    if script == "arabic":
        primary = "arabic" if m > d else "darija"
    elif script == "mixed":
        primary = "french" if f > d else "darija"
    else:
        primary = "darija" if d > 0 and d >= f else "french"
    code_mixed = script == "mixed" or (primary == "darija" and script == "latin") or (d > 0 and f > 0)
    return primary, code_mixed


def _question(kind, primary, script, x=""):
    key = "arabic" if script == "arabic" else ("french" if primary == "french" else "darija_latin")
    return CLARIFY[kind][key].format(x=x)


class Assistant:
    def __init__(self, intent_mode: str = config.INTENT_MODE, store: Store | None = None):
        self.store = store or Store()
        self.matcher = MedicationMatcher(self.store.meds)
        self.intents = IntentClassifier(intent_mode)

    # ------------------------------------------------------------------
    def _assign(self, matches, ex):
        """Rattache dosage et forme au médicament le plus proche."""
        def dist(idx, m):
            return idx - (m.end - 1) if idx >= m.end else (m.start - idx) + 0.5
        attrs = [{"dose": None, "form": None} for _ in matches]
        for idx, label, mg in ex.doses:
            k = min(range(len(matches)), key=lambda i: dist(idx, matches[i]))
            attrs[k]["dose"] = attrs[k]["dose"] or (label, mg)
        for idx, form in ex.forms:
            k = min(range(len(matches)), key=lambda i: dist(idx, matches[i]))
            attrs[k]["form"] = attrs[k]["form"] or form
        return attrs

    def analyze(self, text: str, modality: str = "text", transcript: str | None = None, do_fulfill: bool = True) -> dict:
        t0 = time.perf_counter()
        doc = make_doc(text)
        script = detect_script(text)
        primary, code_mixed = detect_language(doc, script)
        intent, conf, source = self.intents.predict(doc.norm)
        ex = extract_all(doc)

        matches, warnings = [], []
        if intent == "out_of_scope" and source == "ml":     # garde-fou : un médicament certain = demande valide
            ms = self.matcher.find(doc, ex.consumed)
            if ms and max(m.score for m in ms) >= config.MED_EXTRA:
                intent, conf, source = "check_availability", 0.5, "ml+med"
        if intent is None or intent in MED_INTENTS:
            matches = self.matcher.find(doc, ex.consumed)
            if matches:   # précision : un médicament secondaire doit être quasi certain
                best = max(m.score for m in matches)
                matches = [m for m in matches if m.score == best or m.score >= config.MED_EXTRA]
            if intent is None:
                matches = [m for m in matches if m.score >= config.MED_ACCEPT]   # sans indice : médicament certain exigé
                intent, conf, source = ("check_availability", 0.4, "fallback_med") if matches else ("out_of_scope", 0.5, "fallback")

        meds, first_rows = [], None
        if intent != "out_of_scope" and matches:
            attrs = self._assign(matches, ex)
            for m, a in zip(matches, attrs):
                dose_label, dose_mg = a["dose"] if a["dose"] else (None, None)
                rows, flags = self.matcher.products(m, dose_mg, a["form"])
                warnings += [f"{m.name}:{f}" for f in flags]
                meds.append({"raw": m.raw, "name": m.name, "dosage": dose_label, "form": a["form"],
                             "match_score": round(m.score / 100, 3),
                             "candidates": list(dict.fromkeys(self.matcher.label(r) for _, r in
                                                              rows.sort_values("ppv").head(6).iterrows()))[:3]})
                if first_rows is None and m.score >= config.MED_ACCEPT:
                    first_rows = rows

        quantity = ex.quantity or (ex.qty_bare if intent == "reserve_order" else None)
        entities = {"medications": meds, "quantity": quantity, "pharmacy_name": ex.pharmacy_name,
                    "city": ex.city, "neighborhood": ex.neighborhood, "time_reference": ex.time_reference}

        status, question = "ok", None
        if intent == "out_of_scope":
            status = "out_of_scope"
            entities = {"medications": [], "quantity": None, "pharmacy_name": None, "city": None,
                        "neighborhood": None, "time_reference": None}
        elif intent in MED_INTENTS:
            if not meds:
                status, question = "needs_clarification", _question("missing_med", primary, script)
            elif max(m["match_score"] for m in meds) * 100 < config.MED_ACCEPT:
                best = max(meds, key=lambda m: m["match_score"])
                status, question = "needs_clarification", _question("unsure_med", primary, script, best["name"])
        elif intent in {"pharmacy_phone", "pharmacy_hours"} and not (ex.pharmacy_name or ex.city):
            status, question = "needs_clarification", _question("missing_pharmacy", primary, script)
        elif intent == "pharmacy_location" and not (ex.city or ex.neighborhood or ex.pharmacy_name):
            status, question = "needs_clarification", _question("missing_place", primary, script)

        fulfillment = None
        if do_fulfill and status == "ok":
            if intent in PHARMACY_INTENTS:
                fulfillment = fulfill_pharmacy(self.store, intent, entities)
            elif intent in MED_INTENTS:
                fulfillment = fulfill(self.store, intent, entities, first_rows, ex.city)

        result = {
            "request_id": str(uuid.uuid4()),
            "input": {"modality": modality, "raw_text": text, "transcript": transcript},
            "language": {"primary": primary, "script": script, "code_mixed": code_mixed},
            "intent": {"label": intent, "confidence": conf},
            "entities": entities,
            "status": status,
            "clarification_question": question,
            "warnings": warnings,
            "fulfillment": fulfillment,
            "meta": {"intent_source": source, "pipeline_version": VERSION,
                     "latency_ms": round((time.perf_counter() - t0) * 1000, 1)},
        }
        result["reply"] = build_reply(result)
        result["meta"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return result
