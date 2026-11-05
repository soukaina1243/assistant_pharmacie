"""Reconnaissance d'un médicament à partir d'une photo de boîte : OCR -> preuves -> appariement au référentiel."""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from .. import config
from ..lexicon import FORM_LOOKUP, FORM_REGEX
from ..matcher import MedicationMatcher
from ..textnorm import is_arabic, norm_token, skeleton
from . import preprocess as pp
from .ocr import read_text

# Mots d'emballage (jamais des noms de médicaments) : forme, mentions légales, laboratoires courants.
PACK_STOP = set("""boite boites comprime comprimes gelule gelules sachet sachets sirop solution injectable ampoule ampoules flacon
tube laboratoire laboratoires labo usage oral orale voie notice lire attentivement conserver abri chaleur humidite enfants
enfant portee adulte adultes dosage pharma pharmaceutique pharmaceutiques maroc marketing cooper sothema biopharma afric phar
galenica laprophan bottu novartis sanofi pfizer gsk bayer roche merck abbott mylan teva biogaran zentiva sandoz servier lilly
generique princeps lot exp fab fabrique prix ppv sans ordonnance posologie contient excipient comprimé gélules""".split())
FORM_AR_MAP = {"اقراص": "comprimé", "قرص": "comprimé", "كبسولات": "gélule", "كبسوله": "gélule", "شراب": "sirop",
               "اكياس": "sachet", "حقن": "injectable", "تحاميل": "suppositoire"}
AR_PACK_STOP = {norm_token(x) for x in ["أقراص", "قرص", "كبسولات", "كبسولة", "شراب", "أكياس", "حقن", "تحاميل", "علبة", "دواء", "الدواء", "فموي",
                                        "للاستعمال", "مرهم", "كريم", "جيل", "قطرة", "للعين", "الفموي", "الاستعمال", "عن", "طريق", "الفم"]}
RE_DOSE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mg|g|ml|mcg|ug|ui)\b")
RE_PACK = re.compile(r"(\d{1,3})\s*(?:comprim|gelule|sachet|ampoule|suppositoire|capsule)")
ADVICE = {"image_floue": "Photo floue : stabilisez le téléphone et rapprochez-vous de la boîte.",
          "image_sombre": "Photo trop sombre : ajoutez de la lumière.",
          "reflets": "Reflets sur l'emballage : inclinez légèrement la boîte.",
          "basse_resolution": "Résolution trop faible : rapprochez-vous ou utilisez une meilleure photo."}


def _clean(t: str) -> str:
    return re.sub(r"[^\w\u0600-\u06FF]", "", norm_token(t))


def _fix_digits(t: str) -> str:
    return t.translate(str.maketrans("SOolI|B", "5001118")) if re.fullmatch(r"[0-9SOolI|B]{2,5}", t) and re.search(r"\d", t) else t


class BoxRecognizer:
    def __init__(self, matcher: MedicationMatcher):
        self.m = matcher
        self.meds = matcher.meds
        self._cache = {}
        self._pl = [p.replace(" ", "") for p in matcher.plain]          # index préfixe / suffixe (noms tronqués)
        self._p4, self._s4 = defaultdict(list), defaultdict(list)
        for i, (name, kind, _) in enumerate(matcher.entries):
            if kind == "brand" and len(self._pl[i]) >= 5:
                self._p4[self._pl[i][:4]].append(i)
                self._s4[self._pl[i][-4:]].append(i)

    # ------------------------------------------------------------------ preuves « nom de produit »
    def _attempt_evidence(self, ws) -> dict:
        """Preuves d'UN essai d'OCR : {(nom, 'brand'|'dci'): {'e', 'lat', 'where', 'cnt'}}."""
        ev = {}
        ok_lat = [w.h for w in ws if w.script == "latin" and w.conf >= 50 and len(w.text) >= 3]
        hmax = max(ok_lat or [1])

        def bump(key, sc, conf, prom, script, text):
            e = (sc / 100) * min(1.0, max(0.2, (conf - 30) / 60)) * prom * (0.85 if script == "arabic" else 1.0)
            d = ev.setdefault(key, {"e": 0.0, "lat": 0.0, "where": [], "cnt": 0})
            d["cnt"] += 1
            if script == "latin":
                d["lat"] = max(d["lat"], e)
            if e > d["e"]:
                d["e"], d["where"] = e, [text, round(sc, 1), round(conf)]

        lines = defaultdict(list)
        for w in ws:
            if w.conf >= 35:
                lines[(w.script, w.line)].append(w)
        for (script, _), lw in lines.items():
            lw.sort(key=lambda w: w.x)
            runs, cur = [], []
            for w in lw:                                        # séquences de tokens « utilisables »
                t = _clean(w.text)
                ar_ok = script != "arabic" or (t not in AR_PACK_STOP and len(skeleton(t[2:] if t.startswith("ال") else t)) >= 4)
                if len(t) >= 4 and not t.isdigit() and t not in PACK_STOP and ar_ok and not self.m._skip(t):
                    cur.append((t, w))
                elif cur:
                    runs.append(cur); cur = []
            if cur:
                runs.append(cur)
            for run in runs:
                for i in range(len(run)):
                    for L in (1, 2, 3):
                        if i + L > len(run):
                            break
                        toks = [t for t, _ in run[i:i + L]]
                        ws_ = [w for _, w in run[i:i + L]]
                        conf = sum(w.conf for w in ws_) / L
                        prom = 0.35 + 0.65 * min(1.0, max(w.h for w in ws_) / hmax) ** 1.3
                        text = " ".join(w.text for w in ws_)
                        cands = [toks] + ([["".join(toks)]] if L > 1 else [])
                        sc, ent = max((self.m.score_tokens(c) for c in cands), key=lambda x: x[0])
                        if ent is not None and sc >= config.MED_CANDIDATE:
                            bump((ent[0], ent[1]), sc, conf, prom, script, text)
                        # nom TRONQUÉ (boîte partiellement visible) : préfixe ou suffixe d'une marque
                        t0 = toks[0]
                        is_dci = ent is not None and ent[1] == "dci" and sc >= 88
                        if (L == 1 and script == "latin" and len(t0) >= 5 and conf >= 60 and not is_dci
                                and not any(pw.startswith(t0) for pw in PACK_STOP)):
                            for idx in self._p4.get(t0[:4], ()):
                                pl = self._pl[idx]
                                if pl.startswith(t0) and pl != t0 and len(t0) / len(pl) >= 0.5:
                                    bump((self.m.entries[idx][0], "brand"), 88 + 8 * len(t0) / len(pl), conf, prom, script, text + " (début)")
                            for idx in self._s4.get(t0[-4:], ()):
                                pl = self._pl[idx]
                                if pl.endswith(t0) and pl != t0 and len(t0) / len(pl) >= 0.5:
                                    bump((self.m.entries[idx][0], "brand"), 84 + 8 * len(t0) / len(pl), conf, prom, script, text + " (fin)")
        return ev

    def _evidence(self, words) -> dict:
        """Fusionne les preuves de tous les essais (calcul mis en cache par essai)."""
        by_att = defaultdict(list)
        for w in words:
            by_att[w.attempt].append(w)
        merged = {}
        for att, ws in by_att.items():
            if att not in self._cache:
                self._cache[att] = self._attempt_evidence(ws)
            for k, d in self._cache[att].items():
                m = merged.setdefault(k, {"e": 0.0, "lat": 0.0, "where": [], "n": 0})
                m["lat"] = max(m["lat"], d["lat"]); m["n"] = max(m["n"], d["cnt"])
                if d["e"] > m["e"]:
                    m["e"], m["where"] = d["e"], d["where"]
        for d in merged.values():
            d["e"] = min(1.0, d["e"] + 0.05 * min(2, d["n"] - 1))
        return merged

    def _enough(self, words) -> bool:
        """Arrêt de l'OCR : marque certaine ET (dosage lu, OU une seule dose existe pour cette marque, OU 3 essais déjà faits)."""
        ev = self._evidence(words)
        sure = [k[0] for k, v in ev.items() if k[1] == "brand" and v["e"] >= config.BOX_STOP]
        if not sure:
            return False
        if len({w.attempt for w in words}) >= 3 or self._attributes(words)[0]:
            return True
        doses = {d for b in sure for d in self.meds.iloc[list(self.m._by_brand.get(b, []))]["dosage_mg"].dropna().round(4)}
        return len(doses) <= 1

    # ------------------------------------------------------------------ attributs : dosage, forme, conditionnement
    def _attributes(self, words):
        doses, forms, packs = defaultdict(int), defaultdict(int), defaultdict(int)
        by = defaultdict(list)
        for w in words:
            if w.conf >= 35:
                by[(w.attempt, w.script, w.line)].append(w)
        for (_, script, _), lw in by.items():
            lw.sort(key=lambda w: w.x)
            toks = [norm_token(w.text).strip("._,;:()[]|") for w in lw]
            toks = [_fix_digits(t) if i + 1 < len(toks) and toks[i + 1][:2] in {"mg", "ml", "g", "ug", "mc", "ui"} or re.fullmatch(r"[0-9SOolI|B]{2,5}(mg|g|ml)", t) else t
                    for i, t in enumerate(toks)]
            text = " ".join(toks)
            for v, u in RE_DOSE.findall(text.replace(",", ".")):
                val = float(v)
                mg = {"mg": val, "g": val * 1000, "mcg": val / 1000, "ug": val / 1000}.get(u)
                if mg:
                    doses[round(mg, 4)] += 1
            for m in RE_PACK.findall(text):
                packs[int(m)] += 1
            for t in toks:
                f = FORM_AR_MAP.get(t) if is_arabic(t) else FORM_LOOKUP.get(t)
                if f:
                    forms[f] += 1
        return dict(doses), dict(forms), dict(packs)

    # ------------------------------------------------------------------ API
    def recognize(self, img: np.ndarray, top_k: int = 3, use_arabic: bool = True) -> dict:
        q = pp.quality_metrics(img)
        self._cache = {}
        ocr = read_text(img, enough=self._enough, use_arabic=use_arabic)
        words = ocr["words"]
        ev = self._evidence(words)
        doses, forms, packs = self._attributes(words)
        form = max(forms, key=forms.get) if forms else None
        dci_found = sorted([k[0] for k, v in ev.items() if k[1] == "dci" and v["e"] >= 0.5], key=lambda n: -ev[(n, "dci")]["e"])
        brands = sorted([(k[0], v) for k, v in ev.items() if k[1] == "brand" and v["e"] >= 0.30], key=lambda kv: -kv[1]["e"])[:5]

        cands = []
        strong_dci = [d for d in dci_found if ev[(d, "dci")]["e"] >= 0.8]

        def add_rows(rows, base, why, arabic_only=False):
            for _, r in rows.iterrows():
                s, notes = base, list(why)
                mg = r["dosage_mg"]
                if doses and pd.notna(mg):
                    if any(abs(mg - d) < 1e-6 for d in doses):
                        s += 0.25; notes.append("dosage lu")
                    else:
                        s -= 0.10
                if form and re.search(FORM_REGEX.get(form, "$^"), str(r["forme"])):
                    s += 0.10; notes.append("forme lue")
                if packs and any(int(n) in packs for n in re.findall(r"\d+", str(r["presentation"]))):
                    s += 0.08; notes.append("conditionnement lu")
                if dci_found and any(d in str(r["dci"]) for d in dci_found):
                    s += 0.07; notes.append("DCI lue")
                elif strong_dci and base > 0.30 and str(r["dci"]):   # la DCI imprimée contredit cette marque
                    s -= 0.25; notes.append("DCI incompatible")
                if arabic_only:
                    s = min(s, 0.65); notes.append("preuve arabe seule (plafonnée)")
                cands.append((min(1.0, max(0.0, s)), r, notes))

        for name, v in brands:
            rows = self.meds.iloc[list(self.m._by_brand.get(name, []))]
            add_rows(rows, v["e"] * 0.72, [f"nom « {v['where'][0]} » (score {v['where'][1]})"], arabic_only=v["lat"] < 0.3)
        if not brands and dci_found:                          # générique : seule la DCI est lisible
            for d in dci_found[:2]:
                rows = self.meds.iloc[list(self.m._by_dci.get(d, []))]
                add_rows(rows, min(0.30, ev[(d, "dci")]["e"] * 0.3), [f"DCI « {d} »"])
        cands.sort(key=lambda c: -c[0])
        seen, matches = set(), []
        for s, r, notes in cands:
            key = (r["nom"], r["forme"], r["presentation"])
            if key in seen:
                continue
            seen.add(key)
            matches.append({"rank": len(matches) + 1, "code": str(r["code"]), "name": r["nom"], "brand": r["brand"], "dci": r["dci"],
                            "dosage": None if pd.isna(r["dosage"]) else f"{r['dosage']:g} {str(r['unite']).lower()}",
                            "dosage_mg": None if pd.isna(r["dosage_mg"]) else float(r["dosage_mg"]),
                            "form": r["forme"], "presentation": r["presentation"],
                            "ppv_mad": None if pd.isna(r["ppv"]) else float(r["ppv"]),
                            "reimbursement_pct": None if pd.isna(r["taux"]) else float(r["taux"]),
                            "score": round(s, 3), "evidence": notes})
            if len(matches) >= top_k:
                break

        reliable = [w for w in words if w.conf >= 50 and len(w.text) >= 3]
        warns = list(q["warnings"])
        if len(reliable) < 2:
            status, conf = "no_text", 0.0
        else:
            conf = matches[0]["score"] if matches else 0.0
            tb = matches[0]["brand"] if matches else ""
            # concurrent = autre marque dont le nom n'est pas une extension du premier (DOLIPRANE / DOLIPRANE VITAMINE C)
            second = next((m["score"] for m in matches[1:] if not (m["brand"].startswith(tb) or tb.startswith(m["brand"]))), 0.0) if matches else 0.0
            status = "ok" if conf >= config.BOX_OK and conf - second >= 0.08 else ("low_confidence" if conf >= config.BOX_LOW else "unknown")
            if matches and not doses:
                alts = sorted({f"{r['dosage']:g} {str(r['unite']).lower()}" for _, r, _ in cands[:40]
                               if r["brand"] == matches[0]["brand"] and pd.notna(r["dosage"])})
                if len(alts) > 1:
                    warns.append("dosage_non_lu")
        bw = [w for w in words if w.attempt == ocr["best_attempt"] or w.script == "arabic"]
        lines = defaultdict(list)
        for w in bw:
            lines[(w.attempt, w.script, w.line)].append(w)
        out_lines = [{"text": " ".join(w.text for w in sorted(lw, key=lambda w: w.x)),
                      "conf": round(sum(w.conf for w in lw) / len(lw)), "script": lw[0].script}
                     for _, lw in sorted(lines.items(), key=lambda kv: min(w.y for w in kv[1])) if any(w.conf >= 30 for w in lw)]
        return {"status": status, "confidence": round(conf, 3), "matches": matches,
                "detected": {"brand_candidates": [{"name": n, "evidence": round(v["e"], 3), "read_as": v["where"][0]} for n, v in brands],
                             "dci": dci_found, "dosage_mg": sorted(doses), "form": form, "pack_sizes": sorted(packs)},
                "ocr": {"lines": out_lines[:30], "n_words": len(words), "angle": ocr["angle"], "variant": ocr["variant"],
                        "attempts": len(ocr["attempts"]), "time_ms": ocr["time_ms"]},
                "quality": q, "warnings": warns, "advice": [ADVICE[w] for w in q["warnings"] if w in ADVICE]}
