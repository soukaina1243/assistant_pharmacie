"""Rapprochement flou multi-script des noms de médicaments avec le référentiel.

Stratégie : alias exacts -> égalité exacte -> (distance sur texte latin + distance sur
squelette phonétique). Les noms arabes ne sont comparés qu'au squelette.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz, process

from . import config
from .lexicon import (CLASSIFIERS, FORM_LOOKUP, FORM_REGEX, NUM_WORDS, TIME_TAGS, is_function)
from .textnorm import is_arabic, make_doc, norm_token, skeleton

_TIME_WORDS = {w for ws in TIME_TAGS.values() for p in ws for w in p.split()}


@dataclass
class MedMatch:
    raw: str
    name: str
    kind: str          # brand | dci
    score: float       # 0-100
    start: int
    end: int
    alternatives: list


class MedicationMatcher:
    def __init__(self, meds: pd.DataFrame, aliases_path=config.ALIASES_JSON):
        self.meds = meds.reset_index(drop=True)
        brands = self.meds.groupby("brand").size()
        dcis = self.meds[self.meds["dci"] != ""].groupby("dci").size()
        self.entries = [(b, "brand", int(c)) for b, c in brands.items()]
        self.entries += [(d, "dci", int(c)) for d, c in dcis.items() if d not in brands.index]
        self.plain = [norm_token(e[0]) if " " not in e[0] else make_doc(e[0]).norm for e in self.entries]
        self.skel = [skeleton(p.replace(" ", "")) for p in self.plain]
        self.exact = {}
        for idx, p in enumerate(self.plain):
            if p not in self.exact or self.entries[idx][1] == "brand":
                self.exact[p] = idx
        self.alias = {}
        try:
            for k, v in json.load(open(aliases_path, encoding="utf-8")).items():
                self.alias[make_doc(k).norm] = v.upper()
        except FileNotFoundError:
            pass
        self._by_brand = self.meds.groupby("brand").indices
        self._by_dci = self.meds.groupby("dci").indices

    # ------------------------------------------------------------ scoring d'un n-gramme
    def _bonus(self, idx):
        return min(2.0, 0.3 * math.log2(1 + self.entries[idx][2]))

    def _score_span(self, span: list):
        q = " ".join(span)
        if is_arabic(q):
            variants = [q]
            if q.startswith("ال"):
                variants.append(q[2:])
            if q[0] in "فبلو":
                variants.append(q[1:])
            for v in variants:
                if v in self.alias:
                    return 100.0, self._idx_of(self.alias[v])
            sk = skeleton(variants[-1] if q.startswith("ال") else q)
            if q.startswith("ال"):
                sk = skeleton(q[2:])
            if len(sk) < 3:
                return 0.0, None
            hits = process.extract(sk, self.skel, scorer=fuzz.ratio, limit=8, score_cutoff=70)
            best = max(((s - 3 + self._bonus(i), i) for _, s, i in hits), default=(0.0, None))
            return min(best[0], 99.0), best[1]
        # latin
        if q in self.alias:
            return 100.0, self._idx_of(self.alias[q])
        if q in self.exact:
            return 100.0, self.exact[q]
        if len(q.replace(" ", "")) < 4:
            return 0.0, None
        sk = skeleton(q.replace(" ", ""))
        cand = {i for _, _, i in process.extract(q, self.plain, scorer=fuzz.ratio, limit=10, score_cutoff=60)}
        if len(sk) >= 3:
            cand |= {i for _, _, i in process.extract(sk, self.skel, scorer=fuzz.ratio, limit=10, score_cutoff=70)}
        best = (0.0, None)
        for i in cand:
            sp = fuzz.ratio(q, self.plain[i])
            if sp < 70:
                continue
            ss = fuzz.ratio(sk, self.skel[i]) if sk and self.skel[i] else 0
            sc = 0.6 * sp + 0.4 * ss + self._bonus(i)
            if sc > best[0]:
                best = (sc, i)
        return min(best[0], 99.0), best[1]

    def _idx_of(self, name: str):
        key = make_doc(name).norm
        return self.exact.get(key)

    def score_tokens(self, span: list):
        """Score (0-100) et entrée (nom, type, effectif) du meilleur produit/DCI pour un n-gramme de tokens."""
        score, idx = self._score_span(span)
        return score, (self.entries[idx] if idx is not None else None)

    def _skip(self, tok: str) -> bool:
        base = tok[2:] if tok.startswith("ال") else tok
        return (len(base) < 3 or re.fullmatch(r"[\d.]+", tok) is not None or is_function(tok)
                or tok in FORM_LOOKUP or tok in CLASSIFIERS or tok in NUM_WORDS or tok in _TIME_WORDS)

    # ------------------------------------------------------------ API publique
    def find(self, doc, blocked: set) -> list:
        toks, n = doc.tokens, len(doc.tokens)
        cands = []
        for i in range(n):
            if i in blocked or self._skip(toks[i]):
                continue
            for L in (1, 2, 3):
                j = i + L
                if j > n or any(k in blocked or self._skip(toks[k]) for k in range(i, j)):
                    break
                score, idx = self._score_span(toks[i:j])
                if idx is not None and score >= config.MED_CANDIDATE:
                    cands.append((score, L, i, j, idx))
        cands.sort(key=lambda c: (-c[0], -c[1]))
        used, out = set(), []
        for score, L, i, j, idx in cands:
            if used & set(range(i, j)):
                continue
            used.update(range(i, j))
            name, kind, _ = self.entries[idx]
            out.append(MedMatch(doc.text(i, j), name, kind, round(score, 1), i, j, []))
        return sorted(out, key=lambda m: m.start)

    def products(self, m: MedMatch, dosage_mg=None, form=None):
        """Lignes du référentiel pour un match, filtrées par dosage/forme si possible."""
        idxs = (self._by_brand if m.kind == "brand" else self._by_dci).get(m.name, [])
        df = self.meds.iloc[list(idxs)]
        flags = []
        if dosage_mg is not None:
            f = df[(df["dosage_mg"] - dosage_mg).abs() < 1e-6]
            if len(f):
                df = f
            else:
                flags.append("dosage_not_found")
        if form:
            f = df[df["forme"].str.contains(FORM_REGEX.get(form, form.upper()), regex=True, na=False)]
            if len(f):
                df = f
            else:
                flags.append("form_not_found")
        return df, flags

    @staticmethod
    def label(row) -> str:
        pres = f" | {row['presentation']}" if row["presentation"] else ""
        return f"{row['nom']} | {row['forme']}{pres}"
