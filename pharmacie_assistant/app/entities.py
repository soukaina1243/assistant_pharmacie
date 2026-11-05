"""Extraction d'entités par règles : dosage, forme, quantité, ville, quartier, pharmacie, temps."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .lexicon import (CITY_LOOKUP, CLASSIFIERS, FORM_LOOKUP, NAME_SKIP_LEADING, NB_MARKERS,
                      NEIGHBORHOODS, NUM_WORDS, PHARMACY_WORDS, TIME_LABEL, TIME_TAGS, UNIT_MAP,
                      is_function)
from .textnorm import Doc, is_arabic

RE_DOSE = re.compile(r"(\d+(?:\.\d+)?)(mg|gr|g|ml|mcg|ui)")
RE_NUM = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class Extracted:
    doses: list = field(default_factory=list)   # (idx, "1000 mg", mg_value|None)
    forms: list = field(default_factory=list)   # (idx, "sirop")
    quantity: int | None = None
    qty_bare: int | None = None                 # petit nombre sans classificateur (ex: "2 Doliprane")
    city: str | None = None
    neighborhood: str | None = None
    pharmacy_name: str | None = None
    time_reference: str | None = None
    consumed: set = field(default_factory=set)


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def _dose(val: float, unit: str):
    unit = UNIT_MAP.get(unit, unit)
    mg = {"mg": val, "g": val * 1000, "mcg": val / 1000}.get(unit)
    return f"{_fmt(val)} {unit}", mg


def extract_dosage_qty_form(doc: Doc, ex: Extracted) -> None:
    toks, n, i = doc.tokens, len(doc.tokens), 0
    while i < n:
        t = toks[i]
        nxt = toks[i + 1] if i + 1 < n else ""
        m = RE_DOSE.fullmatch(t)
        if m:
            label, mg = _dose(float(m.group(1)), m.group(2))
            ex.doses.append((i, label, mg)); ex.consumed.add(i)
        elif RE_NUM.fullmatch(t):
            v = float(t)
            if nxt in UNIT_MAP:
                label, mg = _dose(v, nxt)
                ex.doses.append((i, label, mg)); ex.consumed.update({i, i + 1}); i += 1
            elif (nxt in CLASSIFIERS or (nxt in FORM_LOOKUP and v < 10)) and v < 100:
                ex.quantity = ex.quantity or int(v); ex.consumed.add(i)
            elif v >= 10:
                label, mg = _dose(v, "mg")
                ex.doses.append((i, label, mg)); ex.consumed.add(i)
            else:
                ex.qty_bare = ex.qty_bare or int(v); ex.consumed.add(i)
        elif t in NUM_WORDS and (nxt in CLASSIFIERS or nxt in FORM_LOOKUP):
            ex.quantity = ex.quantity or NUM_WORDS[t]; ex.consumed.add(i)
        if t in FORM_LOOKUP:
            ex.forms.append((i, FORM_LOOKUP[t])); ex.consumed.add(i)
        i += 1


def _arabic_variants(s: str):
    yield s
    if s.startswith("ال"):
        yield s[2:]
    if len(s) > 2 and s[0] in "فبلو":
        yield s[1:]
        if s[1:].startswith("ال"):
            yield s[3:]


def extract_city(doc: Doc, ex: Extracted) -> None:
    toks, n = doc.tokens, len(doc.tokens)
    for L in (3, 2, 1):
        for i in range(n - L + 1):
            if any(k in ex.consumed for k in range(i, i + L)):
                continue
            s = " ".join(toks[i:i + L])
            cands = list(_arabic_variants(s)) if is_arabic(s) else [s]
            if not is_arabic(s) and L == 1 and len(s) > 4 and s[0] == "f":
                cands.append(s[1:])                       # "frabat"
            for c in cands:
                if c in CITY_LOOKUP:
                    ex.city = CITY_LOOKUP[c]
                    ex.consumed.update(range(i, i + L))
                    return


def extract_neighborhood(doc: Doc, ex: Extracted) -> None:
    toks, n = doc.tokens, len(doc.tokens)
    for L in (3, 2, 1):                                   # lexique de quartiers connus
        for i in range(n - L + 1):
            if any(k in ex.consumed for k in range(i, i + L)):
                continue
            s = " ".join(toks[i:i + L])
            if s in NEIGHBORHOODS:
                ex.neighborhood = NEIGHBORHOODS[s]
                ex.consumed.update(range(i, i + L))
                return
    for i, t in enumerate(toks):                          # motif générique « hay X »
        if t in NB_MARKERS and i not in ex.consumed:
            j = i + 1
            while j < n and j < i + 3 and j not in ex.consumed and not is_function(toks[j]):
                j += 1
            if j > i + 1:
                words = doc.orig[i:j]
                ex.neighborhood = " ".join(w if (is_arabic(w) or not w.islower()) else w.capitalize() for w in words)
                ex.consumed.update(range(i, j))
                return


def extract_pharmacy_name(doc: Doc, ex: Extracted) -> None:
    toks, n = doc.tokens, len(doc.tokens)
    for i, t in enumerate(toks):
        if t not in PHARMACY_WORDS:
            continue
        j = i + 1
        while j < n and toks[j] in NAME_SKIP_LEADING:
            j += 1
        k = j
        while k < n and k < j + 4 and k not in ex.consumed and not is_function(toks[k]):
            k += 1
        if k > j:
            ex.pharmacy_name = " ".join(w if (is_arabic(w) or not w.islower()) else w.capitalize() for w in doc.orig[j:k])
            ex.consumed.update(range(j, k))
            return


def extract_time(doc: Doc, ex: Extracted) -> None:
    pad = f" {doc.norm} "
    hit = {tag for tag, phrases in TIME_TAGS.items() if any(f" {p} " in pad for p in phrases)}
    parts = []
    if "garde" in hit:
        parts.append("garde")
    for tag in ("now", "tonight", "night", "today", "tomorrow", "sunday"):
        if tag in hit:
            parts.append(TIME_LABEL[tag]); break
    if "closing" in hit and not parts:
        parts.append("fermeture")
    ex.time_reference = " ".join(parts) or None


def extract_all(doc: Doc, want_pharmacy: bool = True) -> Extracted:
    ex = Extracted()
    extract_dosage_qty_form(doc, ex)
    extract_city(doc, ex)
    extract_neighborhood(doc, ex)
    if want_pharmacy:
        extract_pharmacy_name(doc, ex)
    extract_time(doc, ex)
    return ex
