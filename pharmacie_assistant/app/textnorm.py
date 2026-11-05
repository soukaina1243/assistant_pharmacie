"""Normalisation (arabe / arabizi / français) et clé phonétique 'squelette'.

Le squelette est la pièce maîtresse du rapprochement multi-script : « دوليبران »,
« doliprane » et « dolipran » donnent tous 'dlbrn'.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

AR_RE = re.compile(r"[\u0600-\u06FF]")
LAT_RE = re.compile(r"[A-Za-z]")
_AR_DIAC = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_AR_FIX = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه",
                         "ؤ": "و", "ئ": "ي", "ک": "ك", "ی": "ي"})


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm_token(tok: str) -> str:
    t = tok.lower()
    t = _AR_DIAC.sub("", t)
    t = t.translate(_AR_FIX)
    t = strip_accents(t)
    t = re.sub(r"([^\W\d_])\1{2,}", r"\1\1", t)  # kayyyyn -> kayyn (lettres seulement)
    return t.strip("._")


@dataclass
class Doc:
    raw: str
    orig: list      # tokens d'origine (pour l'affichage)
    tokens: list    # tokens normalisés (alignés 1:1 avec orig)

    @property
    def norm(self) -> str:
        return " ".join(self.tokens)

    def text(self, i: int, j: int) -> str:
        return " ".join(self.orig[i:j])


def make_doc(text: str) -> Doc:
    t = unicodedata.normalize("NFKC", text or "").translate(_AR_DIGITS)
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)
    orig, toks = [], []
    for m in re.finditer(r"[\w.]+", t):
        o = m.group().strip("._")
        n = norm_token(o)
        if n:
            orig.append(o)
            toks.append(n)
    return Doc(text, orig, toks)


def is_arabic(s: str) -> bool:
    return bool(AR_RE.search(s))


def detect_script(text: str) -> str:
    ar, la = len(AR_RE.findall(text)), len(LAT_RE.findall(text))
    if ar == 0 and la == 0:
        return "latin"
    if ar and la and min(ar, la) / (ar + la) >= 0.2:
        return "mixed"
    return "arabic" if ar > la else "latin"


# ------------------------------------------------------------------ squelette
_AR_SKEL = {
    **{c: "" for c in "اءويهعىةؤئ"},
    "ب": "b", "پ": "b", "ت": "t", "ث": "t", "ط": "t", "ج": "j", "چ": "j", "ح": "", "خ": "k",
    "د": "d", "ذ": "d", "ض": "d", "ظ": "d", "ر": "r", "ز": "s", "س": "s", "ش": "s", "ص": "s",
    "غ": "k", "ق": "k", "ك": "k", "گ": "k", "ف": "f", "ڤ": "f", "ل": "l", "م": "m", "ن": "n",
}
_LAT_DIGRAPHS = [("ph", "f"), ("ch", "s"), ("sh", "s"), ("kh", "k"), ("th", "t"),
                 ("ck", "k"), ("qu", "k"), ("gn", "n")]


def skeleton(s: str) -> str:
    """Clé phonétique consonantique commune à l'arabe et au latin."""
    s = norm_token(s)
    for a, b in _LAT_DIGRAPHS:
        s = s.replace(a, b)
    s = re.sub(r"c(?=[eiy])", "s", s)
    out = []
    for ch in s:
        if ch in _AR_SKEL:
            out.append(_AR_SKEL[ch])
        elif ch in "aeiouyhw":
            continue
        elif ch in "bp":
            out.append("b")
        elif ch in "fv":
            out.append("f")
        elif ch in "ckqg9":
            out.append("k")
        elif ch in "sz":
            out.append("s")
        elif ch == "x":
            out.append("ks")
        elif ch.isalpha():
            out.append(ch)
        # chiffres (sauf 9) ignorés
    return re.sub(r"(.)\1+", r"\1", "".join(out))
