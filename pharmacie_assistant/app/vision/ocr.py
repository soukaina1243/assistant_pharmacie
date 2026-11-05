"""OCR (Tesseract) avec stratégie de robustesse : variantes d'image, orientation, redressement, arabe + latin."""
from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import preprocess as pp

Q_GOOD = 14.0      # qualité jugée suffisante : on arrête de chercher d'autres variantes/angles


class OCRUnavailable(RuntimeError):
    pass


def _tesseract():
    try:
        import pytesseract
    except ImportError as e:
        raise OCRUnavailable("pytesseract n'est pas installé : pip install pytesseract") from e
    cmd = os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
    if not cmd:
        raise OCRUnavailable("Tesseract introuvable. Installez-le (voir README) ou définissez TESSERACT_CMD.")
    pytesseract.pytesseract.tesseract_cmd = cmd
    return pytesseract


def ocr_available() -> bool:
    try:
        _tesseract()
        return True
    except OCRUnavailable:
        return False


@dataclass
class Word:
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int
    line: tuple
    script: str          # latin | arabic
    variant: str
    angle: float
    attempt: int = 0


def _pass(gray: np.ndarray, lang: str, script: str, variant: str, angle: float, psm: int = 11) -> list:
    t = _tesseract()
    d = t.image_to_data(gray, lang=lang, config=f"--oem 1 --psm {psm}", output_type=t.Output.DICT)
    out = []
    for i, txt in enumerate(d["text"]):
        txt = (txt or "").strip()
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            continue
        if not txt or conf < 0:
            continue
        out.append(Word(txt, conf, d["left"][i], d["top"][i], d["width"][i], d["height"][i],
                        (d["block_num"][i], d["par_num"][i], d["line_num"][i]), script, variant, angle))
    return out


def quality(words: list) -> float:
    """Somme (longueur du mot x confiance) des mots alphabétiques fiables : mesure « combien de vrai texte lu »."""
    return float(sum(min(len(w.text), 10) * w.conf / 100 for w in words if w.conf >= 50 and len(w.text) >= 3
                     and re.search(r"[A-Za-zÀ-ÿ\u0600-\u06FF]{3,}", w.text)))


def read_text(img: np.ndarray, enough=None, use_arabic: bool = True, allow_rectify: bool = True, budget_ms: int = 15000) -> dict:
    """Lit le texte d'une image de boîte.

    `enough(words) -> bool` est fourni par le reconnaisseur : l'OCR s'arrête dès que le nom du produit est
    reconnu avec certitude (et non dès qu'il a lu « du texte »). Sinon il essaie, dans l'ordre :
    image réduite -> boîte redressée (perspective) -> autres variantes -> autres orientations -> passe arabe.
    """
    t0 = time.perf_counter()
    base = pp.normalize_size(img)
    attempts, words_all = [], []
    state = {"best_q": -1.0, "cfg": ("stretch", 0, base), "best_attempt": 0}

    def run(image, variant, angle, lang="fra", script="latin", psm=11):
        ws = _pass(pp.variant(pp.rotate(image, angle), variant), lang, script, variant, angle, psm)
        idx = len(attempts)
        for w in ws:
            w.attempt = idx
        q = quality(ws)
        attempts.append({"variant": variant, "angle": angle, "lang": lang, "quality": round(q, 1)})
        words_all.extend(ws)
        if script == "latin" and q > state["best_q"]:
            state.update(best_q=q, cfg=(variant, angle, image), best_attempt=idx)

    def done():
        return (enough is not None and enough(words_all)) or (time.perf_counter() - t0) * 1000 > budget_ms

    run(base, "stretch", 0)
    if not done():
        run(base, "half", 0)
    skew = pp.estimate_skew(base)
    if not done() and abs(skew) >= 3:                      # photo penchée : on redresse d'après les lignes dominantes
        run(base, "stretch", -skew)
    if not done():
        quad = pp.find_box_quad(base) if allow_rectify else None
        if quad is not None:
            rect = pp.rectify(base, quad)
            for a in (0, 90, 270):
                run(rect, "stretch", a)
                if done():
                    break
    if not done():
        for v, a in [("binary", 0), ("inverted", 0), ("sharp", 0), ("stretch", 90), ("stretch", 270), ("stretch", 180),
                     ("stretch", -12), ("stretch", 12)]:
            run(base, v, a)
            if done():
                break
    if use_arabic and not (enough is not None and enough(words_all)):
        v, a, image = state["cfg"]
        run(image, v, a, lang="ara", script="arabic")
    return {"words": words_all, "angle": state["cfg"][1], "variant": state["cfg"][0], "quality": round(max(state["best_q"], 0), 1),
            "best_attempt": state["best_attempt"], "skew": round(skew, 1), "attempts": attempts, "time_ms": round((time.perf_counter() - t0) * 1000)}
