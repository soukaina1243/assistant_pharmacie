"""Tests du Challenge 2 (reconnaissance de boîtes). Ignorés si Tesseract n'est pas installé."""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import make_box_dataset as mb  # noqa: E402

from app.main import app  # noqa: E402
from app.vision import preprocess as pp  # noqa: E402
from app.vision.ocr import Word, ocr_available  # noqa: E402
from app.vision.recognize import BoxRecognizer  # noqa: E402

needs_ocr = pytest.mark.skipif(not ocr_available(), reason="Tesseract non installé")


def make_box(brand="DOLIPRANE", rot=0.0, seed=1):
    rows = [r for r in mb.pick_products("dev", 400, 1)] + [r for r in mb.pick_products("test", 400, 1)]
    row = next(r for r in rows if r["brand"] == brand)
    rng = random.Random(seed)
    img, _ = mb.scene_with_box(mb.render_face(row, rng), rng, rot=rot, persp=0.02)
    return row, img


@pytest.fixture(scope="module")
def rec():
    from app.data import Store
    from app.matcher import MedicationMatcher
    return BoxRecognizer(MedicationMatcher(Store().meds))


def test_quality_metrics_dark_and_blur():
    dark = np.full((600, 800, 3), 20, np.uint8)
    assert "image_sombre" in pp.quality_metrics(dark)["warnings"]
    noisy = np.random.default_rng(0).integers(0, 255, (600, 800, 3), dtype=np.uint8)
    assert "image_floue" in pp.quality_metrics(cv2.GaussianBlur(noisy, (0, 0), 15))["warnings"]


def test_skew_estimation_and_rotation():
    _, img = make_box("DOLIPRANE", rot=14.0)
    sk = pp.estimate_skew(img)
    assert 8 <= abs(sk) <= 20


def test_truncated_name_uses_prefix(rec):
    """Boîte partielle : « CLAMO » (début de CLAMOXYL) doit être rapproché de CLAMOXYL, pas d'une marque voisine."""
    w = [Word("CLAMO", 92, 0, 0, 400, 120, (1, 1, 1), "latin", "stretch", 0, 0)]
    rec._cache = {}
    ev = rec._evidence(w)
    assert ev[("CLAMOXYL", "brand")]["e"] > 0.8


def test_dci_is_not_taken_as_brand(rec):
    w = [Word("Dexamethasone", 92, 0, 0, 300, 40, (1, 1, 2), "latin", "stretch", 0, 0),
         Word("DECADRON", 95, 0, 0, 600, 120, (1, 1, 1), "latin", "stretch", 0, 0)]
    rec._cache = {}
    ev = rec._evidence(w)
    assert ev[("DECADRON", "brand")]["e"] > ev.get(("DEXAMETHASONE MYLAN", "brand"), {"e": 0})["e"]


@needs_ocr
@pytest.mark.parametrize("brand", ["DOLIPRANE", "SPASFON"])
def test_recognize_clean_box(rec, brand):
    row, img = make_box(brand)
    res = rec.recognize(img)
    assert res["matches"] and res["matches"][0]["brand"] == brand
    assert res["status"] in {"ok", "low_confidence"}


@needs_ocr
def test_no_text_image(rec):
    res = rec.recognize(np.full((500, 700, 3), 200, np.uint8))
    assert res["status"] == "no_text" and res["matches"] == []


@needs_ocr
def test_box_api():
    _, img = make_box("DOLIPRANE")
    ok, buf = cv2.imencode(".jpg", img)
    with TestClient(app) as c:
        assert c.get("/health").json()["ocr_available"] is True
        r = c.post("/recognize/box?lang=fr", files={"file": ("box.jpg", buf.tobytes(), "image/jpeg")})
        assert r.status_code == 200 and r.json()["matches"][0]["brand"] == "DOLIPRANE" and r.json()["reply"]
        bad = c.post("/recognize/box", files={"file": ("x.jpg", b"pas une image", "image/jpeg")})
        assert bad.status_code == 400
