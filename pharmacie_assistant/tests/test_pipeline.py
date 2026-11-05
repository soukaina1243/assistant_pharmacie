import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pipeline import Assistant
from app.textnorm import skeleton


@pytest.fixture(scope="module")
def asst():
    return Assistant()


def test_skeleton_cross_script():
    assert skeleton("دوليبران") == skeleton("doliprane") == skeleton("dolipran")


@pytest.mark.parametrize("text,intent,med", [
    ("wach kayn doliprane 1000 f rabat ?", "check_availability", "DOLIPRANE"),
    ("واش كاين دوليبران 1000؟", "check_availability", "DOLIPRANE"),
    ("chhal taman augmentin 1g ?", "price", "AUGMENTIN"),
    ("Est-ce que le Voltarène est remboursable par la CNOPS ?", "reimbursement", "VOLTARENE"),
    ("بغيت نحجز 2 علب د أوغمنتين", "reserve_order", "AUGMENTIN"),
])
def test_medication_requests(asst, text, intent, med):
    r = asst.analyze(text)
    assert r["intent"]["label"] == intent
    assert r["entities"]["medications"][0]["name"] == med
    assert r["status"] == "ok"


def test_dosage_and_quantity(asst):
    r = asst.analyze("bghit n7ejez 3 dyal doliprane 1000")
    assert r["entities"]["quantity"] == 3
    assert r["entities"]["medications"][0]["dosage"] == "1000 mg"


def test_clarification_and_out_of_scope(asst):
    assert asst.analyze("بغيت نحجز")["status"] == "needs_clarification"
    r = asst.analyze("Quel médicament dois-je prendre pour mon mal de tête ?")
    assert r["status"] == "out_of_scope" and r["entities"]["medications"] == []


def test_pharmacy_entities(asst):
    r = asst.analyze("3tini numro dyal pharmacie Ibn Sina f Rabat")
    assert r["intent"]["label"] == "pharmacy_phone"
    assert r["entities"]["pharmacy_name"] == "Ibn Sina" and r["entities"]["city"] == "RABAT"


def test_api():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/analyze/text", json={"text": "chhal taman doliprane 500 ?"})
        assert r.status_code == 200 and r.json()["fulfillment"]["type"] == "price"
        assert c.post("/analyze/text", json={"text": ""}).status_code == 422
        assert c.get("/medications/search", params={"q": "augmantin"}).json()[0]["name"] == "AUGMENTIN"


def test_wer_cer():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from eval_asr import cer, wer
    assert wer("wach kayn doliprane", "wach kayn doliprane") == 0
    assert abs(wer("wach kayn doliprane", "wach kayn") - 1 / 3) < 1e-9
    assert cer("abc", "abd") > 0


def test_ui_and_reply():
    with TestClient(app) as c:
        page = c.get("/")
        assert page.status_code == 200 and "Assistant pharmacie" in page.text
        r = c.post("/analyze/text", json={"text": "chhal taman doliprane 500 ?"}).json()
        assert r["reply"] and "dh" in r["reply"]
        assert "asr_available" in c.get("/health").json()
