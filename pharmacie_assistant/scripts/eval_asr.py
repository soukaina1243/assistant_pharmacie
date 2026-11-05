"""Mesure le WER/CER de la transcription vocale et son effet sur l'analyse.

Dossier attendu : data/audio/  avec  xxx.wav (ou mp3/m4a/ogg)  +  xxx.txt (transcription correcte).
Optionnel : xxx.json = annotation complète (même format que le corpus) pour mesurer l'intention après ASR.

    pip install faster-whisper
    python scripts/eval_asr.py data/audio
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.textnorm import make_doc  # noqa: E402


def _edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = make_doc(ref).tokens, make_doc(hyp).tokens
    return _edit(r, h) / max(1, len(r))


def cer(ref: str, hyp: str) -> float:
    r, h = make_doc(ref).norm.replace(" ", ""), make_doc(hyp).norm.replace(" ", "")
    return _edit(r, h) / max(1, len(r))


if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/audio")
    from app.asr import transcribe
    from app.pipeline import Assistant
    asst = Assistant()
    rows, ok = [], 0
    for audio in sorted(p for p in folder.iterdir() if p.suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg", ".flac"}):
        ref_file = audio.with_suffix(".txt")
        if not ref_file.exists():
            continue
        ref = ref_file.read_text(encoding="utf-8").strip()
        hyp, lang = transcribe(str(audio))
        row = {"file": audio.name, "ref": ref, "hyp": hyp, "wer": round(wer(ref, hyp), 3), "cer": round(cer(ref, hyp), 3)}
        gold = audio.with_suffix(".json")
        if gold.exists():
            g = json.loads(gold.read_text(encoding="utf-8"))
            row["intent_ok_after_asr"] = asst.analyze(hyp, "voice", hyp, False)["intent"]["label"] == g["intent"]
        rows.append(row)
        print(row)
    if rows:
        n = len(rows)
        print(f"\n{n} fichiers | WER moyen {sum(r['wer'] for r in rows) / n:.3f} | CER moyen {sum(r['cer'] for r in rows) / n:.3f}")
        k = [r for r in rows if "intent_ok_after_asr" in r]
        if k:
            print(f"Intention correcte après ASR : {sum(r['intent_ok_after_asr'] for r in k)}/{len(k)}")
        (ROOT / "reports/asr_eval.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
