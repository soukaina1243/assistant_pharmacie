"""Aide à l'écriture de jeux d'évaluation annotés à la main (format identique au corpus)."""
import json
from pathlib import Path

LANG = {"dl": ("darija", "latin"), "fr": ("french", "latin"), "ar": ("darija", "arabic")}
EMPTY = {"medications": [], "quantity": None, "pharmacy_name": None, "city": None, "neighborhood": None, "time_reference": None}


def M(raw, name, dosage=None, form=None):
    return {"raw": raw, "name": name, "dosage": dosage, "form": form}


class Set:
    def __init__(self, prefix):
        self.rows, self.prefix = [], prefix

    def add(self, style, text, intent, meds=(), q=None, ph=None, city=None, nb=None, t=None, st="ok"):
        p, s = LANG[style]
        self.rows.append({
            "id": f"{self.prefix}{len(self.rows)+1:03d}", "text": text,
            "language": {"primary": p, "script": s, "code_mixed": p == "darija" and s == "latin"},
            "intent": intent,
            "entities": {"medications": list(meds), "quantity": q, "pharmacy_name": ph, "city": city,
                         "neighborhood": nb, "time_reference": t},
            "status": st})

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in self.rows)
        print(len(self.rows), "->", path)
