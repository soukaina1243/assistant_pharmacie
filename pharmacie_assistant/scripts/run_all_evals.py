"""Lance tous les jeux d'évaluation et écrit reports/RESULTS.md."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from app.data import Store  # noqa: E402
from app.pipeline import Assistant  # noqa: E402
from evaluate import evaluate, load  # noqa: E402

SETS = [("Corpus initial (56) — écrit avec les règles", "data/corpus_pharmacie.jsonl"),
        ("Synthétique difficile (384) — gabarits inédits + bruit", "data/synthetic/test_hard.jsonl"),
        ("Dev réaliste (69) — utilisé pour la mise au point", "data/eval/dev_realistic.jsonl"),
        ("Final (50) — écrit après la mise au point", "data/eval/test_final.jsonl")]
COLS = [("intent", "Intention"), ("status", "Statut"), ("med_top1", "Médic. top-1"), ("dosage", "Dosage"),
        ("pharmacy_name", "Pharmacie"), ("city", "Ville"), ("time_reference", "Temps"), ("lang_primary", "Langue")]

store = Store()
lines = ["# Résultats d'évaluation\n", "Généré par `scripts/run_all_evals.py`.\n"]
for title, path in SETS:
    rows = load(ROOT / path)
    lines += [f"\n## {title}\n", "| Mode | " + " | ".join(c[1] for c in COLS) + " | F1 macro intention |",
              "|---|" + "---|" * (len(COLS) + 1)]
    for mode in ["rules", "ml", "hybrid"]:
        res, _ = evaluate(rows, Assistant(intent_mode=mode, store=store))
        lines.append(f"| {mode} | " + " | ".join(f"{100 * res.get(k, float('nan')):.1f} %" for k, _ in COLS)
                     + f" | {100 * res['intent_macro_f1']:.1f} % |")
(ROOT / "reports/RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
