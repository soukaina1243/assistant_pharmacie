# Résultats d'évaluation

Généré par `scripts/run_all_evals.py`.


## Corpus initial (56) — écrit avec les règles

| Mode | Intention | Statut | Médic. top-1 | Dosage | Pharmacie | Ville | Temps | Langue | F1 macro intention |
|---|---|---|---|---|---|---|---|---|---|
| rules | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| ml | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| hybrid | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |

## Synthétique difficile (384) — gabarits inédits + bruit

| Mode | Intention | Statut | Médic. top-1 | Dosage | Pharmacie | Ville | Temps | Langue | F1 macro intention |
|---|---|---|---|---|---|---|---|---|---|
| rules | 87.5 % | 92.2 % | 93.8 % | 97.9 % | 94.4 % | 97.8 % | 96.1 % | 92.7 % | 86.7 % |
| ml | 81.8 % | 90.9 % | 89.1 % | 93.2 % | 98.9 % | 99.2 % | 97.5 % | 92.7 % | 80.5 % |
| hybrid | 93.2 % | 94.8 % | 91.7 % | 95.8 % | 98.3 % | 100.0 % | 97.8 % | 92.7 % | 92.9 % |

## Dev réaliste (69) — utilisé pour la mise au point

| Mode | Intention | Statut | Médic. top-1 | Dosage | Pharmacie | Ville | Temps | Langue | F1 macro intention |
|---|---|---|---|---|---|---|---|---|---|
| rules | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 98.6 % | 100.0 % |
| ml | 98.6 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 98.6 % | 99.1 % |
| hybrid | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 98.6 % | 100.0 % |

## Final (50) — écrit après la mise au point

| Mode | Intention | Statut | Médic. top-1 | Dosage | Pharmacie | Ville | Temps | Langue | F1 macro intention |
|---|---|---|---|---|---|---|---|---|---|
| rules | 98.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 96.0 % | 98.3 % |
| ml | 86.0 % | 92.0 % | 96.8 % | 96.8 % | 100.0 % | 100.0 % | 100.0 % | 96.0 % | 85.9 % |
| hybrid | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 96.0 % | 100.0 % |
