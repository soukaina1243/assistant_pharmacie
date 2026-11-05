# Challenge 2 -- résultats de la reconnaissance de boîtes

Images SYNTHÉTIQUES (rendus propres, pas de vraies photos) : résultats optimistes. Voir README.

## Jeu de TEST (30 marques jamais vues pendant le réglage) -- résultat officiel

### Reconnaissance de boîtes -- jeu « test » (270 images)

| Condition | n | Marque top-1 | Marque top-3 | Produit top-1 (marque+dosage) | Réponses « ok » | Précision quand « ok » | Erreurs confiantes | Temps médian |
|---|---|---|---|---|---|---|---|---|
| clean | 30 | 100.0 % | 100.0 % | 96.7 % | 90.0 % | 100.0 % | 0.0 % | 0.7 s |
| blur | 30 | 83.3 % | 90.0 % | 80.0 % | 56.7 % | 100.0 % | 0.0 % | 0.9 s |
| dark | 30 | 96.7 % | 96.7 % | 93.3 % | 86.7 % | 100.0 % | 0.0 % | 0.8 s |
| glare | 30 | 90.0 % | 90.0 % | 83.3 % | 83.3 % | 100.0 % | 0.0 % | 1.1 s |
| rot15 | 30 | 86.7 % | 90.0 % | 80.0 % | 86.7 % | 96.2 % | 3.3 % | 1.4 s |
| rot90 | 30 | 90.0 % | 96.7 % | 83.3 % | 90.0 % | 100.0 % | 0.0 % | 2.7 s |
| partial | 30 | 66.7 % | 66.7 % | 63.3 % | 46.7 % | 100.0 % | 0.0 % | 1.3 s |
| lowres | 30 | 100.0 % | 100.0 % | 96.7 % | 96.7 % | 100.0 % | 0.0 % | 0.8 s |
| hard | 30 | 90.0 % | 90.0 % | 83.3 % | 83.3 % | 100.0 % | 0.0 % | 0.9 s |
| ALL | 270 | 89.3 % | 91.1 % | 84.4 % | 80.0 % | 99.5 % | 0.4 % | 1.2 s |

## Jeu de réglage (dev, 20 marques) -- mesuré pendant la mise au point, avant le dernier correctif (arrêt de l'OCR / dosage)

### Reconnaissance de boîtes -- jeu « dev » (180 images)

| Condition | n | Marque top-1 | Marque top-3 | Produit top-1 (marque+dosage) | Réponses « ok » | Précision quand « ok » | Erreurs confiantes | Temps médian |
|---|---|---|---|---|---|---|---|---|
| clean | 20 | 95.0 % | 95.0 % | 85.0 % | 90.0 % | 100.0 % | 0.0 % | 0.6 s |
| blur | 20 | 85.0 % | 85.0 % | 80.0 % | 60.0 % | 100.0 % | 0.0 % | 0.6 s |
| dark | 20 | 95.0 % | 95.0 % | 90.0 % | 95.0 % | 100.0 % | 0.0 % | 0.8 s |
| glare | 20 | 75.0 % | 85.0 % | 65.0 % | 75.0 % | 93.3 % | 5.0 % | 0.6 s |
| rot15 | 20 | 80.0 % | 85.0 % | 80.0 % | 70.0 % | 92.9 % | 5.0 % | 1.6 s |
| rot90 | 20 | 90.0 % | 95.0 % | 85.0 % | 80.0 % | 100.0 % | 0.0 % | 2.9 s |
| partial | 20 | 70.0 % | 75.0 % | 55.0 % | 45.0 % | 100.0 % | 0.0 % | 2.5 s |
| lowres | 20 | 100.0 % | 100.0 % | 100.0 % | 85.0 % | 100.0 % | 0.0 % | 0.6 s |
| hard | 20 | 85.0 % | 85.0 % | 75.0 % | 80.0 % | 100.0 % | 0.0 % | 0.7 s |
| ALL | 180 | 86.1 % | 88.9 % | 79.4 % | 75.6 % | 98.5 % | 1.1 % | 0.8 s |
