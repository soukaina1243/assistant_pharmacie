# Assistant pharmacie Darija / Arabe / Français

Transforme une demande de patient (texte ou voix, en arabe, Darija, arabizi ou français) en **interprétation structurée JSON**
(intention, médicaments, dosage, quantité, pharmacie, ville…), exposée par une **API FastAPI**.

## Démarrage rapide

```bash
pip install -r requirements.txt
# les données prétraitées sont déjà fournies ; pour tout régénérer :
python scripts/prepare_data.py            # référentiel CNOPS -> data/processed/medications.csv
python scripts/make_demo_data.py          # pharmacies + stock SIMULÉS -> data/demo/
python scripts/generate_synthetic.py      # corpus synthétique train / test difficile
python scripts/train_intent.py            # -> models/intent.joblib

uvicorn app.main:app --reload             # interface de chat : http://127.0.0.1:8000/   |   API : /docs
python -m app.cli "wach kayn doliprane 1000 f rabat ?"
python -m pytest -q
python scripts/evaluate.py -v             # métriques (rules / ml / hybrid) sur data/corpus_pharmacie.jsonl
python scripts/evaluate.py --corpus data/synthetic/test_hard.jsonl
```

## Interface de chat (texte + voix)

Ouvrez **http://127.0.0.1:8000/** : un chat où l'on écrit ou parle. Chaque réponse affiche la réponse en langage naturel
(dans la langue de l'utilisateur), des pastilles d'interprétation (intention, médicament, ville…), les cartes pharmacies/prix
et le JSON complet (dépliable).

- **Micro 🎤** : reconnaissance du navigateur (Web Speech API, Chrome ou Edge, connexion internet requise, `localhost` autorisé).
  Choisir la langue de la voix (Darija `ar-MA`, français, arabe). Aucun module à installer.
- **Reconnaissance serveur (Whisper)** : cochez l'option après `pip install faster-whisper` (enregistrement → `/analyze/voice`).
- **Lecture vocale** de la réponse (option) : voix du navigateur ; la lecture de Darija en lettres latines est médiocre.
- Confidentialité : avec la reconnaissance du navigateur, l'audio est traité par le service du navigateur (Google pour Chrome).

## Challenge 2 : reconnaissance d'une boîte de médicament (vision + OCR)

Le bouton **📷** du chat (ou `POST /recognize/box`) prend la photo d'une boîte, lit son texte (français et arabe) et retrouve le produit
dans le référentiel : marque, dosage, forme, DCI, prix, remboursement. Le résultat se branche sur le chatbot texte
(boutons « Prix », « Remboursement », « Disponibilité »).

```bash
curl -X POST "localhost:8000/recognize/box?lang=fr&top_k=3" -F file=@boite.jpg
```

### Installer l'OCR (obligatoire pour ce challenge)
`pip install -r requirements.txt` installe OpenCV et `pytesseract`, mais il faut aussi le **programme Tesseract** avec l'arabe et le français :

| Système | Commande |
|---|---|
| Windows | installeur UB-Mannheim (github.com/UB-Mannheim/tesseract/wiki), cocher les langues *Arabic* et *French*, puis `set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` |
| macOS | `brew install tesseract tesseract-lang` |
| Ubuntu / Debian | `sudo apt install tesseract-ocr tesseract-ocr-ara tesseract-ocr-fra` |

Vérifier : `tesseract --list-langs` doit afficher `ara` et `fra`. Sans Tesseract, le reste du projet fonctionne ; le bouton 📷 est grisé.

### Pipeline
```
photo → qualité (flou, luminosité, reflets) → normalisation de taille et d'exposition
      → OCR Tesseract (latin + arabe) avec essais successifs : image réduite, redressement de la boîte,
        correction d'inclinaison (Hough), variantes binaire / inversée / nette, rotations 90-180-270
      → preuves « nom du produit » : tokens lus × (appariement flou + clé phonétique) × confiance OCR × taille du texte
      → croisement avec dosage, forme, conditionnement et DCI lus sur la boîte
      → classement des produits du référentiel → statut : ok | low_confidence | unknown | no_text
```
Idées clés (code dans `app/vision/`) :
- **L'OCR s'arrête quand le produit est certain**, pas quand il a lu « du texte » (sinon du charabia suffit à s'arrêter).
- **Taille du texte** : le nom commercial est le texte le plus grand ; la DCI imprimée en petit pèse moins.
- **Cohérence marque / DCI** : une DCI lue avec certitude pénalise une marque incompatible (un antifongique n'est pas du paracétamol).
- **Noms tronqués** (boîte partiellement visible) : un début ou une fin de nom (`CLAMO`) est rapproché de la marque (`CLAMOXYL`).
- **L'arabe seul ne donne jamais « ok »** : les mots arabes courts collisionnent avec des marques (on ignore aussi « أقراص »…).
- **Le système préfère dire « non identifié » à se tromper** : `ok` n'est donné que si le score et l'écart avec le concurrent sont suffisants.

### Jeu de données (construit, SYNTHÉTIQUE)
`scripts/make_box_dataset.py` dessine, pour chaque produit du référentiel, une boîte (marque, DCI, dosage, forme, texte arabe translittéré,
code-barres et logo parasites), la pose en perspective sur un fond, puis applique 9 conditions : `clean, blur, dark, glare, rot15, rot90, partial, lowres, hard`.
```bash
python scripts/make_box_dataset.py --split dev  --n 20 --seed 1     # réglage
python scripts/make_box_dataset.py --split test --n 30 --seed 2     # test : marques DIFFÉRENTES de dev
```
`data/boxes/sample/` contient 4 produits × 9 conditions pour essayer tout de suite. Le rendu dépend des polices de votre système
(l'arabe exige Pillow avec libraqm : plutôt Linux / WSL / Colab) ; les images régénérées ne seront donc pas identiques à celles du rapport.

### Évaluation
```bash
python scripts/eval_boxes.py --split test         # reprend son cache si interrompu
python scripts/eval_boxes.py --labels data/boxes_real/labels.jsonl     # VOS vraies photos
```
Format d'une ligne `labels.jsonl` : `{"file": "photo1.jpg", "condition": "reel", "brand": "DOLIPRANE", "dosage_mg": 500.0}` (chemin relatif au dossier du fichier).

**Résultat officiel : jeu de test, 270 images, 30 marques jamais vues pendant le réglage** (détail : `reports/BOX_RESULTS.md`)

| Condition | Marque top-1 | Produit top-1 (marque+dosage) | Précision quand « ok » |
|---|---|---|---|
| clean | 100 % | 96,7 % | 100 % |
| blur | 83,3 % | 80,0 % | 100 % |
| dark | 96,7 % | 93,3 % | 100 % |
| glare | 90,0 % | 83,3 % | 100 % |
| rot15 | 86,7 % | 80,0 % | 96,2 % |
| rot90 | 90,0 % | 83,3 % | 100 % |
| **partial** | **66,7 %** | 63,3 % | 100 % |
| lowres | 100 % | 96,7 % | 100 % |
| hard | 90,0 % | 83,3 % | 100 % |
| **Ensemble** | **89,3 %** (top-3 : 91,1 %) | **84,4 %** | **99,5 %** |

Le système répond « ok » dans 80 % des cas ; **1 seule erreur confiante sur 270** (0,4 %). Temps : médiane 1,2 s, 90ᵉ percentile 5,4 s,
mais **jusqu'à 21 s** sur des images très dégradées (CPU seul, 1 cœur).

**À lire honnêtement** : ces images sont des rendus propres et simples (polices nettes, mise en page régulière), pas de vraies photos :
les chiffres sont **optimistes**. Les vraies boîtes ont des reflets, des courbures, des logos, des polices décoratives et des fonds chargés.
Le point faible mesuré est la boîte partiellement visible (66,7 %) : si le nom est coupé, il n'y a rien à lire. Le jeu de réglage (dev) a servi
à trouver 5 défauts réels (mots d'emballage arabes pris pour des marques, DCI prise pour un début de marque, arrêt trop précoce de l'OCR,
dosage non lu, inclinaison) ; le jeu de test n'a été évalué qu'après.

### Limites et pistes
Pas de vraies photos (à collecter : ~50-100 boîtes, 3 conditions) ; OCR classique (Tesseract), non entraîné sur des emballages ; pas de
comparaison visuelle (logo, couleurs) entre produits similaires : seul le texte distingue deux produits (d'où l'importance du dosage) ;
pistes : détecteur de boîte (YOLO), OCR neuronal (PaddleOCR / EasyOCR), embeddings visuels (CNN) pour départager des emballages proches,
affinage des seuils sur de vraies photos.

## Architecture

```
Texte ─────────────────────────────┐
Voix ─► ASR (faster-whisper) ──────┤
                                   ▼
        normalisation + langue/graphie (arabe | arabizi | français)
                                   ▼
        intention : règles ⊕ TF-IDF+LogReg (hybride)
                                   ▼
        entités : dosage, forme, quantité, ville, quartier, pharmacie, temps
                                   ▼
        médicament : alias → exact → distance texte + « squelette phonétique » (arabe ⇄ latin)
                                   ▼
        validation (champs obligatoires par intention) → ok | needs_clarification | out_of_scope
                                   ▼
        JSON + « fulfillment » (prix, remboursement, stock, infos pharmacie)
```

**Idée clé : le squelette phonétique** (`app/textnorm.py::skeleton`). `دوليبران`, `doliprane` et `dolipran` donnent tous
`dlbrn` : on compare des consonnes normalisées, ce qui absorbe translittération, voyelles omises et fautes de frappe.

| Fichier | Rôle |
|---|---|
| `app/textnorm.py` | normalisation, détection de graphie, squelette phonétique |
| `app/lexicon.py` | villes, formes, mots-outils, motifs d'intention |
| `app/intent.py` | classifieur règles / ML / hybride |
| `app/entities.py` | extraction d'entités par règles |
| `app/matcher.py` | rapprochement flou avec le référentiel (5 917 produits, 2 403 marques) |
| `app/lookup.py` | réponses : prix, remboursement, disponibilité, infos pharmacie |
| `app/pipeline.py` | orchestration + schéma de sortie |
| `app/main.py` | API : `/analyze/text`, `/analyze/voice`, `/medications/search`, `/pharmacies`, `/health` |
| `scripts/` | préparation des données, génération synthétique, entraînement, évaluation (`evaluate.py`, `run_all_evals.py`, `eval_asr.py`), vraies pharmacies (`build_pharmacies.py`) |
| `data/eval/` | jeu de mise au point (69) et jeu final (50) |
| `docs/` | `rapport.tex` (brouillon de rapport avec état de l'art et analyse d'erreurs), `schema_json.tex` |

## API

```bash
curl -X POST localhost:8000/analyze/text -H 'content-type: application/json' \
     -d '{"text": "واش الأوغمنتين كيتعوض من الكنوبس؟"}'
curl -X POST localhost:8000/analyze/voice -F file=@message.wav     # nécessite faster-whisper
```

La sortie suit le schéma du rapport (`request_id`, `input`, `language`, `intent`, `entities`, `status`,
`clarification_question`, `warnings`) + `fulfillment` + `meta`.

## Résultats (mesurés) — détail dans `reports/RESULTS.md` (`python scripts/run_all_evals.py`)

Quatre jeux, tous écrits par l'auteur du système ou générés : aucun n'est composé de vrais utilisateurs.
**Jeu final (50 messages, écrit après la mise au point)** : hybride **96 % d'intention, 98 % de statut, 100 % médicament top-1**
avant toute correction ; 100 % après correction des 3 erreurs vues (ce chiffre-là est « contaminé », ne pas le citer comme test).

## Anciens résultats (premières versions)

Tous les jeux sont écrits par les auteurs du système ou générés : **aucun message d'utilisateur réel**.

| Jeu | Mode | Intention | Statut | Médicament top-1 | Dosage |
|---|---|---|---|---|---|
| **Final (50)**, écrit après la mise au point, évalué *avant* toute correction | règles | 94,0 % | 98,0 % | 100 % | 100 % |
| | ML seul | 86,0 % | 92,0 % | 96,8 % | 96,8 % |
| | **hybride** | **96,0 %** | **98,0 %** | **100 %** | **100 %** |
| Synthétique difficile (384), gabarits inédits + bruit | règles | 87,5 % | 92,2 % | 93,8 % | 97,9 % |
| | ML seul | 81,8 % | 90,9 % | 89,1 % | 93,2 % |
| | **hybride** | **93,2 %** | **94,8 %** | **91,7 %** | **95,8 %** |

Après correction des 3 défauts révélés par le jeu final, l'hybride y atteint 100 %, **mais ce chiffre n'est plus indépendant**
(les corrections en viennent). De même, 100 % sur le corpus initial (56) et le jeu de mise au point (69) illustre le risque de
sur-ajustement, pas la qualité réelle. Tableaux complets : `reports/RESULTS.md` (`python scripts/run_all_evals.py`).
Latence : 10,5 ms en moyenne (médiane 4,8 ms, p95 54 ms) sur CPU, hors transcription vocale.

**Erreurs typiques** : mots courants pris pour des marques (« temps » → TMS), formulations sans mot-clé, variantes
orthographiques non prévues (« tmn »), nom de pharmacie mal délimité (« dyalkom »), langue ambiguë sur un seul mot.
Analyse détaillée : `docs/rapport.tex` (section 6.3).

**La vraie mesure = ~100 messages réels** (écrits/dictés par des proches), jamais utilisés pour régler les règles.
Annotez-les au format de `data/corpus_pharmacie.jsonl` puis : `python scripts/evaluate.py --corpus data/mon_test.jsonl -v`.

## Voix : évaluer le WER

Placez des enregistrements et leurs transcriptions dans `data/audio/` (`msg01.wav` + `msg01.txt`), puis
`pip install faster-whisper` et `python scripts/eval_asr.py`. Le script donne le WER et le pourcentage de messages dont
l'analyse (intention + médicament) reste identique malgré les erreurs de transcription.

## Ce qui est réel, simulé, limité

- **Réel** : référentiel CNOPS (prix PPV, taux de remboursement, DCI, dosage, forme). ⚠ Édition **2014** : prix et taux indicatifs ;
  remplacez par le référentiel data.gov.ma plus récent (`python scripts/prepare_data.py nouveau_fichier.xlsx`).
- **Simulé** : stock par pharmacie (aucune source publique) et pharmacies de démo (`source = synthetic_demo`).
  Données réelles : `scripts/build_pharmacies.py` (AMMPS + OpenStreetMap, non testé contre les sites réels).
- **Non mesuré** : la transcription vocale (code prêt, `pip install faster-whisper`). Whisper gère mal la Darija :
  mesurez le WER sur vos enregistrements. `app/asr.py` injecte un vocabulaire du domaine pour aider.
- **Sécurité** : aucun conseil médical ; les demandes du type « que prendre pour la toux ? » sont `out_of_scope`.
- Les translittérations arabes du jeu synthétique sont générées par règles (même famille que le squelette) : optimiste.
  `data/aliases.json` permet d'ajouter des graphies réelles.

## Que faire du corpus JSONL ?

1. **Jeu de test (principal)** : `evaluate.py` le compare à la sortie du pipeline champ par champ. Ne l'utilisez pas pour régler les règles.
2. **Étendre** : ajoutez des lignes réelles (messages de proches), gardez ~100 lignes « jamais vues » pour le rapport.
3. **Entraîner** (optionnel) : `python scripts/train_intent.py --extra data/corpus_pharmacie.jsonl` — mais vous perdez alors ce jeu comme test.

## Pistes d'amélioration

LLM avec sortie JSON contrainte + validation par le référentiel (comparer à ce baseline) ; fine-tuning DarijaBERT/CAMeLBERT
(intention + NER) ; vraies données pharmacies ; mesure du WER Whisper et post-correction ; dialogue multi-tours (réponse à une
clarification) ; journalisation et métriques en production.

## Correspondance avec le cahier des charges

Étude et état de l'art → `docs/rapport.tex` (références vérifiées) · Architecture → ce README · NLP + parole → `app/` · API → `app/main.py` ·
Tests + évaluation → `tests/`, `scripts/evaluate.py`, `reports/`.
