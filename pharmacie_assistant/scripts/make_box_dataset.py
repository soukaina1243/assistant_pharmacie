"""Génère un jeu d'images de BOÎTES de médicaments (synthétiques) avec vérité terrain.

Chaque produit du référentiel est dessiné (nom de marque, DCI, dosage, forme, texte arabe, distracteurs), posé en
perspective sur un fond, puis dégradé selon une CONDITION : clean, blur, dark, glare, rot15, rot90, partial, lowres, hard.

    python scripts/make_box_dataset.py --split dev  --n 40 --seed 1
    python scripts/make_box_dataset.py --split test --n 40 --seed 2

Dev et test utilisent des marques DIFFÉRENTES (partition fixe) : on règle sur dev, on mesure sur test.
LIMITE : ce sont des rendus propres (polices, mise en page simples), pas de vraies photos : résultats optimistes.
"""
import argparse
import json
import math
import random
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/boxes"
CONDITIONS = ["clean", "blur", "dark", "glare", "rot15", "rot90", "partial", "lowres", "hard"]


def _font(spec):
    try:
        return subprocess.check_output(["fc-match", "-f", "%{file}", spec]).decode()
    except Exception:
        return None


_FB_LATIN = ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/timesbd.ttf", "C:/Windows/Fonts/verdanab.ttf",
             "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
_FB_ARABIC = ["C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
LATIN_FONTS = sorted(f for f in {_font(s) for s in ["DejaVu Sans:bold", "FreeSans:bold", "FreeSerif:bold", "Carlito:bold",
                                                    "Bitstream Charter:bold", "Caladea:bold", "DejaVu Serif:bold"]} | set(_FB_LATIN)
                     if f and Path(f).exists())
ARABIC_FONTS = sorted(f for f in {_font("DejaVu Sans"), _font("FreeSerif"), _font("FreeSans")} | set(_FB_ARABIC) if f and Path(f).exists())
from PIL import features as _pf  # noqa: E402
HAS_RAQM = _pf.check("raqm")   # sans libraqm, l'arabe ne peut pas être rendu correctement : la ligne arabe est alors omise
if not LATIN_FONTS:
    raise SystemExit("Aucune police TrueType trouvée : installez DejaVu Sans / Arial ou éditez _FB_LATIN dans ce script.")
PALETTES = [((255, 255, 255), (0, 122, 94)), ((245, 245, 250), (30, 70, 160)), ((255, 248, 235), (200, 60, 40)),
            ((235, 250, 245), (0, 100, 120)), ((250, 240, 250), (120, 40, 140)), ((30, 40, 70), (250, 200, 60)),
            ((250, 250, 250), (20, 20, 20)), ((255, 236, 210), (230, 120, 20))]
LABS = ["LABORATOIRES SOTHEMA", "PHARMA 5", "LABORATOIRES AFRIC-PHAR", "MARKETING PHARMA", "COOPER PHARMA", "BIOPHARMA MAROC"]
FORM_AR = {"COMPRIME": "أقراص", "GELULE": "كبسولات", "SIROP": "شراب", "SACHET": "أكياس", "INJECTABLE": "حقن", "AMPOULE": "حقن",
           "SUPPOSITOIRE": "تحاميل", "COLLYRE": "قطرة للعين", "POMMADE": "مرهم", "CREME": "كريم", "GEL": "جيل"}
FORM_FR = {"COMPRIME": "comprimés", "GELULE": "gélules", "SIROP": "sirop", "SACHET": "sachets", "INJECTABLE": "solution injectable",
           "AMPOULE": "ampoules", "SUPPOSITOIRE": "suppositoires", "COLLYRE": "collyre", "POMMADE": "pommade", "CREME": "crème", "GEL": "gel"}
_L2A = {"b": "ب", "c": "ك", "d": "د", "f": "ف", "g": "غ", "h": "ه", "j": "ج", "k": "ك", "l": "ل", "m": "م", "n": "ن", "p": "ب",
        "q": "ق", "r": "ر", "s": "س", "t": "ت", "v": "ف", "w": "و", "x": "كس", "z": "ز"}


def ar_translit(w, rng):
    w = w.lower()
    for a, b in [("ch", "ش"), ("sh", "ش"), ("ph", "ف"), ("th", "ت"), ("kh", "خ"), ("ou", "و")]:
        w = w.replace(a, b)
    out = []
    for i, ch in enumerate(w):
        if ch in "aeiouy":
            if i == 0:
                out.append("ا" if ch in "ae" else "أو")
            elif rng.random() < 0.8:
                out.append({"a": "ا", "e": "ي", "i": "ي", "o": "و", "u": "و", "y": "ي"}[ch])
        else:
            out.append(_L2A.get(ch, ch))
    return "".join(out)


# ------------------------------------------------------------------ choix des produits
def pick_products(split, n, seed):
    meds = pd.read_csv(ROOT / "data/processed/medications.csv").fillna("")
    ok = meds[(meds["dosage_mg"].astype(str) != "") & (meds["forme"] != "") & meds["brand"].str.fullmatch(r"[A-Z][A-Z ]{4,15}")]
    brands = sorted(ok["brand"].unique())
    random.Random(0).shuffle(brands)
    mine = brands[0::2] if split == "dev" else brands[1::2]
    for b in ["DOLIPRANE", "AUGMENTIN", "SPASFON", "EFFERALGAN", "SMECTA", "GAVISCON", "VOLTARENE", "ZYRTEC", "CLAMOXYL", "MAXILASE"]:
        if b in mine:
            mine.remove(b); mine.insert(0, b)
    rng = random.Random(seed)
    rows = []
    for b in mine:
        sub = ok[ok["brand"] == b]
        if len(sub):
            rows.append(sub.iloc[rng.randrange(len(sub))])
        if len(rows) >= n:
            break
    return rows


def fit_font(path, text, max_w, max_size, min_size=18, rtl=False):
    mk = (lambda s: ImageFont.truetype(path, s, layout_engine=ImageFont.Layout.RAQM)) if rtl else (lambda s: ImageFont.truetype(path, s))
    lo, hi, best = min_size, max_size, min_size
    while lo <= hi:
        mid = (lo + hi) // 2
        f = mk(mid)
        if f.getbbox(text)[2] - f.getbbox(text)[0] <= max_w:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return mk(best)


def pack_text(row):
    forme = str(row["forme"])
    key = next((k for k in FORM_FR if k in forme), None)
    nums = re.findall(r"\d+", str(row["presentation"]))
    n = nums[1] if len(nums) > 1 else (nums[0] if nums else "")
    word = FORM_FR.get(key, forme.lower())
    return (f"Boîte de {n} {word}" if n else f"Boîte de {word}"), key


def render_face(row, rng, W=900, H=620):
    bg, accent = rng.choice(PALETTES)
    ink = (255, 255, 255) if sum(bg) < 300 else (20, 20, 20)
    face = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(face)
    d.rectangle([0, 0, W, int(H * 0.34)], fill=accent)
    band_ink = (255, 255, 255) if sum(accent) < 450 else (15, 15, 15)
    brand = str(row["brand"])
    f = fit_font(rng.choice(LATIN_FONTS), brand, W * 0.84, int(H * 0.26))
    bb = f.getbbox(brand)
    d.text(((W - (bb[2] - bb[0])) / 2 - bb[0], H * 0.17 - (bb[3] - bb[1]) / 2 - bb[1]), brand, font=f, fill=band_ink)
    y = H * 0.38
    dci = str(row["dci"]).title()
    if dci:
        f2 = fit_font(rng.choice(LATIN_FONTS), dci, W * 0.8, int(H * 0.075))
        d.text((W * 0.08, y), dci, font=f2, fill=ink); y += H * 0.1
    v = float(row["dosage"])
    dos = f"{int(v) if v.is_integer() else v} {str(row['unite']).lower()}"
    f3 = fit_font(rng.choice(LATIN_FONTS), dos, W * 0.35, int(H * 0.13))
    d.rounded_rectangle([W * 0.07, y, W * 0.07 + (f3.getbbox(dos)[2] + 40), y + f3.getbbox(dos)[3] + 24], 14, outline=accent, width=5)
    d.text((W * 0.07 + 20, y + 8), dos, font=f3, fill=ink); y += H * 0.19
    ptxt, key = pack_text(row)
    f4 = fit_font(rng.choice(LATIN_FONTS), ptxt, W * 0.8, int(H * 0.06))
    d.text((W * 0.08, y), ptxt, font=f4, fill=ink); y += H * 0.085
    if HAS_RAQM and ARABIC_FONTS:
        af = fit_font(rng.choice(ARABIC_FONTS), "أ" * 8, W * 0.4, int(H * 0.07), rtl=True)
        d.text((W * 0.92, y), ar_translit(brand.split()[0], rng) + "  " + FORM_AR.get(key, ""), font=af, fill=ink,
               anchor="ra", direction="rtl", language="ar")
    f5 = ImageFont.truetype(rng.choice(LATIN_FONTS), int(H * 0.035))
    d.text((W * 0.08, H * 0.93), rng.choice(LABS) + "  -  Voie orale", font=f5, fill=ink)
    for i in range(rng.randint(8, 18)):                       # faux code-barres (distracteur)
        x = W * 0.7 + i * 9
        d.rectangle([x, H * 0.72, x + rng.choice([3, 5, 7]), H * 0.9], fill=ink)
    d.ellipse([W * 0.78, H * 0.42, W * 0.93, H * 0.42 + W * 0.15], outline=accent, width=8)
    return face


def scene_with_box(face, rng, rot=0.0, persp=0.04, out=(1280, 960)):
    Wf, Hf = face.size
    bgc = rng.choice([(180, 160, 140), (220, 220, 225), (90, 100, 110), (200, 190, 170), (60, 60, 60)])
    scene = np.full((out[1], out[0], 3), bgc, np.uint8)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 6, scene.shape)
    grad = np.linspace(-25, 25, out[0])[None, :, None]
    scene = np.clip(scene + noise + grad, 0, 255).astype(np.uint8)
    scale = rng.uniform(0.9, 1.05) * min(out[0] * 0.82 / Wf, out[1] * 0.82 / Hf)
    w, h = Wf * scale, Hf * scale
    cx, cy = out[0] / 2 + rng.uniform(-30, 30), out[1] / 2 + rng.uniform(-30, 30)
    pts = np.float32([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    a = math.radians(rot)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pts = (pts @ R.T) + [cx, cy]
    pts += np.float32([[rng.uniform(-persp, persp) * w, rng.uniform(-persp, persp) * h] for _ in range(4)])
    src = np.float32([[0, 0], [Wf, 0], [Wf, Hf], [0, Hf]])
    M = cv2.getPerspectiveTransform(src, pts.astype(np.float32))
    warped = cv2.warpPerspective(np.array(face)[:, :, ::-1], M, out, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(np.full((Hf, Wf), 255, np.uint8), M, out)
    shadow = cv2.GaussianBlur(np.roll(mask, (14, 14), (0, 1)), (0, 0), 14)
    scene = (scene * (1 - 0.35 * shadow[..., None] / 255.0)).astype(np.uint8)
    scene = np.where(mask[..., None] > 0, warped, scene)
    return scene, pts


def apply_condition(img, pts, cond, rng):
    img = img.copy()
    h, w = img.shape[:2]
    if cond in ("blur", "hard"):
        if rng.random() < 0.5 or cond == "hard":
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(2.5, 4.0) if cond == "blur" else rng.uniform(1.8, 2.6))
        else:
            k = rng.randint(13, 21); ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
            M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
            img = cv2.filter2D(img, -1, cv2.warpAffine(ker, M, (k, k)))
    if cond in ("dark", "hard"):
        g = 0.32 if cond == "dark" else 0.5
        img = np.clip(img.astype(np.float32) * g, 0, 255)
        yy, xx = np.mgrid[0:h, 0:w]
        img *= (1 - 0.35 * (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)))[..., None]
        img = np.clip(img + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 9, img.shape), 0, 255).astype(np.uint8)
    if cond == "glare":
        ov = np.zeros((h, w), np.float32)
        cx, cy = pts.mean(0)
        for _ in range(rng.randint(1, 2)):
            ex, ey = int(cx + rng.uniform(-0.3, 0.3) * w * 0.6), int(cy + rng.uniform(-0.3, 0.3) * h * 0.5)
            cv2.ellipse(ov, (ex, ey), (int(w * rng.uniform(0.08, 0.14)), int(h * rng.uniform(0.07, 0.12))), rng.uniform(0, 180), 0, 360, 1.0, -1)
        ov = cv2.GaussianBlur(ov, (0, 0), 30)
        img = np.clip(img.astype(np.float32) * (1 - 0.9 * ov[..., None]) + 255 * 0.9 * ov[..., None], 0, 255).astype(np.uint8)
    if cond == "partial":                                      # seule une partie de la boîte est visible
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        if rng.choice(["top", "left"]) == "top":
            img = img[max(0, int(y0 - 20)):int(y0 + (y1 - y0) * rng.uniform(0.52, 0.62)), :]
        else:
            img = img[:, max(0, int(x0 - 20)):int(x0 + (x1 - x0) * rng.uniform(0.55, 0.65))]
    if cond == "lowres":
        img = cv2.resize(img, (560, int(560 * h / w)), interpolation=cv2.INTER_AREA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", help="dossier de sortie (défaut : le nom du split)")
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    base = OUT / (a.name or a.split)
    labels = []
    for row in pick_products(a.split, a.n, a.seed):
        face = render_face(row, rng)
        for cond in a.conditions:
            rot = {"rot15": rng.choice([-1, 1]) * rng.uniform(11, 18), "rot90": 0, "hard": rng.uniform(-10, 10)}.get(cond, rng.uniform(-4, 4))
            img, pts = scene_with_box(face, rng, rot=rot, persp=0.035)
            if cond == "rot90":
                img = cv2.rotate(img, rng.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]))
                pts = pts[:, ::-1]
            img = apply_condition(img, pts, cond, rng)
            path = base / cond / f"{row['code']}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 25 if cond == "lowres" else 90])
            labels.append({"file": str(path.relative_to(OUT)), "condition": cond, "code": str(row["code"]), "brand": row["brand"],
                           "nom": row["nom"], "dosage_mg": None if row["dosage_mg"] == "" else float(row["dosage_mg"]),
                           "forme": row["forme"], "split": a.split})
    base.mkdir(parents=True, exist_ok=True)
    (base / "labels.jsonl").write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in labels) + "\n", encoding="utf-8")
    print(f"{a.split}: {len(labels)} images ({len(labels) // max(1, len(a.conditions))} produits x {len(a.conditions)} conditions) -> {base}")


if __name__ == "__main__":
    main()
