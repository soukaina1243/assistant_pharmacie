"""Prétraitement d'image (OpenCV) : décodage, taille, qualité (flou / luminosité / reflets), variantes pour l'OCR."""
from __future__ import annotations

import math

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)   # applique l'orientation EXIF
    if img is None:
        raise ValueError("Image illisible (formats acceptés : jpg, png, webp, bmp).")
    return img


def normalize_size(img: np.ndarray, target: int = 1500, max_side: int = 1800) -> np.ndarray:
    """Ramène le plus grand côté vers ~target px (agrandit les petites images, réduit les très grandes)."""
    h, w = img.shape[:2]
    m = max(h, w)
    if m < 1100:
        s = target / m
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    if m > max_side:
        s = max_side / m
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def quality_metrics(img: np.ndarray) -> dict:
    """Indicateurs simples qui expliquent un échec d'OCR et permettent de conseiller l'utilisateur."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (640, int(640 * g.shape[0] / g.shape[1]))) if g.shape[1] > 640 else g
    blur = float(cv2.Laplacian(small, cv2.CV_64F).var())            # faible = flou
    mean = float(g.mean())
    glare = float((g >= 250).mean())
    h, w = img.shape[:2]
    warns = []
    if blur < 40:
        warns.append("image_floue")
    if mean < 70:
        warns.append("image_sombre")
    if glare > 0.15:                       # seuil élevé : un emballage blanc ou crème est naturellement très clair
        warns.append("reflets")
    if min(h, w) < 400:
        warns.append("basse_resolution")
    return {"blur_score": round(blur, 1), "brightness": round(mean, 1), "glare_ratio": round(glare, 3),
            "width": w, "height": h, "warnings": warns}


def to_gray_stretch(img: np.ndarray) -> np.ndarray:
    """Gris + étirement d'exposition (percentiles 1-99) : compense les photos sous/sur-exposées."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo, hi = np.percentile(g, [1, 99])
    return np.clip((g - lo) / max(hi - lo, 1.0) * 255, 0, 255).astype(np.uint8)


def to_gray_clahe(img: np.ndarray) -> np.ndarray:
    """Étirement d'exposition puis égalisation locale du contraste (corrige l'éclairage inégal)."""
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(to_gray_stretch(img))


def variant(img: np.ndarray, name: str) -> np.ndarray:
    if name == "stretch":
        return to_gray_stretch(img)
    if name == "half":         # image réduite de moitié : Tesseract lit mieux les très gros caractères
        g = to_gray_clahe(img)
        return cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    g = to_gray_clahe(img)
    if name == "clahe":
        return g
    if name == "binary":       # seuillage adaptatif : robuste aux ombres et dégradés
        return cv2.adaptiveThreshold(cv2.GaussianBlur(g, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 41, 14)
    if name == "sharp":        # masque flou : réaccentue les contours (images floues)
        blur = cv2.GaussianBlur(g, (0, 0), 3)
        return cv2.addWeighted(g, 2.2, blur, -1.2, 0)
    if name == "inverted":     # texte clair sur fond sombre
        return 255 - g
    raise KeyError(name)


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotation (sens horaire si angle>0) ; multiples de 90 sans perte, sinon agrandit le cadre."""
    a = angle % 360
    if a == 0:
        return img
    if a == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if a == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if a == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    border = int(np.median(img)) if img.ndim == 2 else (int(np.median(img)),) * 3
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=border)


def find_box_quad(img: np.ndarray):
    """Cherche le plus grand quadrilatère (la boîte) ; renvoie ses 4 coins ou None."""
    h, w = img.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(g, 40, 120), np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c)
        if area < 0.2 * h * w or area > 0.97 * h * w:
            continue
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best = approx.reshape(4, 2).astype(np.float32)
            break
    return best


def rectify(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Redresse la boîte (correction de perspective) à partir de ses 4 coins."""
    s = quad.sum(1); d = np.diff(quad, axis=1).ravel()
    tl, br, tr, bl = quad[np.argmin(s)], quad[np.argmax(s)], quad[np.argmin(d)], quad[np.argmax(d)]
    W = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    H = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if W < 100 or H < 100:
        return img
    M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC)


'''def estimate_skew(img: np.ndarray) -> float:
    """Inclinaison dominante (degrés, sens horaire > 0) d'après les lignes de contours (bords de boîte, lignes de texte)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sc = 800 / max(g.shape)
    small = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else g
    edges = cv2.Canny(cv2.GaussianBlur(small, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=80, minLineLength=int(0.12 * max(small.shape)), maxLineGap=10)
    if lines is None:
        return 0.0
    angs, wts = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        while a > 45:
            a -= 90
        while a <= -45:
            a += 90
        angs.append(a); wts.append(math.hypot(x2 - x1, y2 - y1))
    order = np.argsort(angs)
    cw = np.cumsum(np.array(wts)[order])
    return float(np.array(angs)[order][np.searchsorted(cw, cw[-1] / 2)])   # médiane pondérée par la longueur
''' 
def estimate_skew(img: np.ndarray) -> float:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sc = 800 / max(g.shape)

        small = (
            cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            if sc < 1
            else g
        )

        edges = cv2.Canny(
            cv2.GaussianBlur(small, (3, 3), 0),
            50,
            150,
        )

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 360,
            threshold=80,
            minLineLength=int(0.12 * max(small.shape)),
            maxLineGap=10,
        )

        if lines is None:
            return 0.0

        lines = np.array(lines).reshape(-1, 4)

        angs = []
        wts = []

        for x1, y1, x2, y2 in lines:
            a = math.degrees(math.atan2(y2 - y1, x2 - x1))

            while a > 45:
                a -= 90
            while a <= -45:
                a += 90

            angs.append(a)
            wts.append(math.hypot(x2 - x1, y2 - y1))

        order = np.argsort(angs)
        cw = np.cumsum(np.array(wts)[order])

        return float(
            np.array(angs)[order][
                np.searchsorted(cw, cw[-1] / 2)
            ]
        )