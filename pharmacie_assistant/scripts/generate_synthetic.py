"""Génère des requêtes synthétiques annotées.

 - data/synthetic/train.jsonl     : (text, intent)  -> entraînement du classifieur d'intention
 - data/synthetic/test_hard.jsonl : annotation complète (mêmes champs que le corpus), formulations
                                    DIFFÉRENTES de celles du train + bruit + médicaments rares.
Limite : un test synthétique reste optimiste. Le vrai test = vos ~100 messages écrits à la main.
"""
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.lexicon import CITIES, FORMS  # noqa: E402

random.seed(7)
meds = pd.read_csv(ROOT / "data/processed/medications.csv").fillna({"forme": ""})
pop = meds.groupby("brand").size().sort_values(ascending=False)
single = [b for b in pop.index if re.fullmatch(r"[A-Z]{5,}", b)]
TOP, ALL = single[:300], single

NAMES = {"Ibn Sina": "ابن سينا", "Al Amal": "الأمل", "Centrale": "المركزية", "Ennahda": "النهضة", "Hassan II": "الحسن الثاني",
         "Al Massira": "المسيرة", "Al Andalous": "الأندلس", "Al Fath": "الفتح", "Ibn Rochd": "ابن رشد", "Atlas": "الأطلس"}
CITY_KEYS = ["RABAT", "CASABLANCA", "SALE", "MARRAKECH", "FES", "TANGER", "MEKNES", "AGADIR", "OUJDA", "KENITRA", "TETOUAN"]
NB_LAT = ["Agdal", "Maarif", "Souissi", "Gueliz", "Hay Hassani", "Hay Riad", "Gauthier", "Hivernage", "Anfa"]
NB_AR = ["حي الرياض", "حي المسيرة", "حي السلام", "درب السلطان"]
FORM_SURF = {"fr": {"comprimé": "comprimés", "sirop": "sirop", "sachet": "sachets", "gélule": "gélules", "injectable": "injectable",
                    "suppositoire": "suppositoires", "collyre": "collyre", "pommade": "pommade", "gouttes": "gouttes"},
             "dl": {"comprimé": "comprimé", "sirop": "sirop", "sachet": "sachet", "gélule": "gélule", "injectable": "injectable",
                    "suppositoire": "suppositoire", "collyre": "collyre", "pommade": "pommade", "gouttes": "gouttes"},
             "ar": {"comprimé": "كومبريمي", "sirop": "سيروب", "sachet": "ساشي", "injectable": "ابرة"}}
FORM_RX = {"comprimé": "COMPRIME", "sirop": "SIROP|BUVABLE", "gélule": "GELULE", "sachet": "SACHET", "injectable": "INJECTABLE",
           "suppositoire": "SUPPOSITOIRE", "collyre": "COLLYRE", "pommade": "POMMADE|CREME", "gouttes": "GOUTTES"}
TIMES = {"dl": [("daba", "maintenant"), ("lyoum", "aujourd'hui"), ("ghda", "demain")],
         "fr": [("maintenant", "maintenant"), ("ce soir", "ce soir"), ("aujourd'hui", "aujourd'hui"), ("demain", "demain"), ("dimanche", "dimanche")],
         "ar": [("دابا", "maintenant"), ("اليوم", "aujourd'hui"), ("الليلة", "cette nuit"), ("غدا", "demain"), ("نهار الأحد", "dimanche")]}
LIT_TIME = {"nhar l-7ad": "dimanche"}

# ---------------------------------------------------------------- translittération latin -> arabe (approx.)
_L2A = {"b": "ب", "c": "ك", "d": "د", "f": "ف", "g": "غ", "h": "ه", "j": "ج", "k": "ك", "l": "ل", "m": "م", "n": "ن", "p": "ب",
        "q": "ق", "r": "ر", "s": "س", "t": "ت", "v": "ف", "w": "و", "x": "كس", "z": "ز"}


def ar_translit(w):
    w = w.lower()
    for a, b in [("ch", "ش"), ("sh", "ش"), ("ph", "ف"), ("th", "ت"), ("kh", "خ"), ("ou", "و")]:
        w = w.replace(a, b)
    out = []
    for i, ch in enumerate(w):
        if ch in "aeiouy":
            if i == 0:
                out.append("ا" if ch in "ae" else ("أو" if ch in "ou" else "إي"))
            elif random.random() < 0.75:
                out.append({"a": "ا", "e": "ي", "i": "ي", "o": "و", "u": "و", "y": "ي"}[ch])
        else:
            out.append(_L2A.get(ch, ch))
    return "".join(out)


def typo(w):
    if len(w) < 6 or random.random() > 0.25:
        return w
    i = random.randint(1, len(w) - 2)
    k = random.choice(["del", "dup", "swap"])
    return w[:i] + w[i + 1:] if k == "del" else (w[:i] + w[i] + w[i:] if k == "dup" else w[:i] + w[i + 1] + w[i] + w[i + 2:])


# ---------------------------------------------------------------- slots
def pick_med(style):
    brand = random.choice(TOP) if random.random() < 0.6 else random.choice(ALL)
    rows = meds[meds["brand"] == brand]
    surf = ar_translit(brand) if style == "ar" else typo(random.choice([brand.title(), brand.lower(), brand.title()]))
    return brand, rows, surf


def pick_dose(rows, style):
    r = rows[rows["dosage_mg"].notna() & (rows["dosage_mg"] > 0)]
    if r.empty or random.random() < 0.5:
        return "", None
    v = float(r.sample(1).iloc[0]["dosage_mg"])
    if v >= 1000 and v % 1000 == 0 and random.random() < 0.5:
        n = int(v // 1000)
        return random.choice([f"{n}g", f"{n} g"]), f"{n} g"
    n = int(v) if v == int(v) else v
    forms = [f"{n} mg", f"{n}mg"] + ([f"{n}"] if v >= 10 else [])
    return random.choice(forms), f"{n} mg"


def pick_form(rows, style):
    avail = [c for c, rx in FORM_RX.items() if rows["forme"].str.contains(rx, regex=True).any() and c in FORM_SURF[style]]
    if not avail or random.random() < 0.7:
        return "", None
    c = random.choice(avail)
    return FORM_SURF[style][c], c


def pick_city(style):
    k = random.choice(CITY_KEYS)
    al = CITIES[k]
    if style == "ar":
        s = random.choice([a for a in al if re.search(r"[\u0600-\u06FF]", a)])
        return k, ("ف" + s if random.random() < 0.5 else s)
    return k, random.choice([a for a in al if not re.search(r"[\u0600-\u06FF]", a)]).title()


def fill(tpl, style, intent):
    brand, rows, msurf = pick_med(style)
    dsurf, dgold = pick_dose(rows, style)
    fsurf, fgold = pick_form(rows, style)
    ck, csurf = pick_city(style)
    pl, pa = random.choice(list(NAMES.items()))
    psurf, pgold = (pa, pa) if style == "ar" else (pl, pl)
    nb = random.choice(NB_AR if style == "ar" else NB_LAT)
    qty = random.randint(1, 6)
    tsurf, tgold = random.choice(TIMES[style])
    text = tpl.format(med=msurf, dose=dsurf, form=fsurf, city=csurf, ph=psurf, nb=nb, qty=qty, time=tsurf)
    text = re.sub(r"\s+", " ", text).strip()
    ent = {"medications": [], "quantity": None, "pharmacy_name": None, "city": None, "neighborhood": None, "time_reference": None}
    if "{med}" in tpl:
        ent["medications"] = [{"raw": msurf, "name": brand, "dosage": dgold if "{dose}" in tpl else None,
                               "form": fgold if "{form}" in tpl else None}]
    if "{qty}" in tpl:
        ent["quantity"] = qty
    if "{ph}" in tpl:
        ent["pharmacy_name"] = pgold
    if "{city}" in tpl:
        ent["city"] = ck
    if "{nb}" in tpl:
        ent["neighborhood"] = nb
    parts = []
    if "garde" in tpl or "الحراسة" in tpl or "7ras" in tpl:
        parts.append("garde")
    if "{time}" in tpl:
        parts.append(tgold)
    else:
        for lit, g in LIT_TIME.items():
            if lit in tpl:
                parts.append(g)
    if not parts and any(x in tpl for x in ["ferme", "kat9fl", "كتسد"]):
        parts.append("fermeture")
    ent["time_reference"] = " ".join(parts) or None
    return text, ent


def noise(text, style):
    if style == "ar":
        return text
    if random.random() < 0.3:
        text = text.lower()
    if random.random() < 0.2:
        text = text.replace("?", "").strip()
    if random.random() < 0.15:
        text += random.choice([" svp", " stp", " merci"] if style == "fr" else [" 3afak", " merci"])
    return text


# ---------------------------------------------------------------- gabarits  (train | test)
T = {}
def add(intent, style, train, test): T[(intent, style)] = (train, test)  # noqa: E704

add("check_availability", "dl",
    ["wach kayn {med} ?", "kayn {med} {dose} 3ndkom ?", "wach kayna {med} f {city} ?", "3ndkom {med} {form} ?"],
    ["salam wach l9ito {med} {dose} ?", "{med} {dose} kayn f pharmacie ?", "wach mazal kayn {med} f {city} ?", "bghit n3ref wach kayn {med} {form}"])
add("check_availability", "fr",
    ["Avez-vous du {med} {dose} ?", "Est-ce que vous avez {med} en {form} ?", "{med} {dose} disponible ?", "Je cherche du {med} à {city}"],
    ["Bonjour, auriez-vous {med} ?", "Vous avez {med} {dose} en stock ?", "Y a-t-il du {med} à {city} ?", "Dispo {med} {form} ?"])
add("check_availability", "ar",
    ["واش كاين {med} {dose}؟", "كاين {med} عندكم؟", "عندكم {med} {form}؟", "واش كاين {med} {city}؟"],
    ["واش لقيتو {med}؟", "{med} {dose} متوفر؟", "واش مازال كاين {med} {city}؟", "بغيت نعرف واش كاين {med}"])
add("price", "dl",
    ["chhal taman {med} ?", "bch7al {med} {dose} ?", "{med} {dose} bch7al ?", "taman dyal {med} chhal ?"],
    ["chhal kaysswa {med} {form} ?", "{med} {dose} chhal ?", "ch7al khasni nkhls 3la {med} {dose}", "taman {med} {form} ?"])
add("price", "fr",
    ["Quel est le prix du {med} {dose} ?", "{med} {dose} prix SVP", "C'est combien le {med} ?", "Combien coûte {med} {form} ?"],
    ["Tarif {med} {dose} ?", "À combien est vendu le {med} ?", "Je voudrais connaître le prix de {med}", "{med} {dose} ça coûte combien ?"])
add("price", "ar",
    ["شحال ثمن {med} {dose}؟", "بشحال كاين {med}؟", "{med} {dose} بشحال؟", "شحال كيساوي {med} {form}؟"],
    ["ثمن {med}؟", "{med} {dose} شحال؟", "شحال خاصني نخلص على {med}؟", "السعر ديال {med}"])
add("reimbursement", "dl",
    ["wach {med} kayt3awed mn CNSS ?", "{med} kay3awed la cnops ?", "wach {med} 3la hsab l-mutuelle ?", "{med} remboursable ?"],
    ["wach CNOPS katt3awd {med} ?", "{med} {dose} kayt3awad ?", "wach kayddiha l-assurance {med} ?", "{med} {dose} fiha t3wid ?"])
add("reimbursement", "fr",
    ["Est-ce que {med} est remboursé ?", "{med} remboursable par la CNOPS ?", "Le {med} est-il pris en charge par la CNSS ?", "Remboursement {med} {dose} ?"],
    ["{med} est couvert par l'AMO ?", "Ma mutuelle rembourse {med} ?", "Prise en charge de {med} {form} ?", "Le {med} passe-t-il avec la CNOPS ?"])
add("reimbursement", "ar",
    ["واش {med} كيتعوض من الكنوبس؟", "{med} كيتعوض؟", "واش {med} {dose} كيتعوض من الضمان؟", "{med} كيتعوض من التأمين؟"],
    ["الكنوبس كتعوض {med}؟", "واش كيدخل {med} فالتعويض؟", "{med} فيه تعويض؟", "واش الضمان كيغطي {med}؟"])
add("pharmacy_phone", "dl",
    ["3tini numro dyal pharmacie {ph} f {city}", "numro tel dyal pharmacie {ph}", "chno numro dyal pharmacie {ph} ?", "nmra dyal pharmacie {ph} f {city}"],
    ["bghit ncontacti pharmacie {ph} {city}", "wach 3ndk watsap dyal pharmacie {ph} ?", "tilifon dyal pharmacie {ph} f {city}", "3afak numro pharmacie {ph}"])
add("pharmacy_phone", "fr",
    ["Quel est le numéro de la pharmacie {ph} à {city} ?", "Numéro de téléphone de la pharmacie {ph}", "Je voudrais appeler la pharmacie {ph} à {city}", "Contact de la pharmacie {ph} {city}"],
    ["Pouvez-vous me donner le tel de la pharmacie {ph} ?", "Comment joindre la pharmacie {ph} à {city} ?", "Téléphone pharmacie {ph}", "Numéro pharmacie {ph} {city} svp"])
add("pharmacy_phone", "ar",
    ["عطيني نمرة التيليفون ديال صيدلية {ph}", "نمرة صيدلية {ph} {city}", "شنو هي نمرة صيدلية {ph}؟", "رقم هاتف صيدلية {ph}"],
    ["بغيت نتاصل بصيدلية {ph} {city}", "واتساب ديال صيدلية {ph}؟", "تيليفون صيدلية {ph}", "عافاك رقم صيدلية {ph} {city}"])
add("pharmacy_location", "dl",
    ["fin kayna pharmacie f {nb} {city} ?", "fin kayna pharmacie {ph} f {city} ?", "pharmacie qriba mn {nb}", "fin kayn pharmacie qrib mn {nb} ?"],
    ["fin ka t9a3 pharmacie {ph} f {city}", "adresse dyal pharmacie {ph} {city}", "ana f {nb} {city} fin kayna pharmacie", "pharmacie {ph} fin hiya ?"])
add("pharmacy_location", "fr",
    ["Où est la pharmacie {ph} à {city} ?", "Où trouver une pharmacie à {nb} {city} ?", "Pharmacie la plus proche de {nb}", "Adresse de la pharmacie {ph} {city}"],
    ["Localisation pharmacie {ph} {city}", "La pharmacie {ph}, elle est où ?", "Je cherche une pharmacie près de {nb} {city}", "Itinéraire vers la pharmacie {ph}"])
add("pharmacy_location", "ar",
    ["فين كاينة صيدلية {ph} {city}؟", "فين كاينة صيدلية قريبة من {nb}؟", "صيدلية قريبة من {nb}", "عنوان صيدلية {ph} {city}"],
    ["فين كاينة صيدلية {ph}؟", "أين توجد صيدلية {ph} {city}؟", "موقع صيدلية {ph}", "أنا فـ{nb} فين كاينة صيدلية؟"])
add("pharmacy_hours", "dl",
    ["wach pharmacie {ph} m7loula {time} ?", "chmen pharmacie de garde f {city} {time} ?", "wach pharmacie {ph} mfto7a nhar l-7ad ?", "imta kat9fl pharmacie {ph} ?"],
    ["pharmacie de garde {city} {time}", "7ras f {city} {time}", "wach mazal m7loula pharmacie {ph} ?", "wach pharmacie {ph} {city} khdama {time}"])
add("pharmacy_hours", "fr",
    ["La pharmacie {ph} est ouverte {time} ?", "Quelle est la pharmacie de garde à {city} {time} ?", "À quelle heure ferme la pharmacie {ph} ?", "Horaires de la pharmacie {ph} {city}"],
    ["Pharmacie de garde {city} {time}", "Pharmacie {ph} ouverte {time} ?", "Y a-t-il une pharmacie de nuit à {city} ?", "Jusqu'à quelle heure est ouverte la pharmacie {ph} ?"])
add("pharmacy_hours", "ar",
    ["واش صيدلية {ph} مفتوحة {time}؟", "شنو هي صيدلية الحراسة {city} {time}؟", "فوقاش كتسد صيدلية {ph}؟", "واش صيدلية {ph} مسدودة {time}؟"],
    ["صيدلية الحراسة {city} {time}", "واش صيدلية {ph} مازال مفتوحة؟", "امتا كتسد صيدلية {ph}؟", "صيدلية {ph} فتحات {time}؟"])
add("reserve_order", "dl",
    ["bghit n7ejez {qty} dyal {med}", "bghit n7ejez {med} {dose}", "n7ejez liya {qty} 3olab d {med} 3nd pharmacie {ph}", "bghit ncommandi {med} {form}"],
    ["wach tqdro t7jzo liya {qty} 3olab {med}", "n9dar nakhod {qty} d {med} ghda ?", "preparez liya {med} 3nd pharmacie {ph}", "7ejz liya {med} {dose}"])
add("reserve_order", "fr",
    ["Je voudrais réserver {qty} boîtes de {med}", "Réservez-moi {med} {dose} à la pharmacie {ph}", "Pouvez-vous me mettre de côté {qty} {med} ?", "Je souhaite commander {med} {form}"],
    ["Je réserve {qty} boîtes {med} {dose}", "Commande de {med} {dose} svp", "Mettez de côté {med} pour moi à la pharmacie {ph}", "Pouvez-vous préparer {qty} boîtes de {med} ?"])
add("reserve_order", "ar",
    ["بغيت نحجز {qty} علب د {med}", "بغيت نحجز {med} {dose}", "حجز ليا {qty} {med} عند صيدلية {ph}", "بغيت نطلب {med} {form}"],
    ["نقدر نحجز {med} {dose}؟", "عافاك جهز ليا {qty} علب {med}", "بغيت {qty} علب د {med} نجي ناخدهم", "حجزو ليا {med}"])

OOS = {"dl": (["chno l-jaw lyoum ?", "chkoun rba7 l-match ?", "9ol liya nokta", "chno ndir l s3al dyali ?", "bghit n3ref salaire dyal ostad", "kifach ndir compte f lbanka ?"],
              ["chhal b9a 3la l-match ?", "chno ndir l wja3 dyal rassi ?", "wach kayn taxi ?", "salam labas"]),
       "fr": (["Quel temps fait-il demain ?", "Raconte-moi une blague", "Quel médicament dois-je prendre pour la toux ?", "Qui a gagné le match hier ?", "Je veux des conseils pour mon mal de dos", "Bonjour"],
              ["Quelle est la météo à Rabat ?", "Que faire contre la fièvre ?", "Peux-tu me donner un conseil pour maigrir ?", "Merci beaucoup"]),
       "ar": (["شنو هو الطقس اليوم؟", "شكون ربح الماتش؟", "قول ليا نكتة", "شنو ندير لصداع؟", "بغيت نعرف كيفاش نفتح حساب فالبنك", "السلام عليكم"],
              ["شحال بقا على المباراة؟", "شنو ندير لوجع البطن؟", "واش كاين طاكسي؟", "شكرا بزاف"])}
CLAR = {"dl": [("reserve_order", "bghit n7ejez"), ("price", "chhal taman ?"), ("pharmacy_phone", "3tini numro dyal pharmacie"), ("pharmacy_hours", "wach pharmacie m7loula daba ?")],
        "fr": [("reserve_order", "Je voudrais réserver"), ("price", "Quel est le prix ?"), ("pharmacy_phone", "Numéro de la pharmacie ?"), ("pharmacy_hours", "La pharmacie est ouverte ?")],
        "ar": [("reserve_order", "بغيت نحجز"), ("price", "شحال الثمن؟"), ("pharmacy_phone", "نمرة الصيدلية؟"), ("pharmacy_hours", "واش الصيدلية مفتوحة؟")]}
CLAR_TEST = {"dl": [("reserve_order", "n7ejez liya"), ("price", "chhal hadchi ?"), ("pharmacy_phone", "numro dyal l-pharmacie 3afak"), ("pharmacy_hours", "wach l-pharmacie mfto7a ?")],
             "fr": [("reserve_order", "Réservez-moi"), ("price", "Combien ça coûte ?"), ("pharmacy_phone", "Le numéro de la pharmacie svp"), ("pharmacy_hours", "Elle ferme à quelle heure ?")],
             "ar": [("reserve_order", "حجزو ليا"), ("price", "بشحال؟"), ("pharmacy_phone", "عطيني نمرة الصيدلية"), ("pharmacy_hours", "واش الصيدلية مسدودة؟")]}
LANG = {"dl": ("darija", "latin"), "fr": ("french", "latin"), "ar": ("darija", "arabic")}
EMPTY = {"medications": [], "quantity": None, "pharmacy_name": None, "city": None, "neighborhood": None, "time_reference": None}


def row(i, text, style, intent, ent, status):
    p, s = LANG[style]
    return {"id": f"s{i:04d}", "text": text, "language": {"primary": p, "script": s, "code_mixed": (p == "darija" and s == "latin")},
            "intent": intent, "entities": ent, "status": status}


def main():
    train, test = [], []
    for (intent, style), (tr, te) in T.items():
        for tpl in tr:
            for _ in range(12):
                t, _e = fill(tpl, style, intent)
                train.append({"text": noise(t, style), "intent": intent})
        for tpl in te:
            for _ in range(4):
                t, e = fill(tpl, style, intent)
                test.append(row(len(test), noise(t, style), style, intent, e, "ok"))
    for style, (tr, te) in OOS.items():
        for t in tr:
            train += [{"text": noise(t, style), "intent": "out_of_scope"} for _ in range(6)]
        for t in te:
            test += [row(len(test), noise(t, style), style, "out_of_scope", dict(EMPTY), "out_of_scope") for _ in range(2)]
    for style in CLAR:
        for intent, t in CLAR[style]:
            train += [{"text": noise(t, style), "intent": intent} for _ in range(5)]
        for intent, t in CLAR_TEST[style]:
            test += [row(len(test), noise(t, style), style, intent, dict(EMPTY), "needs_clarification") for _ in range(2)]
    out = ROOT / "data/synthetic"
    out.mkdir(exist_ok=True, parents=True)
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in train)
    with open(out / "test_hard.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in test)
    print(f"train: {len(train)} | test_hard: {len(test)}")


if __name__ == "__main__":
    main()
