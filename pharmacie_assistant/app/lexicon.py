"""Lexiques : villes, formes, temps, mots-outils, motifs d'intention.

Tout est passé par `N()` (même normalisation que le texte utilisateur) : les motifs
arabes de INTENT_PATTERNS sont déjà écrits sous forme normalisée (ة->ه, أ->ا, ى->ي).
"""
from __future__ import annotations

import re

from .textnorm import make_doc


def N(s: str) -> str:
    return make_doc(s).norm


INTENTS = ["check_availability", "price", "reimbursement", "pharmacy_phone",
           "pharmacy_location", "pharmacy_hours", "reserve_order", "out_of_scope"]
# priorité en cas d'égalité de score
INTENT_PRIORITY = ["reserve_order", "reimbursement", "price", "pharmacy_phone",
                   "pharmacy_hours", "pharmacy_location", "check_availability", "out_of_scope"]
MED_INTENTS = {"check_availability", "price", "reimbursement", "reserve_order"}
PHARMACY_INTENTS = {"pharmacy_phone", "pharmacy_location", "pharmacy_hours"}

# ------------------------------------------------------------------ villes
CITIES = {
    "RABAT": ["rabat", "rbat", "الرباط", "رباط"],
    "CASABLANCA": ["casablanca", "casa", "kaza", "dar el beida", "dar beida", "الدار البيضاء", "دار البيضاء", "كازا"],
    "SALE": ["sale", "sla", "سلا"],
    "MARRAKECH": ["marrakech", "marrakesh", "marrakch", "mrakech", "مراكش"],
    "FES": ["fes", "fas", "فاس"],
    "TANGER": ["tanger", "tangier", "tanja", "طنجة"],
    "MEKNES": ["meknes", "mknas", "مكناس"],
    "AGADIR": ["agadir", "أكادير"],
    "OUJDA": ["oujda", "وجدة"],
    "KENITRA": ["kenitra", "القنيطرة", "قنيطرة"],
    "TETOUAN": ["tetouan", "tetwan", "تطوان"],
    "TEMARA": ["temara", "تمارة"],
    "MOHAMMEDIA": ["mohammedia", "المحمدية"],
    "EL JADIDA": ["el jadida", "eljadida", "jadida", "الجديدة"],
    "NADOR": ["nador", "الناظور"],
    "BENI MELLAL": ["beni mellal", "بني ملال"],
    "SETTAT": ["settat", "سطات"],
    "KHOURIBGA": ["khouribga", "خريبكة"],
    "SAFI": ["safi", "آسفي"],
    "LAAYOUNE": ["laayoune", "layoune", "العيون"],
    "DAKHLA": ["dakhla", "الداخلة"],
    "ESSAOUIRA": ["essaouira", "mogador", "الصويرة"],
    "TAZA": ["taza", "تازة"],
    "BERRECHID": ["berrechid", "برشيد"],
    "LARACHE": ["larache", "العرائش"],
    "AL HOCEIMA": ["al hoceima", "hoceima", "الحسيمة"],
    "OUARZAZATE": ["ouarzazate", "ورزازات"],
    "ERRACHIDIA": ["errachidia", "الرشيدية"],
    "IFRANE": ["ifrane", "إفران"],
    "TIZNIT": ["tiznit", "تيزنيت"],
    "GUELMIM": ["guelmim", "كلميم"],
    "BERKANE": ["berkane", "بركان"],
    "TAOURIRT": ["taourirt", "تاوريرت"],
    "KHEMISSET": ["khemisset", "الخميسات"],
    "SIDI KACEM": ["sidi kacem", "سيدي قاسم"],
    "SKHIRAT": ["skhirat", "الصخيرات"],
}
CITY_LOOKUP = {N(a): c for c, al in CITIES.items() for a in al + [c.lower()]}

NEIGHBORHOODS = {N(k): v for k, v in {
    "agdal": "Agdal", "maarif": "Maarif", "ma3arif": "Maarif", "gauthier": "Gauthier",
    "souissi": "Souissi", "gueliz": "Gueliz", "hivernage": "Hivernage", "anfa": "Anfa",
    "californie": "Californie", "bourgogne": "Bourgogne", "ain sebaa": "Ain Sebaa",
    "sidi maarouf": "Sidi Maarouf", "bernoussi": "Bernoussi", "ville nouvelle": "Ville Nouvelle",
    "malabata": "Malabata", "founty": "Founty", "talborjt": "Talborjt", "hamria": "Hamria",
    "yacoub el mansour": "Yacoub El Mansour", "narjiss": "Narjiss", "iberia": "Iberia",
}.items()}
NB_MARKERS = {N(x) for x in ["hay", "حي", "derb", "درب", "quartier", "lotissement"]}

# ------------------------------------------------------------------ formes galéniques
FORMS = {
    "sirop": ["sirop", "syrop", "sirops", "سيروب", "شراب"],
    "comprimé": ["comprime", "comprimes", "cp", "cpr", "comprim", "كومبريمي", "قرص", "اقراص"],
    "gélule": ["gelule", "gelules", "capsule", "capsules", "caps", "كبسولة"],
    "sachet": ["sachet", "sachets", "ساشي", "ساشيه"],
    "injectable": ["injectable", "injection", "piqure", "ampoule", "ampoules", "ابرة", "حقنة"],
    "suppositoire": ["suppositoire", "suppositoires", "suppo"],
    "collyre": ["collyre", "collyres"],
    "pommade": ["pommade", "creme", "gel", "pommada"],
    "gouttes": ["gouttes", "goutte", "قطرة"],
}
FORM_LOOKUP = {N(a): c for c, al in FORMS.items() for a in al}
FORM_REGEX = {  # canon -> regex sur la colonne FORME du référentiel
    "sirop": r"SIROP|BUVABLE", "comprimé": r"COMPRIME", "gélule": r"GELULE|CAPSULE",
    "sachet": r"SACHET|POUDRE|GRANULE", "injectable": r"INJECTABLE|AMPOULE",
    "suppositoire": r"SUPPOSITOIRE", "collyre": r"COLLYRE", "pommade": r"POMMADE|CREME|GEL",
    "gouttes": r"GOUTTES",
}

# ------------------------------------------------------------------ dosage / quantité
UNIT_MAP = {"mg": "mg", "g": "g", "gr": "g", "ml": "ml", "mcg": "mcg", "ui": "ui",
            N("ملغ"): "mg", N("مغ"): "mg", N("غ"): "g", N("غرام"): "g"}
CLASSIFIERS = {N(x) for x in ["dyal", "d", "de", "du", "des", "boite", "boites", "علب", "علبة",
                              "3olba", "3olab", "3olbat", "olba", "olab", "flacon", "plaquette",
                              "ديال", "د", "unite", "unites"]}
NUM_WORDS = {N(k): v for k, v in {
    "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "wahed": 1, "wa7ed": 1,
    "jouj": 2, "joj": 2, "tlata": 3, "tlat": 3, "rb3a": 4, "khamsa": 5,
    "واحد": 1, "جوج": 2, "زوج": 2, "تلاتة": 3, "ربعة": 4, "خمسة": 5}.items()}

# ------------------------------------------------------------------ temps
TIME_TAGS = {
    "now": ["daba", "دابا", "دبا", "maintenant", "tout de suite", "now"],
    "tonight": ["ce soir", "had l3chiya", "هاد العشية", "l3chiya"],
    "night": ["الليلة", "lila", "had lila", "cette nuit"],
    "today": ["lyoum", "اليوم", "aujourd hui", "aujourdhui", "today"],
    "tomorrow": ["ghda", "غدا", "demain"],
    "sunday": ["nhar l 7ad", "nhar l7ad", "dimanche", "الاحد", "نهار الاحد"],
    "closing": ["ferme", "fermeture", "kat9fl", "katsed", "kaysd", "كتسد", "كيسد"],
    "garde": ["garde", "7ras", "حراسة", "الحراسة"],
}
TIME_TAGS = {k: [N(x) for x in v] for k, v in TIME_TAGS.items()}
TIME_LABEL = {"now": "maintenant", "tonight": "ce soir", "night": "cette nuit",
              "today": "aujourd'hui", "tomorrow": "demain", "sunday": "dimanche"}

PHARMACY_WORDS = {N(x) for x in ["pharmacie", "pharmacy", "pharma", "صيدلية", "الصيدلية", "pharmacien"]}
NAME_SKIP_LEADING = {N(x) for x in ["de", "du", "des", "dyal", "d", "ديال", "د", "la", "le", "l"]}

# ------------------------------------------------------------------ mots-outils (jamais des médicaments)
_FUNC = """
wach wash kayn kayna kayniin kayen 3ndkom 3andkom 3ndek 3andek 3andk 3ndk 3tini 3tina 3afak
bghit bghina bghiti nbghi chhal bch7al bchhal ch7al chno chnou chmen shno fin imta wa9tach
lyoum daba dba ghda dyal dyalo dyalha mn men 3la 3nd 3and li lli hadi hada had hadchi dwa
dwaya drari ghir bzaf shwiya m3a ila wla awla w we numro numero nmra tel telephone adresse
pharmacie pharmacy pharma pharmacies taman prix combien tarif t3awed kayt3awed kay3awed cnss
cnops amo remboursable rembourse remboursement m7loula m7loul mfto7a ouverte ouvert ferme
fermeture garde nuit horaire horaires heure qrib qriba 9rib proche pres plus n7ejez n7jez
7ejez commander reserver reservez reserve commande bonjour bonsoir salam salut merci svp stp
il vous plait avez avoir vous je tu elle nous voudrais voudrai veux cherche chercher trouver
le la les un une du des au aux en dans sur pour par avec sans est et ou ce cette ces mon ma
mes quel quelle quels quelles qui que quoi comment moi toi me te se dispo disponible
disponibles stock boite boites medicament medicaments dois prendre mal numero dyali ndir
واش هل كاين كاينة كاينين عندكم عندك عطيني عطينا بغيت بغينا بغيتي شحال بشحال شنو شنوهو اش
فين امتا فوقاش اليوم دابا دبا غدا ديال في من على عند لي لا ما هاد هاذا هذا الدوا دوا دواء
الدواء علب علبة ثمن سعر نمرة التيليفون تيليفون هاتف صيدلية الصيدلية قريبة قريب مفتوحة مفتوح
مسدود نحجز نطلب حجز ليا كيتعوض يتعوض الكنوبس كنوبس الضمان عافاك لو سمحت فضلك الحراسة حراسة
مع الى إلى او ولا
f fi a au aux chez
dyalkom dyalhom dyalna dyali dyalek dyalo dyalha mazal mazala vendu vend vendre koulchi tqdro tqdr t7jzo khdama khdam mfto7a ghali rkhis khasni nkhls
auriez auriez-vous souhaite souhaiter mettre cote preparer preparez connaitre faire contre
coute couter ncontacti kifach nsaft ka t9a3 tl3 hiya hna temma
لقيتو مازال خاصني نخلص كيساوي كيغطي كيدخل فتحات خدامة غالي رخيص بزاف شكرا عافاك السلام عليكم جهز حجزو
ناخدهم نجي نقدر نعرف ندير فين ليوم
"""
FUNCTION_WORDS = {N(w) for w in _FUNC.split()}

# ------------------------------------------------------------------ motifs d'intention (regex sur texte normalisé)
_P = lambda pat, w: (re.compile(pat), w)  # noqa: E731
INTENT_PATTERNS = {
    "check_availability": [
        _P(r"\bkayn", 2), _P(r"\b3(?:a)?ndk", 2), _P(r"\bdispo", 3), _P(r"\bavez\b", 2),
        _P(r"\bcherche", 2), _P(r"\bstock", 2), _P(r"\bl9it", 1),
        _P(r"\bكاين", 2), _P(r"\bعندكم|\bعندك\b", 2), _P(r"متوفر|موجود|متاح", 3),
    ],
    "price": [
        _P(r"\btaman|\btman\b|\btmn\b|\bthaman", 3), _P(r"\bprix\b|\btarif|\bcombien|\bcb\b", 3),
        _P(r"\bb?ch7al|\bb?chhal|\bb?chal\b", 2),
        _P(r"ثمن|تمن|سعر", 3), _P(r"شحال", 2),
    ],
    "reimbursement": [
        _P(r"t3aw|3aw[ei]d|3awd", 4), _P(r"rembours", 4),
        _P(r"\bcnops\b|\bcnss\b|\bamo\b|\bmutuelle|\bassurance|\bcouvert|prise? en charge", 3),
        _P(r"تعو[ضو]|تعويض", 4), _P(r"كنوبس|الضمان|\bضمان", 3),
    ],
    "pharmacy_phone": [
        _P(r"\bnumro|\bnumero|\bnmra|\bnemra|\bnomra|\bnum\b", 3),
        _P(r"\btel\b|telephone|\btlf\b|tilifon|\bwhatsapp|\bwatsap|\bcontact", 3),
        _P(r"\bappel|\bjoindre", 2),
        _P(r"نمره|نمرت|رقم|هاتف|تيليفون|تلفون|واتساب|اتصال", 3),
    ],
    "pharmacy_location": [
        _P(r"\bfin\b", 3), _P(r"\bou est\b|\bou se trouve|\bou trouver", 3),
        _P(r"\badresse|\blocalis|\bitineraire|\bdirection|\bgps\b|\bmaps?\b", 3),
        _P(r"\bqrib|\b9rib|\bpres de|\bproche|\bplus proche", 2),
        _P(r"\bفين\b", 3), _P(r"\bاين\b", 3), _P(r"عنوان|موقع|خريطه", 3), _P(r"قريب", 2),
    ],
    "pharmacy_hours": [
        _P(r"\bm7lo?ul|\bmfto7|\bmaftou7|\bmftou7|\bouvert|\bferm|\bhorair|\bheure|\bgarde\b|"
           r"\b7ras|\bnuit\b|\bimta\b|\bwa9tach|\bw9taach|\bkat9f|\bkatsed|\bkaysd|\bkayf?t7|\bkat7ll", 3),
        _P(r"مفتوح|مسدود|كيسد|كتسد|حراسه|ساعه|فوقاش|امتا|امتي", 3),
    ],
    "reserve_order": [
        _P(r"7[ae]?j[ae]?z|\b7jz", 4), _P(r"\breserv", 4), _P(r"command", 3),
        _P(r"حجز", 4), _P(r"نطلب|طلب|نكوموندي|كوموند", 3), _P(r"جهز", 2),
    ],
    "out_of_scope": [
        _P(r"\bjaw\b|\bmeteo\b|\bweather\b|\bmatch\b|\bfoot\b|\bblague\b|\bnokta\b|\bchno ndir|"
           r"\bchnu ndir|\bchno (?:nakhod|nchri|nchrb)|\bque prendre|\bcontre la |\bquel medicament|\bdois je|\bmal de\b|\bs3al|\bsa3la|\bw?ja3\b|"
           r"\bdouleur|\btoux\b|\bfievre|\bconseil|\bsalaire|\bpolitique", 4),
        _P(r"طقس|\bالجو\b|ماتش|مباراه|نكته|سعال|صداع|وجع|حراره|\bندير\b|\bشنو (?:نشري|ناخد|نشرب)", 4),
    ],
}
ALL_PATTERNS = [rx for lst in INTENT_PATTERNS.values() for rx, _ in lst]


def is_function(tok: str) -> bool:
    """Vrai si le token est un mot-outil / mot-clé d'intention (donc pas un médicament)."""
    if tok in FUNCTION_WORDS or tok in PHARMACY_WORDS or tok in NB_MARKERS:
        return True
    return any(rx.search(tok) for rx in ALL_PATTERNS)


# ------------------------------------------------------------------ détection de langue
DARIJA_MARKERS = {N(x) for x in """wach wash kayn kayna kayniin 3ndkom 3andkom 3tini chhal ch7al bch7al bchhal
 taman bghit n7ejez dyal daba fin chno numro qriba chmen m7loula qrib 9rib lyoum imta wa9tach dwa kayt3awed kay3awed
 واش كاين كاينة شحال بشحال بغيت ديال دابا فين عندكم كيتعوض شنو نمرة عطيني نحجز نطلب هاد دراري""".split()}
FRENCH_MARKERS = {N(x) for x in """bonjour bonsoir avez vous quel quelle combien je voudrais cherche du des pour mon
 svp reserver reservez remboursable est ou prix heure ferme soir moi boite comprimes dispo disponible""".split()}
MSA_MARKERS = {N(x) for x in "هل اين كم سعر اريد يوجد متى لو سمحت فضلك".split()}
