"""Réponse en langage naturel (français / Darija arabizi / Darija arabe) à partir de l'analyse + fulfillment."""
from __future__ import annotations


def _lang(r: dict) -> str:
    if r["language"]["script"] == "arabic":
        return "ar"
    return "fr" if r["language"]["primary"] == "french" else "dl"


def _n(x) -> str:
    x = float(x)
    return (f"{x:.2f}".rstrip("0").rstrip(".") if x != int(x) else str(int(x))).replace(".", ",")


T = {
    "fr": {
        "oos": "Je peux vous aider pour les médicaments (disponibilité, prix, remboursement, réservation) et les pharmacies "
               "(téléphone, adresse, horaires, garde). Je ne donne pas de conseil médical.",
        "scope_city": " à {city}", "scope_all": " (toutes villes — précisez une ville pour filtrer)",
        "yes": "{name} est disponible dans {n} pharmacie(s){scope}. Par exemple : {ph}.",
        "no": "Je ne trouve pas {name} en stock{scope} (stock simulé).",
        "ph": "{name} ({city}), tél. {phone}",
        "range": "{name} : de {lo} à {hi} MAD selon la présentation (prix public indicatif, référentiel 2014).",
        "one": "{name} : {x} MAD (prix public indicatif, référentiel 2014).",
        "r70": "{name} : remboursable à {r} %.", "r0": "{name} : non remboursable.",
        "rmix": "{name} : remboursement de {r} % selon la présentation.",
        "res": "Brouillon de réservation : {q} × {name}{at}. Aucune commande n'est réellement envoyée (démonstration).",
        "at": " à la pharmacie {p}",
        "phone": "{name} ({city}) : {phone}.", "loc": "{name} — {address} ({city}).",
        "hours": "{name} : {h}.", "duty": " Pharmacie de garde.",
        "duty_list": "Pharmacies de garde{scope} : {l}.",
        "none": "Je n'ai pas trouvé cette pharmacie dans la base de démonstration.",
    },
    "dl": {
        "oos": "Nقدر n3awnek ghir f l-adwiya (tawafor, taman, t3wid, 7jz) w f s-saydaliyat (numro, 3onwan, w9t, 7ras). "
               "Ma kan3ti 7ta nasi7a tibbiya.".replace("Nقدر", "N9dar"),
        "scope_city": " f {city}", "scope_all": " (f ga3 l-mdon — 9ol liya ina mdina)",
        "yes": "{name} kayn f {n} pharmacie{scope}. Mital : {ph}.",
        "no": "Ma l9itch {name} f l-stock{scope} (stock tajribi).",
        "ph": "{name} ({city}), tel {phone}",
        "range": "{name} : mn {lo} 7ta {hi} dh 7asb l-3olba (taman ta9ribi, 2014).",
        "one": "{name} : {x} dh (taman ta9ribi, 2014).",
        "r70": "{name} : kayt3awed b {r} %.", "r0": "{name} : ma kayt3awedch.",
        "rmix": "{name} : t3wid {r} % 7asb l-3olba.",
        "res": "Mosawwada dyal l-7jz : {q} × {name}{at}. Ma tsayftat 7ta commande 7a9i9iya (demo).",
        "at": " 3nd pharmacie {p}",
        "phone": "{name} ({city}) : {phone}.", "loc": "{name} — {address} ({city}).",
        "hours": "{name} : {h}.", "duty": " Pharmacie de garde.",
        "duty_list": "Pharmacies de garde{scope} : {l}.",
        "none": "Ma l9itch had l-pharmacie f l-base dyal demo.",
    },
    "ar": {
        "oos": "نقدر نعاونك غير فالأدوية (التوفر، الثمن، التعويض، الحجز) والصيدليات (الهاتف، العنوان، الأوقات، الحراسة). "
               "ما كنعطيش نصائح طبية.",
        "scope_city": " فـ{city}", "scope_all": " (فكل المدن — حدد المدينة)",
        "yes": "{name} كاين فـ {n} صيدلية{scope}. مثلا: {ph}.",
        "no": "ما لقيتش {name} فالمخزون{scope} (مخزون تجريبي).",
        "ph": "{name} ({city})، الهاتف: {phone}",
        "range": "{name}: من {lo} حتى {hi} درهم حسب العلبة (ثمن تقريبي، مرجع 2014).",
        "one": "{name}: {x} درهم (ثمن تقريبي، مرجع 2014).",
        "r70": "{name}: كيتعوض بنسبة {r} %.", "r0": "{name}: ما كيتعوضش.",
        "rmix": "{name}: نسبة التعويض {r} % حسب العلبة.",
        "res": "مسودة الحجز: {q} × {name}{at}. ما تسيفطات حتى طلبية فعلية (تجريبي).",
        "at": " عند صيدلية {p}",
        "phone": "{name} ({city}): {phone}.", "loc": "{name} — {address} ({city}).",
        "hours": "{name}: {h}.", "duty": " صيدلية حراسة.",
        "duty_list": "صيدليات الحراسة{scope}: {l}.",
        "none": "ما لقيتش هاد الصيدلية فقاعدة البيانات التجريبية.",
    },
}


def build_reply(r: dict) -> str | None:
    lang, t = _lang(r), None
    t = T[lang]
    if r["status"] == "needs_clarification":
        return r["clarification_question"]
    if r["status"] == "out_of_scope":
        return t["oos"]
    f = r.get("fulfillment")
    if not f:
        return None
    ents, intent = r["entities"], r["intent"]["label"]
    med = ents["medications"][0] if ents["medications"] else None
    name = (med["name"].title() + (f" {med['dosage']}" if med["dosage"] else "")) if med else ""
    city = ents["city"].title() if ents["city"] else None
    scope = t["scope_city"].format(city=city) if city else t["scope_all"]

    if intent in {"check_availability", "reserve_order"}:
        if f.get("available"):
            ph = " ; ".join(t["ph"].format(name=p["name"], city=p["city"].title(), phone=p["phone"]) for p in f["pharmacies"][:2])
            msg = t["yes"].format(name=name, n=len(f["pharmacies"]), scope=scope, ph=ph)
        else:
            msg = t["no"].format(name=name, scope=scope)
        if intent == "reserve_order":
            at = t["at"].format(p=ents["pharmacy_name"]) if ents["pharmacy_name"] else ""
            msg = t["res"].format(q=ents["quantity"] or 1, name=name, at=at) + " " + msg
        return msg
    if intent == "price":
        lo, hi = f["min_ppv_mad"], f["max_ppv_mad"]
        return t["one"].format(name=name, x=_n(lo)) if lo == hi else t["range"].format(name=name, lo=_n(lo), hi=_n(hi))
    if intent == "reimbursement":
        rates = f.get("rates_pct") or []
        if rates == [0]:
            return t["r0"].format(name=name)
        if len(rates) == 1:
            return t["r70"].format(name=name, r=rates[0])
        return t["rmix"].format(name=name, r=" / ".join(map(str, rates))) if rates else t["r0"].format(name=name)
    # pharmacies
    ph = f.get("pharmacies") or []
    if not ph:
        return t["none"]
    if ents.get("time_reference") and "garde" in ents["time_reference"]:
        lst = " ; ".join(t["ph"].format(name=p["name"], city=p["city"].title(), phone=p["phone"]) for p in ph[:3])
        return t["duty_list"].format(scope=scope, l=lst)
    lines = []
    for p in ph[:3]:
        if intent == "pharmacy_phone":
            lines.append(t["phone"].format(name=p["name"], city=p["city"].title(), phone=p["phone"]))
        elif intent == "pharmacy_location":
            lines.append(t["loc"].format(name=p["name"], address=p["address"], city=p["city"].title()))
        else:
            lines.append(t["hours"].format(name=p["name"], h=p["opening_hours"]) + (t["duty"] if p.get("on_duty") else ""))
    return "\n".join(lines)


BOX = {
    "fr": {"ok": "Boîte reconnue : {name}{form} (confiance {pct} %). {price}{reimb}",
           "low": "Je pense reconnaître {name}{form}, mais je n'en suis pas sûr (confiance {pct} %). Pouvez-vous confirmer, ou reprendre la photo ?",
           "unknown": "Je n'ai pas réussi à identifier cette boîte. Reprenez la photo : nom du médicament bien visible, bonne lumière, photo nette.",
           "no_text": "Je ne lis pas de texte sur cette image. Reprenez la photo plus près de la boîte.",
           "price": "Prix public indicatif : {p} MAD (référentiel 2014). ", "reimb": "Remboursement : {r}.", "r0": "non remboursable", "rn": "{r} %",
           "dose": " Dosage non lu : plusieurs existent, vérifiez sur la boîte."},
    "dl": {"ok": "Had l-3olba hiya : {name}{form} (tiqa {pct} %). {price}{reimb}",
           "low": "Kan7sb hiya {name}{form}, walakin ma mt2akedch (tiqa {pct} %). Wach t2akked wla t3awd tsawr ?",
           "unknown": "Ma9dertch n3ref had l-3olba. 3awd tsawr : smiyt d-dwa wad7a, d-daw kafi, w tswira wad7a.",
           "no_text": "Ma knchof 7ta ktaba f had tswira. 3awd tsawr mn qrib.",
           "price": "Taman ta9ribi : {p} dh (2014). ", "reimb": "T3wid : {r}.", "r0": "ma kayt3awedch", "rn": "{r} %",
           "dose": " Ma9drtch n9ra d-dosage : kaynin bzaf, t2akked mn l-3olba."},
    "ar": {"ok": "هاد العلبة هي: {name}{form} (الثقة {pct} %). {price}{reimb}",
           "low": "كنظن هي {name}{form}، ولكن ما متأكدش (الثقة {pct} %). واش تأكد ولا تعاود تصورها؟",
           "unknown": "ما قدرتش نتعرف على هاد العلبة. عاود صورها: اسم الدواء واضح، ضو كافي، وتصويرة واضحة.",
           "no_text": "ما كنشوف حتى كتابة فهاد الصورة. عاود صورها من قريب.",
           "price": "الثمن التقريبي: {p} درهم (2014). ", "reimb": "التعويض: {r}.", "r0": "ما كيتعوضش", "rn": "{r} %",
           "dose": " ما قدرتش نقرا الجرعة: كاينين بزاف، تأكد من العلبة."},
}


def build_box_reply(res: dict, lang: str = "fr") -> str:
    t = BOX.get(lang, BOX["fr"])
    st = res["status"]
    if st in {"no_text", "unknown"} or not res["matches"]:
        return t["no_text"] if st == "no_text" else t["unknown"]
    m = res["matches"][0]
    form = f" ({m['form'].lower()})" if m.get("form") else ""
    pct = round(res["confidence"] * 100)
    adv = " " + " ".join(res["advice"]) if res.get("advice") and st != "ok" else ""
    if st == "low_confidence":
        return t["low"].format(name=m["brand"].title() + (f" {m['dosage']}" if m.get("dosage") else ""), form=form, pct=pct) + adv
    price = t["price"].format(p=_n(m["ppv_mad"])) if m.get("ppv_mad") is not None else ""
    r = m.get("reimbursement_pct")
    reimb = t["reimb"].format(r=t["r0"] if r == 0 else t["rn"].format(r=int(r))) if r is not None else ""
    msg = t["ok"].format(name=m["brand"].title() + (f" {m['dosage']}" if m.get("dosage") else ""), form=form, pct=pct, price=price, reimb=reimb)
    return msg + (t["dose"] if "dosage_non_lu" in res.get("warnings", []) else "")
