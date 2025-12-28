#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import unicodedata
from typing import Dict, List, Any, Tuple, Union
from datetime import datetime

# ---- Dependencias internas (ajusta imports a tu repo) ----
import unicodedata

def deaccent(text: str) -> str:
    if text is None:
        return ""
    # elimina acentos → ASCII
    return unicodedata.normalize("NFKD", str(text)).encode("ASCII", "ignore").decode("ASCII")

from src.utils import ( # type: ignore
    calculate_age_at_graduation,   # <- ya no se usa, pero mantengo import para no tocar más
    CONFIG,
    EDUCATION_MIN_DURATION_YEARS,
    EDUCATION_MAX_DURATION_YEARS,
    EDUCATION_DEFAULT_DURATION,
    WORK_MAX_DURATION_YEARS,
    INT_MAX_DURATION_YEARS,
)

# -------------------- Helpers generales --------------------

_CEFR = {"A1", "A2", "B1", "B2", "C1", "C2"}

def _clean_text(s: Any) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _to_monthyear(s: str) -> Tuple[int, int] | None:
    """Devuelve (YYYY, MM) si s tiene formato MM/YYYY válido."""
    try:
        m = re.match(r"^\s*(\d{2})/(\d{4})\s*$", s)
        if not m:
            return None
        mm, yyyy = int(m.group(1)), int(m.group(2))
        if 1 <= mm <= 12 and 1900 <= yyyy <= datetime.now().year:
            return (yyyy, mm)
    except Exception:
        pass
    return None

def _duration_years_from_mm(s_start: str, s_end: str) -> float:
    try:
        start = _to_monthyear(s_start)
        end = _to_monthyear(s_end)
        if not start or not end:
            return 0.0
        sy, sm = start
        ey, em = end
        if (ey, em) < (sy, sm):
            return 0.0
        return (ey + em / 12.0) - (sy + sm / 12.0)
    except Exception:
        return 0.0

# -------------------- Clasificadores educación --------------------

_MASTER_TERMS = {
    "master", "máster", "msc", "ma", "mba", "postgrado", "posgrado",
    "máster universitario", "m.eng", "meng", "postgraduate", "postgrad", "tfm"
}
# Términos amplios para detectar GRADO (multi-ingeniería + generalistas)
_UG_POSITIVE = {
    "grado", "bachelor", "bsc", "b.eng", "beng",
    "licenciatura", "diplomatura",
    "ingenier", "telecom", "telecommunicat", "telecomunic", "industrial",
    "informat", "computer science", "software", "hardware",
    "aeronáut", "aeronaut", "electr", "mecánic", "mechanic",
    "biomédic", "biomedical", "civil", "químic", "chemical",
    "matemát", "mathemat", "físic", "physic"
}

def _looks_like_master(item: Dict) -> bool:
    fields = _clean_text(item.get("degree", "")) + " " + _clean_text(item.get("field", ""))
    low = deaccent(fields.lower())
    return any(t in low for t in _MASTER_TERMS)

def _looks_like_undergraduate(item: Dict) -> bool:
    fields = _clean_text(item.get("degree", "")) + " " + _clean_text(item.get("field", ""))
    low = deaccent(fields.lower())
    if any(t in low for t in _MASTER_TERMS):
        return False
    return any(t in low for t in _UG_POSITIVE)

def _has_year_range_simple(year_str: str) -> Tuple[int, int] | None:
    """
    Detecta 'YYYY-YYYY' o 'YYYY/YYYY'. Devuelve (start, end) si es válido.
    """
    if not year_str:
        return None
    m = re.match(r"^\s*(\d{4})\s*[-/]\s*(\d{4})\s*$", year_str)
    if not m:
        return None
    sy, ey = int(m.group(1)), int(m.group(2))
    if 1900 <= sy <= ey <= datetime.now().year:
        return sy, ey
    return None

def _open_range_starts(year_str: str) -> int | None:
    """
    Detecta 'YYYY-actualidad|presente|present|current|hoy|now'
    y devuelve el año de inicio si es válido.
    """
    if not isinstance(year_str, str):
        return None
    m = re.match(
        r"^\s*(\d{4})\s*[-/]\s*(actualidad|presente|present|current|hoy|now)\s*$",
        year_str,
        flags=re.IGNORECASE
    )
    if not m:
        return None
    sy = int(m.group(1))
    if 1900 <= sy <= datetime.now().year:
        return sy
    return None

def _is_single_year_only_item(item: Dict) -> bool:
    """
    True si el ítem educativo NO tiene start/end, NO hay rango explícito, y aparece exactamente un año suelto.
    """
    if item.get("start_date") or item.get("end_date"):
        return False
    candidates = [
        item.get("year"), item.get("degree"),
        item.get("field"), item.get("university"),
        item.get("description"), item.get("location"),
    ]
    # descarta si aparece un rango:
    for t in candidates:
        if isinstance(t, str) and _has_year_range_simple(t):
            return False
    # cuenta años sueltos:
    years = []
    for t in candidates:
        if not isinstance(t, str):
            continue
        years.extend(re.findall(r"(19[5-9]\d|20[0-3]\d)", t))
    return len(set(years)) == 1 and len(years) >= 1

def _looks_like_international_edu(item: Dict) -> bool:
    """
    Marca entradas de education que deberían ser 'international':
    - contiene 'erasmus' en cualquier campo
    - degree vacío y 'year' con meses (p.ej. 'Feb-2008/Sep-2008') o intervalo corto con mes
    """
    blob = " ".join(_clean_text(item.get(k, "")) for k in ("degree", "field", "university", "location", "year")).lower()
    if "erasmus" in blob:
        return True
    # mes-YYYY / mes-YYYY
    if re.search(r"(ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)[a-z\-]*/?\s*\d{4}", blob, flags=re.IGNORECASE):
        # si no hay grado explícito, sugiere estancia
        if not _clean_text(item.get("degree", "")):
            return True
    return False

# -------------------- Idiomas → CEFR --------------------

_LANG_STANDARD = {
    'ingles': 'English', 'english': 'English',
    'espanol': 'Spanish', 'español': 'Spanish', 'spanish': 'Spanish',
    'frances': 'French', 'français': 'French', 'french': 'French',
    'aleman': 'German', 'deutsch': 'German', 'german': 'German',
    'portugues': 'Portuguese', 'portuguese': 'Portuguese',
    'italiano': 'Italian', 'italian': 'Italian',
}

_LEVEL_MAP_KEYS = [
    # ordenados de más específicos a genéricos
    ("c2", "C2"), ("c1", "C1"),
    ("b2", "B2"), ("b1", "B1"),
    ("a2", "A2"), ("a1", "A1"),
    ("native", "C2"), ("nativo", "C2"), ("materna", "C2"), ("bilingue", "C2"),
    ("fluent", "C1"), ("avanz", "C1"), ("alto", "C1"), ("advanced", "C1"), ("excellent", "C1"),
    ("intermedio", "B2"), ("medio", "B2"), ("intermediate", "B2"), ("good", "B2"),
    ("basico", "A2"), ("basic", "A2"), ("beginner", "A1"), ("elemental", "A1"),
]

def _normalize_languages(languages_list: List[Any]) -> List[List[str]]:
    if not isinstance(languages_list, list):
        return []

    out: List[List[str]] = []
    seen: set[Tuple[str, str]] = set()

    for lang in languages_list:
        lang_name = ""
        level_raw = ""

        if isinstance(lang, dict):
            lang_name = _clean_text(lang.get("language", ""))
            level_raw = _clean_text(lang.get("level", ""))
        elif isinstance(lang, (list, tuple)) and len(lang) >= 2:
            lang_name = _clean_text(lang[0])
            level_raw = _clean_text(lang[1])
        elif isinstance(lang, str):
            parts = re.split(r"\s*[-—]\s*|\s+", lang.strip(), maxsplit=1)
            lang_name = _clean_text(parts[0])
            level_raw = _clean_text(parts[1] if len(parts) > 1 else "")

        if not lang_name:
            continue

        key = deaccent(lang_name.lower())
        norm_name = _LANG_STANDARD.get(key, lang_name.capitalize())

        lvl_low = deaccent(level_raw.lower())
        mapped = None
        for needle, cefr in _LEVEL_MAP_KEYS:
            if needle in lvl_low:
                mapped = cefr
                break
        if mapped is None:
            # sin pista clara → descarta nivel
            continue

        tup = (norm_name, mapped)
        if tup not in seen:
            out.append([norm_name, mapped])
            seen.add(tup)

    return out

# -------------------- Deduplicación international --------------------

def _dedup_international(items: List[Dict]) -> List[Dict]:
    """
    Dedup por clave canónica: (type, institution, country, start_date, end_date)
    """
    deduped = []
    seen: set[Tuple[str, str, str, str, str]] = set()
    for it in items:
        t = _clean_text(it.get("type", "")).lower() or "study"
        inst = _clean_text(it.get("institution_or_company", "")).lower()
        ctry = _clean_text(it.get("country", "")).lower()
        sd = _clean_text(it.get("start_date", "")).lower()
        ed = _clean_text(it.get("end_date", "")).lower()
        key = (t, inst, ctry, sd, ed)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    return deduped

# -------------------- has_master --------------------

def _detect_has_master(education_list: List[Dict]) -> bool:
    for it in education_list or []:
        if isinstance(it, dict) and _looks_like_master(it):
            return True
    return False

# -------------------- fit_score (con penalización ÚNICA pedida) --------------------

def calculate_fit_score(cv: Dict) -> float:
    """
    Scoring:
      +5 si hay grado/carrera
      +0.5 por experiencia laboral (por item), máx 2.0
      +1.0 si existe alguna experiencia internacional (binario)
      Idiomas: todos suman según nivel CEFR, con mayor peso al inglés (máx 2.0)
      +0.5 por herramienta relevante (CONFIG), máx 2.0
      Penalización ÚNICA:
        -0.5 si NO hay NINGUNA experiencia laboral NI experiencia internacional
    """
    score = 0.0

    # ----- 1) Grado / carrera -----
    has_degree = any(_looks_like_undergraduate(e) for e in cv.get("education_normalized", []))
    if has_degree:
        score += 5.0

    # ----- 2) Experiencia laboral -----
    n_exp = len(cv.get("work_experience_normalized", []) or [])
    score += min(n_exp * 0.5, 2.0)

    # ----- 3) Internacional -----
    intl = cv.get("international_experience_normalized", []) or []
    if len(intl) > 0:
        score += 1.0

    # ----- 4) Idiomas (nuevo criterio) -----
    # pesos base por nivel CEFR
    level_weights = {
        "C2": 1.0,
        "C1": 0.8,
        "B2": 0.6,
        "B1": 0.4,
        "A2": 0.2,
        "A1": 0.1,
    }

    lang_score = 0.0
    for pair in cv.get("languages_normalized", []):
        if not (isinstance(pair, list) and len(pair) == 2):
            continue
        lang_name, level = pair[0], str(pair[1]).upper()
        base = level_weights.get(level)
        if base is None:
            continue

        # detectar inglés con y sin acento / en inglés
        lang_key = deaccent(str(lang_name).lower())
        is_english = any(
            token in lang_key
            for token in ["ingles", "english"]
        )

        if is_english:
            base *= 1.5  # más peso para inglés

        lang_score += base

    # capar contribución total de idiomas
    lang_score = min(lang_score, 2.0)
    score += lang_score

    # ----- 5) Herramientas técnicas relevantes -----
    relevant = 0
    tools = (cv.get("hard_and_soft_skills_normalized", {}) or {}).get("hard_skills", [])
    rel_tools = [t.lower() for t in CONFIG.get("hard_skills", {}).get("relevant_hard_skills", [])]
    for t in tools:
        tnorm = deaccent(str(t).lower())
        if any(rt in tnorm for rt in rel_tools):
            relevant += 1
    score += min(relevant * 0.5, 2.0)

    # ----- 6) Penalización única si no hay nada de experiencia ni internacional -----
    if n_exp == 0 and len(intl) == 0:
        score -= 0.5

    # limitar a [0, 10]
    return max(0.0, min(round(score, 2), 10.0))

# -------------------- Helpers para AgeAtGraduation --------------------

_YEAR_RE = re.compile(r"(19\d{2}|20[0-3]\d)")

def _year_from_any(s: str) -> int | None:
    """Extrae un año (YYYY) de cualquier cadena si existe y es razonable."""
    if not isinstance(s, str):
        return None
    m = _YEAR_RE.search(s)
    if not m:
        return None
    y = int(m.group(1))
    if 1900 <= y <= datetime.now().year + 1:
        return y
    return None

def _end_year_from_edu_item(ed: Dict) -> int | None:
    """
    Intenta obtener el año de finalización de una entrada educativa.
    Prioridad: end_date (MM/YYYY o YYYY) > year (rangos YYYY-YYYY o YYYY/YYYY o año suelto).
    """
    # 1) end_date
    end_raw = _clean_text(ed.get("end_date", ""))
    if end_raw:
        mm_yyyy = _to_monthyear(end_raw)
        if mm_yyyy:
            return mm_yyyy[0]  # YYYY
        y = _year_from_any(end_raw)
        if y:
            return y

    # 2) year (rangos o año suelto)
    yr = _clean_text(ed.get("year", ""))
    if yr:
        # YYYY-YYYY o YYYY/YYYY
        m = re.match(r"^\s*(\d{4})\s*[-/]\s*(\d{4})\s*$", yr)
        if m:
            return int(m.group(2))
        # YYYY-actualidad|present|...
        m2 = re.match(
            r"^\s*(\d{4})\s*[-/]\s*(actualidad|presente|present|current|hoy|now)\s*$",
            yr, flags=re.IGNORECASE
        )
        if m2:
            # No está finalizado → usa None (otro código decidirá el fallback)
            return None
        # un año suelto
        y = _year_from_any(yr)
        if y:
            return y

    return None

def _end_year_undergraduate(education_norm: List[Dict]) -> int | None:
    """
    Devuelve el mayor 'end year' encontrado entre entradas que parecen GRADO.
    """
    years = []
    for ed in education_norm or []:
        try:
            if isinstance(ed, dict) and _looks_like_undergraduate(ed):
                y = _end_year_from_edu_item(ed)
                if y:
                    years.append(y)
        except Exception:
            continue
    return max(years) if years else None

def _compute_age_at_graduation(cv_data: Dict, education_norm: List[Dict], edition_hint: str) -> int:
    """
    Regla pedida:
      - AgeAtGraduation = end_year(UNDERGRAD) - birth_year
      - Si birth_year NO está claro/presente → devolver 22
      - Si end_year no se puede obtener de GRADO → usa 22 como fallback para no romper features
      - Clamp a rango razonable; si queda raro, devuelve 22
    """
    # 1) birth year
    dob = (cv_data.get("personal_information_validated", {}) or {}).get("date_of_birth") \
          or (cv_data.get("personal_information", {}) or {}).get("date_of_birth", "")
    birth_year = _year_from_any(_clean_text(dob))
    if birth_year is None:
        return 22

    # 2) end year (solo GRADO)
    end_year = _end_year_undergraduate(education_norm)
    if end_year is None:
        # No está claro el fin de grado → fallback seguro
        return 22

    age = int(end_year) - int(birth_year)
    if age < 16 or age > 80:
        return 22
    return age

# -------------------- Normalización principal --------------------

def normalize_llm_cv_output(cv_data: Dict) -> Dict:
    """
    Devuelve SOLO los bloques *_normalized y métricas agregadas.
    - Saca estancias/Erasmus de education → international_experience_normalized
    - CEFR en idiomas
    - Dedup international
    - Calcula degree_years y has_master
    """
    raw_text = _clean_text(cv_data.get("raw_text", ""))
    filename = _clean_text(cv_data.get("filename", ""))
    masi_edition = _clean_text(cv_data.get("MASI_Edition", cv_data.get("master_edition", ""))) or str(datetime.now().year)
    detected_language = _clean_text(cv_data.get("detected_language", "unknown"))

    # --- Personal information ---
    pi_src = cv_data.get("personal_information_validated") or cv_data.get("personal_information") or {}
    pi_norm: Dict[str, Any] = {}
    for k, v in pi_src.items():
        if k in {"email", "phone", "address"}:
            pi_norm[k] = _clean_text(v)
        else:
            pi_norm[k] = v

    # --- Education: separar internacional ---
    edu_src = cv_data.get("education_validated") or cv_data.get("education") or []
    education_norm: List[Dict[str, Any]] = []
    international_buf: List[Dict[str, Any]] = []

    for item in edu_src:
        if not isinstance(item, dict):
            continue
        ed = {k: (v if k in {"start_date","end_date","year"} else _clean_text(v)) for k, v in item.items()}

        # mover a internacional si aplica
        if _looks_like_international_edu(ed):
            international_buf.append({
                "type": "Erasmus" if "erasmus" in deaccent(" ".join([ed.get("year",""), ed.get("field",""), ed.get("degree","")]).lower()) else "Study",
                "country": _clean_text(ed.get("location", "")),
                "institution_or_company": _clean_text(ed.get("university", "")),
                "start_date": _clean_text(ed.get("year", "")),
                "end_date": None,
                "description": _clean_text(ed.get("field", "")) or None,
                "duration_years": 0.0
            })
            continue
        
        # calcular duración con reglas + fallbacks
        dur = 0.0
        yr = _clean_text(ed.get("year", ""))
        rng = _has_year_range_simple(yr)
        open_start = _open_range_starts(yr)

        is_master = _looks_like_master(ed)
        is_ug = _looks_like_undergraduate(ed)

        if rng:
            # 'YYYY-YYYY'
            sy, ey = rng
            dur = float(ey - sy)
        elif open_start is not None:
            # 'YYYY-actualidad' --> regla TFG: asumir 5 años
            dur = 5.0
        elif _is_single_year_only_item(ed) and is_ug:
            # un único año suelto pero parece GRADO --> media TFG: 5 años
            dur = 5.0
        else:
            # conserva si ya venía alguna duración previa
            try:
                prev = float(ed.get("duration_years", 0.0) or 0.0)
            except Exception:
                prev = 0.0
            dur = max(dur, prev)

        # Fallback final: si es universitario (grado o máster) y sigue en 0.0 -> 5 años
        if dur == 0.0 and (is_master or is_ug):
            dur = 5.0

        # Clamp a rango razonable
        dur = min(max(dur, EDUCATION_MIN_DURATION_YEARS), EDUCATION_MAX_DURATION_YEARS)
        ed["duration_years"] = round(dur, 2)
        education_norm.append(ed)

    # --- Work experience ---
    work_src = cv_data.get("work_experience_validated") or cv_data.get("work_experience") or []
    work_norm: List[Dict[str, Any]] = []
    tot_work = 0.0
    for w in work_src:
        if not isinstance(w, dict):
            continue
        wn = {k: (v if k in {"start_date","end_date"} else _clean_text(v)) for k, v in w.items()}
        start = _clean_text(wn.get("start_date", ""))
        end = _clean_text(wn.get("end_date", ""))
        if start and not end:
            # si no hay fin, usa edición (diciembre) para cerrar
            try:
                ey = int(re.findall(r"(19|20)\d{2}", masi_edition)[0])
            except Exception:
                ey = datetime.now().year
            end = f"12/{ey}"
            wn["end_date"] = end
        dur = _duration_years_from_mm(start, end) if start and end else 0.0
        wn["duration_years"] = round(min(max(dur, 0.0), WORK_MAX_DURATION_YEARS), 2)
        tot_work += wn["duration_years"]
        work_norm.append(wn)

    # --- International experience ---
    intl_src = cv_data.get("international_experience_validated") or cv_data.get("international_experience") or []
    intl_norm: List[Dict[str, Any]] = []
    # 1) lo que ya venía
    for it in intl_src:
        if not isinstance(it, dict):
            continue
        intl_norm.append({
            "type": _clean_text(it.get("type", "")) or "Study",
            "country": _clean_text(it.get("country", "")),
            "institution_or_company": _clean_text(it.get("institution_or_company", "")),
            "start_date": _clean_text(it.get("start_date", "")),
            "end_date": _clean_text(it.get("end_date", "")),
            "description": _clean_text(it.get("description", "")) or None,
            "duration_years": 0.0
        })
    # 2) lo migrado desde education
    intl_norm.extend(international_buf)
    # 3) dedup
    intl_norm = _dedup_international(intl_norm)

    # --- Languages ---
    langs_src = cv_data.get("languages_validated") or cv_data.get("languages") or []
    languages_norm = _normalize_languages(langs_src)

    # --- Skills ---
    kt_src = (
        cv_data.get("hard_and_soft_skills_validated")
        or cv_data.get("hard_and_soft_skills")
        or {}
    )

    if not isinstance(kt_src, dict):
        kt_src = {}

    raw_hard = (
        kt_src.get("hard_skills")
        or kt_src.get("hard skills")
        or []
    )
    raw_soft = (
        kt_src.get("soft_skills")
        or kt_src.get("soft skills")
        or []
    )

    hard_skills_norm = [
        _clean_text(x) for x in raw_hard if _clean_text(x)
    ]
    soft_skills_norm = [
        _clean_text(x) for x in raw_soft if _clean_text(x)
    ]

    hard_and_soft_skills_norm = {
        "hard_skills": hard_skills_norm,
        "soft_skills": soft_skills_norm,
    }

    # --- Otros / Voluntariado (limpios) ---
    other_norm = [ _clean_text(x) for x in (cv_data.get("other_interests_validated") or cv_data.get("other_interests") or []) if _clean_text(x) ]
    volunteering_src = cv_data.get("volunteering_validated") or cv_data.get("volunteering") or []
    volunteering_norm = []
    for v in volunteering_src:
        if not isinstance(v, dict):
            continue
        volunteering_norm.append({k: _clean_text(v.get(k, "")) for k in ("organization","role","duration","description")})

    # --- degree_years (solo educación universitaria: grado y/o máster) ---
    degree_years = 0.0
    for ed in education_norm:
        if _looks_like_undergraduate(ed) or _looks_like_master(ed):
            try:
                degree_years += float(ed.get("duration_years", 0.0) or 0.0)
            except Exception:
                pass
    if degree_years == 0.0:
        degree_years = 5.0
    degree_years = round(min(max(degree_years, 0.0), EDUCATION_MAX_DURATION_YEARS), 2)

    # --- has_master ---
    has_master = _detect_has_master(education_norm)

    # --- Age at graduation (solo GRADO); si falta nacimiento → 22
    age_at_graduation = _compute_age_at_graduation(cv_data, education_norm, masi_edition)

    # --- NUEVO: bandera binaria de experiencia internacional ---
    has_international_experience = len(intl_norm) > 0

    # --- payload final SOLO normalizado ---
    final_payload = {
        "personal_information_normalized": pi_norm,
        "education_normalized": education_norm,
        "work_experience_normalized": work_norm,
        "international_experience_normalized": intl_norm,
        "languages_normalized": languages_norm,
        "hard_and_soft_skills_normalized": hard_and_soft_skills_norm,
        "other_interests_normalized": other_norm,
        "volunteering_normalized": volunteering_norm,

        # métricas agregadas
        "degree_years": degree_years,
        "total_work_years": round(min(tot_work, WORK_MAX_DURATION_YEARS), 2),
        "has_master": has_master,
        "has_international_experience": has_international_experience,
        "age_at_graduation": age_at_graduation,

        # meta
        "detected_language": detected_language or "unknown",
        "master_edition": masi_edition if masi_edition else None,
        "refined": bool(cv_data.get("refined", False)),
        "processed_and_standardized": True,
    }

    final_payload["fit_score"] = calculate_fit_score(final_payload)
    return final_payload
