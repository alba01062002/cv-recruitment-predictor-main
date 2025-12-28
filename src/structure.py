import os
import json
from typing import Dict, List, Any, Optional
import re
from datetime import datetime
import unicodedata

# =========================
# Helpers comunes de fechas
# =========================

_EDITION_RE = re.compile(r"MASI\s*([0-9]{2})", re.IGNORECASE)

def _edition_to_year(master_edition: Optional[str]) -> Optional[int]:
    if not master_edition:
        return None
    m = _EDITION_RE.search(str(master_edition))
    if not m:
        return None
    yy = int(m.group(1))
    return 2000 + yy if yy <= 39 else 1900 + yy

def parse_date(date_str: str) -> Optional[datetime]:
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        if '-' in date_str and not date_str.strip().isdigit():
            date_str = date_str.split('-')[-1].strip()
        for fmt in ['%Y', '%b %Y', '%B %Y', '%m/%Y', '%Y/%m']:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
    except Exception:
        return None

def calculate_years(start_date: str, end_date: str) -> float:
    if not start_date or not end_date:
        return 0.0
    start = parse_date(start_date)
    end = parse_date(end_date)
    if start and end:
        return (end - start).days / 365.25
    return 0.0

# ==============================================
# Normalización de "actualidad/presente" (exper.)
# ==============================================

_PRESENT_TERMS = {"actualidad", "presente", "trabajo actual", "actual", "hoy", "now", "present", "current"}

def _strip_accents_lower(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII").lower().strip()

def _is_present_token(s: Optional[str]) -> bool:
    t = _strip_accents_lower(s or "")
    if not t:
        return False
    return (t in _PRESENT_TERMS) or t.endswith(" actualidad")

def _mentions_present(*texts: Optional[str]) -> bool:
    for tx in texts:
        t = _strip_accents_lower(tx or "")
        for term in _PRESENT_TERMS:
            if term in t:
                return True
    return False

def _patch_work_end_dates_with_edition(work_list: list, edition_year: Optional[int]) -> None:
    if not isinstance(work_list, list) or not edition_year:
        return
    end_mmYYYY = f"12/{edition_year}"
    for item in work_list:
        if not isinstance(item, dict):
            continue
        endd = item.get("end_date")
        if isinstance(endd, str) and _is_present_token(endd):
            item["end_date"] = end_mmYYYY
            continue
        if (not endd) and _mentions_present(item.get("description"), item.get("position"), item.get("company")):
            item["end_date"] = end_mmYYYY

# =========================================================
# Educación: detección grado/máster y cálculo degree_years
# =========================================================

_YEAR_RE = re.compile(r"(19[5-9]\d|20[0-3]\d)")
_RANGE_RE = re.compile(r"(19[5-9]\d|20[0-3]\d)\s*[-–—]\s*(19[5-9]\d|20[0-3]\d)")

def _has_year_range(text: Optional[str]) -> bool:
    if not text or not isinstance(text, str):
        return False
    return bool(_RANGE_RE.search(text))

def _extract_years_from_text(text: Optional[str]) -> List[int]:
    if not text or not isinstance(text, str):
        return []
    return [int(y) for y in _YEAR_RE.findall(text)]

def _is_single_year_only(edu: Dict) -> bool:
    if not isinstance(edu, dict):
        return False
    if edu.get("start_date") or edu.get("end_date"):
        return False
    candidates = [
        edu.get("year"),
        edu.get("degree"),
        edu.get("field"),
        edu.get("university"),
        edu.get("description"),
        edu.get("location"),
    ]
    if any(_has_year_range(t) for t in candidates if isinstance(t, str)):
        return False
    years_found: List[int] = []
    for t in candidates:
        years_found.extend(_extract_years_from_text(t))
    return len(set(years_found)) == 1 and len(years_found) >= 1

def _duration_from_year_field(year_str: Optional[str]) -> float:
    if not year_str or not isinstance(year_str, str):
        return 0.0
    s = year_str.strip()
    if "-" in s:
        a, b = [p.strip() for p in s.split("-", 1)]
        if a.isdigit() and b.isdigit():
            try:
                return max(0, int(b) - int(a))
            except Exception:
                return 0.0
    return 0.0

# Máster (positivos con límites de palabra) + exclusiones
_MASTER_POS_RE = [
    re.compile(r"\b(m[áa]ster|master)\b", re.IGNORECASE),
    re.compile(r"\bmsc\b", re.IGNORECASE),
    re.compile(r"\bmba\b", re.IGNORECASE),
    re.compile(r"\bm\.?eng\b", re.IGNORECASE),
    re.compile(r"\bmeng\b", re.IGNORECASE),
    re.compile(r"\bpost\s?grado\b", re.IGNORECASE),
    re.compile(r"\bpos\s?grado\b", re.IGNORECASE),
    re.compile(r"\bpostgraduate\b", re.IGNORECASE),
    re.compile(r"\bpostgrad\b", re.IGNORECASE),
    re.compile(r"\bmaster of [a-z]+", re.IGNORECASE),
]
_MASTER_NEG_RE = [
    re.compile(r"\bdea\b", re.IGNORECASE),
    re.compile(r"diploma de estudios avanzados", re.IGNORECASE),
    re.compile(r"suficiencia investigadora", re.IGNORECASE),
    re.compile(r"t[ée]cnico superior", re.IGNORECASE),
    re.compile(r"\bfp\b", re.IGNORECASE),
    re.compile(r"formaci[oó]n profesional", re.IGNORECASE),
    re.compile(r"\bcurso(s)?\b", re.IGNORECASE),
    re.compile(r"\bcertificado\b", re.IGNORECASE),
    re.compile(r"\bexpert[oa]\b", re.IGNORECASE),
]

_ENGINEERING_STEMS = {
    "ingenier", "engineering", "engineer",
    "telecomunic", "teleco", "telecommunication",
    "informat", "computer", "software", "comput", "telematic",
    "electr", "automat", "control", "sistemas", "system", "robot",
    "mecan", "mechanic",
    "industrial",
    "civil",
    "quimic", "chemical",
    "aeroespac", "aerospace", "aeronaut",
    "biomed",
    "material",
    "naval", "ocean",
    "mina", "mining",
    "energia", "energy", "energet"
}
_UG_HINTS = {"grado", "bachelor", "bsc", "licenciatura", "diplomatura", "b.eng", "beng"}

def _looks_like_master(edu: Dict) -> bool:
    if not isinstance(edu, dict):
        return False
    fields = " ".join([
        str(edu.get("degree") or ""),
        str(edu.get("field") or "")
    ])
    t = _strip_accents_lower(fields)
    if any(rx.search(t) for rx in _MASTER_NEG_RE):
        return False
    return any(rx.search(t) for rx in _MASTER_POS_RE)

def _looks_like_undergraduate(edu: Dict) -> bool:
    if not isinstance(edu, dict):
        return False
    fields = " ".join([
        str(edu.get("degree") or ""),
        str(edu.get("field") or "")
    ])
    t = _strip_accents_lower(fields)
    if _looks_like_master(edu):
        return False
    if re.search(r"\b(ingenier[oa]|engineering|engineer)\b", t):
        return True
    if any(stem in t for stem in _ENGINEERING_STEMS):
        return True
    if any(h in t for h in _UG_HINTS) and any(stem in t for stem in _ENGINEERING_STEMS):
        return True
    return False

# =========================
# Normalización principal
# =========================

def normalize_languages(languages: Any) -> List[tuple]:
    language_map = {
        'alto': 'C1', 'medio': 'B2', 'bajo': 'B1', 'nativo': 'C2',
        'avanzado': 'C1', 'intermedio': 'B2', 'básico': 'A2',
        'bilingüe': 'C2', 'native': 'C2'
    }
    normalized = []
    seen = set()

    if not isinstance(languages, list):
        return []

    for lang in languages:
        if isinstance(lang, dict):
            language = str(lang.get('language', '') or '')
            level = str(lang.get('level', '') or '')
            if language and level:
                level = language_map.get(level.lower(), level)
                entry = (language, level)
                if entry not in seen:
                    normalized.append(entry)
                    seen.add(entry)
        elif isinstance(lang, str) and lang:
            parts = re.split(r'\s+|-', lang.strip(), maxsplit=1)
            if len(parts) >= 1:
                language = parts[0].strip()
                level = parts[1].strip() if len(parts) == 2 else 'Unknown'
                level = language_map.get(level.lower(), level)
                entry = (language, level)
                if entry not in seen:
                    normalized.append(entry)
                    seen.add(entry)

    return normalized

def normalize_llm_cv_output(cv_data: Dict) -> Dict:
    normalized_data = dict(cv_data) if isinstance(cv_data, dict) else {}

    # Inicializar campos obligatorios
    normalized_data['personal_information'] = normalized_data.get('personal_information', {}) or {}
    normalized_data['education'] = normalized_data.get('education', [])
    normalized_data['work_experience'] = normalized_data.get('work_experience', [])
    normalized_data['hard_and_soft_skills'] = normalized_data.get('hard_and_soft_skills', [])
    normalized_data['other_interests'] = normalized_data.get('other_interests', [])
    normalized_data['volunteering'] = normalized_data.get('volunteering', [])
    normalized_data['languages'] = normalized_data.get('languages', [])

    # Forzar tipos correctos
    if not isinstance(normalized_data['education'], list):
        normalized_data['education'] = []
    if not isinstance(normalized_data['work_experience'], list):
        normalized_data['work_experience'] = []
    if not isinstance(normalized_data['hard_and_soft_skills'], dict):
        normalized_data['hard_and_soft_skills'] = {}
    if not isinstance(normalized_data['other_interests'], list):
        normalized_data['other_interests'] = []
    if not isinstance(normalized_data['volunteering'], list):
        normalized_data['volunteering'] = []
    if not isinstance(normalized_data['languages'], list):
        normalized_data['languages'] = []

    # 1) Normaliza "actualidad" (EXP)
    edition_year = _edition_to_year(normalized_data.get('master_edition'))
    _patch_work_end_dates_with_edition(normalized_data['work_experience'], edition_year)

    # 2) Educación: degree_years (solo grado)
    degree_years = 0.0
    for edu in normalized_data['education']:
        if not isinstance(edu, dict):
            continue

        # Garantiza claves básicas
        for key in ['degree', 'field', 'university', 'location', 'year']:
            if key not in edu or edu[key] is None:
                edu[key] = ''

        # calcular/rellenar duration_years del item
        dur = _duration_from_year_field(edu.get('year'))
        if not dur:
            # MM/YYYY o YYYY
            def _p(d):
                if not d or not isinstance(d, str):
                    return None
                d = d.strip()
                for fmt in ("%m/%Y", "%Y"):
                    try:
                        return datetime.strptime(d, fmt)
                    except Exception:
                        continue
                return None
            sdt = _p(edu.get("start_date"))
            edt = _p(edu.get("end_date"))
            if sdt and edt:
                dur = max(0.0, (edt - sdt).days / 365.25)
        if not dur and _is_single_year_only(edu):
            dur = 5.0

        prev = edu.get('duration_years')
        if not isinstance(prev, (int, float)) or float(prev) <= 0.0:
            edu['duration_years'] = round(dur, 2)

        # sumar SOLO si parece grado
        if _looks_like_undergraduate(edu):
            degree_years += dur

    normalized_data['degree_years'] = round(degree_years, 2)
    normalized_data.pop('total_education_years', None)  # limpiar legado

    # 3) Experiencia: duraciones
    total_work_years = 0.0
    for exp in normalized_data['work_experience']:
        if not isinstance(exp, dict):
            continue
        for key in ['company', 'position', 'description', 'start_date', 'end_date']:
            if key not in exp or exp[key] is None:
                exp[key] = ''
        duration = calculate_years(exp['start_date'], exp['end_date'])
        exp['duration_years'] = round(duration, 2)
        total_work_years += duration

    normalized_data['total_work_years'] = round(total_work_years, 2)

    # 4) Idiomas
    normalized_data['languages'] = normalize_languages(normalized_data['languages'])

    # 5) has_master (robusto con regex + exclusiones)
    has_master = False
    for edu in normalized_data['education']:
        if _looks_like_master(edu):
            has_master = True
            break
    normalized_data['has_master'] = has_master

    # Other interests
    normalized_data['other_interests'] = list(set([
        str(item) for item in normalized_data['other_interests']
        if isinstance(item, (str, int, float)) and str(item).strip()
    ]))

    # Volunteering
    for vol in normalized_data['volunteering']:
        if isinstance(vol, dict):
            for key in ['organization', 'role', 'description']:
                if key not in vol or vol[key] is None:
                    vol[key] = ''

    normalized_data['processed_and_standardized'] = True
    return normalized_data

def save_to_json(data: Dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"File saved successfully at: {file_path}")
    except Exception as e:
        print(f"Error saving file {file_path}: {e}")