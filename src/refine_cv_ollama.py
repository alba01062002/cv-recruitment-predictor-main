import json
import re
import subprocess
import os
from typing import Dict, Optional, Tuple, List

# --- NEW: helpers for edition year & age-at-edition ---
from datetime import datetime
import unicodedata  # para quitar acentos en 'actualidad'

_EDITION_RE = re.compile(r"MASI\s*([0-9]{2})", re.IGNORECASE)

def _edition_to_year(master_edition: Optional[str]) -> Optional[int]:
    if not master_edition:
        return None
    m = _EDITION_RE.search(str(master_edition))
    if not m:
        return None
    yy = int(m.group(1))
    # MASI01..MASI39 -> 2001..2039; MASI90..99 -> 1990..1999 (por seguridad)
    return 2000 + yy if yy <= 39 else 1900 + yy

def _extract_birth_year(dob: Optional[str]) -> Optional[int]:
    """Acepta 'DD/MM/YYYY', 'MM/YYYY', 'YYYY', etc. Devuelve el año si es plausible."""
    if not dob or not isinstance(dob, str):
        return None
    years = re.findall(r"(19[2-9]\d|20[0-3]\d)", dob)  # 1920–2039
    if years:
        y = int(years[-1])  # último suele ser el año correcto en DD/MM/YYYY
        return y
    return None

def _compute_age_at_year(birth_year: Optional[int], ref_year: Optional[int]) -> Optional[int]:
    if not birth_year or not ref_year:
        return None
    age = ref_year - birth_year
    return age if 14 <= age <= 80 else None
# --- END helpers edad ---

# --- NEW: helpers to normalize "actualidad/presente" in work_experience ---
_PRESENT_TERMS = {"actualidad", "presente", "trabajo actual", "actual", "hoy", "now", "present", "current"}

def _strip_accents_lower(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII").lower().strip()

def _is_present_token(s: Optional[str]) -> bool:
    t = _strip_accents_lower(s or "")
    return t in _PRESENT_TERMS or t.endswith(" actualidad")

def _mentions_present(*texts: Optional[str]) -> bool:
    for tx in texts:
        t = _strip_accents_lower(tx or "")
        for term in _PRESENT_TERMS:
            if term in t:
                return True
    return False

def _patch_work_end_dates_with_edition(work_list: list, edition_year: Optional[int]) -> None:
    """
    Reemplaza end_date 'actualidad/present/...' por 12/<EDITION_YEAR>.
    Si end_date está vacío pero se menciona actualidad en el texto del item, también lo fija.
    No hace nada si edition_year es None.
    """
    if not isinstance(work_list, list) or not edition_year:
        return
    end_mmYYYY = f"12/{edition_year}"
    for item in work_list:
        if not isinstance(item, dict):
            continue
        endd = item.get("end_date")
        # Caso 1: end_date explícito con presente/actualidad
        if isinstance(endd, str) and _is_present_token(endd):
            item["end_date"] = end_mmYYYY
            continue
        # Caso 2: end_date vacío y el texto del item sugiere actualidad
        if (not endd) and _mentions_present(item.get("description"), item.get("position"), item.get("company")):
            item["end_date"] = end_mmYYYY
# --- END helpers actualidad ---

# --- NEW: utilidades educación / degree_years ---
_YEAR_RE = re.compile(r"(19[5-9]\d|20[0-3]\d)")
_RANGE_RE = re.compile(r"(19[5-9]\d|20[0-3]\d)\s*[-–—]\s*(19[5-9]\d|20[0-3]\d)")

def _extract_years_from_text(text: Optional[str]) -> List[int]:
    if not text or not isinstance(text, str):
        return []
    return [int(y) for y in _YEAR_RE.findall(text)]

def _has_year_range(text: Optional[str]) -> bool:
    if not text or not isinstance(text, str):
        return False
    return bool(_RANGE_RE.search(text))

def _is_single_year_only_item(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("start_date") or item.get("end_date"):
        return False
    candidates = [
        item.get("year"),
        item.get("degree"),
        item.get("field"),
        item.get("university"),
        item.get("description"),
        item.get("location"),
    ]
    if any(_has_year_range(t) for t in candidates if isinstance(t, str)):
        return False
    years_found = []
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

# Detección grado vs máster (robusta / generalista ingeniería)
# Positivos de MÁSTER con límites de palabra
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
# Exclusiones que NO son máster
_MASTER_NEG_RE = [
    re.compile(r"\bdea\b", re.IGNORECASE),  # Diploma de Estudios Avanzados
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

def _looks_like_master(item: dict) -> bool:
    txt = _strip_accents_lower(" ".join([
        str(item.get("degree") or ""),
        str(item.get("field") or "")
    ]))
    if any(rx.search(txt) for rx in _MASTER_NEG_RE):
        return False
    return any(rx.search(txt) for rx in _MASTER_POS_RE)

def _looks_like_undergraduate(item: dict) -> bool:
    txt = _strip_accents_lower(" ".join([
        str(item.get("degree") or ""),
        str(item.get("field") or "")
    ]))
    if _looks_like_master(item):
        return False
    if re.search(r"\b(ingenier[oa]|engineering|engineer)\b", txt):
        return True
    if any(stem in txt for stem in _ENGINEERING_STEMS):
        return True
    if any(h in txt for h in _UG_HINTS) and any(stem in txt for stem in _ENGINEERING_STEMS):
        return True
    return False

def _detect_has_master(education_list: list) -> bool:
    if not isinstance(education_list, list):
        return False
    for it in education_list:
        if isinstance(it, dict) and _looks_like_master(it):
            return True
    return False
# --- END utilidades educación ---

class QuotaExceededError(Exception):
    """Compat con tu pipeline (en local no debería saltar)."""
    pass

# Preferencia de modelos
DEFAULT_MODEL_PREFERENCE = [
    "qwen2.5:7b-instruct", 
    "deepseek-r1:7b",       
    "phi3:mini",
]

def _extract_json(text: str) -> str:
    """Extrae JSON de un texto que puede venir con ```json ... ``` o texto + JSON."""
    if not text:
        raise ValueError("Respuesta vacía del modelo.")
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1].strip()
    return text.strip()

def _run_ollama(model: str, prompt: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            check=True
        )
        return result.stdout.strip(), None
    except subprocess.CalledProcessError as e:
        err = e.stderr.strip() if e.stderr else f"ollama run {model} failed with returncode {e.returncode}"
        return None, err
    except Exception as e:
        return None, f"Unexpected error running ollama: {e}"

def refine_cv_with_llm(cv_data: Dict) -> Dict:
    """
    Procesa raw_text con Ollama para devolver un JSON de CV estructurado.
    Preserva master_edition si viene del extractor.
    """
    raw_text = cv_data.get("raw_text", "") or ""
    header_name = (
        cv_data.get("text_from_header")
        or cv_data.get("full_name_from_header")
        or ""
    )
    master_edition = cv_data.get("master_edition")

    if not raw_text:
        print("Warning: No 'raw_text' found in cv_data. Refinement skipped.")
        out = {**cv_data, "refined": False}
        if master_edition and "master_edition" not in out:
            out["master_edition"] = master_edition
        return out

    model_preference = [os.getenv("OLLAMA_MODEL")] if os.getenv("OLLAMA_MODEL") else DEFAULT_MODEL_PREFERENCE

    prompt = f"""
    Analyze this curriculum vitae in Spanish or English. Your task is to extract relevant information and structure it into a JSON object.

    The text has been extracted from a document and may contain OCR or formatting errors. Correct them and complete a JSON with the structure below. Use null for missing fields. Respond ONLY with valid JSON (no extra text).

    JSON structure to fill:
    {{
      "personal_information": {{
        "full_name": "Full name",
        "date_of_birth": "DD/MM/YYYY",
        "gender": "male/female (infer strictly from the person's name and linguistic cues in the CV text)"
        "age_reference_year": "YYYY"
        "age": null,
        "email": "email@example.com",
        "phone": "Phone number",
        "address": "Full address (e.g., Calle Example 123, Madrid, Spain)",
        "LinkedIn": "LinkedIn URL",
        "Instagram": "Instagram URL", 
        "Twitter": "Twitter URL",
        "website": "Personal website URL",
      }},
      "education": [
        {{
          "degree": "Academic degree (e.g., Master's in...)",
          "field": "Field of study (e.g., Aircraft Systems Integration)",
          "university": "Institution name (e.g., Universidad Carlos III Madrid)",
          "location": "City",
          "year": "Completion year or range (e.g., 2020-2022)"
        }}
      ],
      "work_experience": [
        {{
          "company": "Company name",
          "position": "Job title",
          "description": "Brief description of role and responsibilities",
          "start_date": "MM/YYYY",
          "end_date": "MM/YYYY"
        }}
      ],
      "international_experience": [
        {{
          "type": "Erasmus / Internship / Work / Volunteering / Research / Study",
          "country": "Country name",
          "institution_or_company": "Institution or company name",
          "start_date": "MM/YYYY or null",
          "end_date": "MM/YYYY or null",
          "description": "Brief description of the international experience"
        }}
      ],
      
      "projects_performed": [
        {{
        "title": "Project title",
        "type": "TFG / TFM / Master / Research / Professional / Personal",
        "institution": "Institution name",
        "start_date": "MM/YYYY",
        "end_date": "MM/YYYY",
        "description": "Brief description of the project.

        }}
      ],
      "languages": [
        {{
          "language": "Language",
          "level": "Level (e.g., native, bilingual, advanced, intermediate, basic)"
        }}
      ],
        
      "hard_and_soft_skills": {{
        "hard skills": ["Skill 1", "Skill 2"],
        "soft skills": ["Tool 1", "Tool 2"]
      }}
      ],

      "volunteering": [
        {{
          "organization": "Organization name",
          "role": "Role in volunteer work",
          "duration": "Duration (e.g., 2020-2022)"
        }}
      ],

      "other_interests": ["Hobby 1", "Sport 1", "Personal interest 1"],
      "fit_score": 0.0,
      "refined": true
    }}

    Specific instructions:
    - Use only information from raw_text. You must not invent information.
    - Hard skills: technical or theoretical abilities: programming languages, tools, etc. Examples: Excel, MATLAB, AutoCAD, SolidWorks, CATIA, Python
    - Gender: You must write female or male, do not leave this field empty.

    CV text:
    {raw_text}
    """

    last_error = None
    for model in model_preference:
        out, err = _run_ollama(model, prompt)
        if err:
            print(f"[Ollama] {model} error: {err}")
            last_error = err
            continue

        try:
            json_string = _extract_json(out)
            refined_json = json.loads(json_string)
            if isinstance(refined_json, dict):
                refined_json.setdefault("refined", True)
                if master_edition and "master_edition" not in refined_json:
                    refined_json["master_edition"] = master_edition
                # --- Añadidos deterministas ---
                try:
                    # 1) Edad respecto a edición
                    pi = refined_json.setdefault("personal_information", {})
                    edition_year = _edition_to_year(refined_json.get("master_edition") or master_edition)
                    birth_year = _extract_birth_year(pi.get("date_of_birth"))
                    age_at = _compute_age_at_year(birth_year, edition_year)
                    pi["age_reference_year"] = edition_year if edition_year is not None else None
                    pi["age"] = age_at if age_at is not None else None

                    # 2) Normaliza "actualidad" en experiencia
                    _patch_work_end_dates_with_edition(refined_json.get("work_experience", []), edition_year)

                    # 3) degree_years (sumar SOLO los grados)
                    degree_total = 0.0
                    edu_list = refined_json.get("education", [])
                    if isinstance(edu_list, list):
                        for it in edu_list:
                            if not isinstance(it, dict):
                                continue
                            looks_ug = _looks_like_undergraduate(it)
                            if not looks_ug:
                                continue

                            # intentar leer/poner duración
                            dur = 0.0
                            # a) year: "YYYY-YYYY"
                            dur = _duration_from_year_field(it.get("year"))
                            # b) start/end
                            if not dur:
                                sd = it.get("start_date")
                                ed = it.get("end_date")
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
                                sdt = _p(sd)
                                edt = _p(ed)
                                if sdt and edt:
                                    dur = max(0.0, (edt - sdt).days / 365.25)
                            # c) heurística: un año suelto ⇒ 5.0
                            if not dur and _is_single_year_only_item(it):
                                dur = 5.0

                            # reflejar en el item si no venía bien
                            prev = it.get("duration_years")
                            if not isinstance(prev, (int, float)) or float(prev) <= 0.0:
                                it["duration_years"] = round(dur, 2)

                            degree_total += dur

                    refined_json["degree_years"] = round(degree_total, 2)

                    # 4) has_master (robusto con regex y exclusiones)
                    refined_json["has_master"] = _detect_has_master(refined_json.get("education", []))

                    # 5) limpieza legado
                    refined_json.pop("total_education_years", None)

                except Exception:
                    pass

                return refined_json
            else:
                print(f"[Ollama] {model} devolvió un tipo no dict. Probando siguiente modelo...")
        except Exception as e:
            print(f"[Ollama] {model} devolvió salida no-JSON parseable: {e}. Probando siguiente modelo...")
            last_error = str(e)

    print(f"[Ollama] Todos los modelos fallaron. Último error: {last_error}")
    out = {**cv_data, "refined": False, "ollama_error": last_error or "Unknown"}
    if master_edition and "master_edition" not in out:
        out["master_edition"] = master_edition
    # --- Fallback: edad, 'actualidad', degree_years (sumando grados) y has_master ---
    try:
        pi = out.setdefault("personal_information", {})
        edition_year = _edition_to_year(out.get("master_edition") or master_edition)
        birth_year = _extract_birth_year(pi.get("date_of_birth"))
        age_at = _compute_age_at_year(birth_year, edition_year)
        pi["age_reference_year"] = edition_year if edition_year is not None else None
        pi["age"] = age_at if age_at is not None else None

        _patch_work_end_dates_with_edition(out.get("work_experience", []), edition_year)

        degree_total = 0.0
        edu_list = out.get("education", [])
        if isinstance(edu_list, list):
            for it in edu_list:
                if not isinstance(it, dict):
                    continue
                if not _looks_like_undergraduate(it):
                    continue
                dur = _duration_from_year_field(it.get("year"))
                if not dur:
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
                    sdt = _p(it.get("start_date"))
                    edt = _p(it.get("end_date"))
                    if sdt and edt:
                        dur = max(0.0, (edt - sdt).days / 365.25)
                if not dur and _is_single_year_only_item(it):
                    dur = 5.0
                prev = it.get("duration_years")
                if not isinstance(prev, (int, float)) or float(prev) <= 0.0:
                    it["duration_years"] = round(dur, 2)
                degree_total += dur

        out["degree_years"] = round(degree_total, 2)
        out["has_master"] = _detect_has_master(out.get("education", []))
        out.pop("total_education_years", None)

    except Exception:
        pass

    return out