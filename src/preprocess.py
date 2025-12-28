import pandas as pd
import unicodedata
import re
import logging
from typing import Optional, List, Tuple
from fuzzywuzzy import fuzz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- utilidades de normalización de nombres ----------

_PARTICLES = {"de", "del", "de la", "de los", "de las", "la", "las", "los", "y", "da", "das", "do", "dos"}

def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")

def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def normalize_name(name: str) -> str:
    """
    Normaliza un nombre: quita acentos, pasa a minúsculas y limpia espacios/puntuación básica.
    Conserva el orden dado.
    """
    if not isinstance(name, str):
        return ""
    s = _strip_accents(name).lower()
    s = re.sub(r"[^\w\s]", " ", s)          # quita puntuación
    s = _clean_spaces(s)
    return s

def _tokens(name: str) -> List[str]:
    """
    Tokeniza y filtra partículas poco informativas (“de/del/de la/…”).
    """
    n = normalize_name(name)
    toks = n.split()
    # fusiona bigramas de partículas (e.g., "de la")
    i = 0
    out = []
    while i < len(toks):
        if i+1 < len(toks) and f"{toks[i]} {toks[i+1]}" in _PARTICLES:
            i += 2
            continue
        if toks[i] not in _PARTICLES:
            out.append(toks[i])
        i += 1
    return out

def _name_variants(cv_full_name: str) -> List[str]:
    """
    Genera variantes robustas del nombre del CV:
      - tal cual (normalizado)
      - “apellidos, nombre” -> “nombre apellidos”
      - reordenaciones por tokens
    """
    norm = normalize_name(cv_full_name)
    variants = set()
    variants.add(norm)

    # Si viene “apellidos, nombre”
    if "," in cv_full_name:
        # intenta “apellidos, nombre(s)” → “nombre(s) apellidos”
        parts = [p.strip() for p in re.split(r",", cv_full_name, maxsplit=1)]
        if len(parts) == 2:
            left, right = normalize_name(parts[0]), normalize_name(parts[1])
            if left and right:
                variants.add(_clean_spaces(f"{right} {left}"))

    # Variante por orden “nombre(s) apellidos” con tokens filtrados
    toks = _tokens(cv_full_name)
    if len(toks) >= 2:
        # primera palabra como nombre y resto como apellidos
        v1 = _clean_spaces(" ".join([toks[0]] + toks[1:]))
        variants.add(v1)
        # apellidos primero y luego nombre (por si el CSV lo tiene así)
        v2 = _clean_spaces(" ".join(toks[1:] + [toks[0]]))
        variants.add(v2)

    return list(variants)

# ------------------------------------------------------------

def is_non_cv_content(raw_text: str, file_path: str = "unknown_file") -> bool:
    non_cv_indicators = [
        "sign in to continue",
        "captcha",
        "not your computer",
        "private browsing",
        "access denied",
        "404 not found",
        "cookie policy",
        "terms and conditions",
        "this page cannot be displayed",
        "this site uses cookies",
        "javascript required",
        "enable cookies",
        "web scraping detected",
        "robot check"
    ]
    for indicator in non_cv_indicators:
        if indicator.lower() in raw_text.lower():
            logger.warning(f"Non-CV content detected in {file_path} (indicator: '{indicator}'). Processing skipped.")
            return True
    if len(raw_text.strip()) < 50:
        if not re.search(r'\b(nombre|name|educacion|education|experiencia|experience|email|telefono|phone|cv|curriculum)\b', raw_text.lower()):
            logger.warning(f"Extremely short raw_text without obvious CV keywords for {file_path}. Processing skipped. Length: {len(raw_text.strip())}")
            return True
    if not raw_text.strip():
        logger.warning(f"Empty raw_text for {file_path}. Processing skipped.")
        return True
    return False

def generate_labels(cv_data: dict, becarios_df: pd.DataFrame, csv_name: Optional[str] = None) -> pd.Series:
    """
    Empareja automáticamente el nombre del CV con el CSV de seleccionados (sin prompts).
    Reglas:
      1) Igualdad fuerte por token set (100) → match.
      2) token_set_ratio >= 92 → match.
      3) Si hay varios >= 92, elige el mayor score; en empate, el de más tokens compartidos.
      4) Si ningún candidato alcanza 92, outcome = 0.
    Maneja formatos “Apellidos, Nombre”, acentos, partículas y ordenes distintos.
    """
    full_name = (cv_data.get('personal_information', {}) or {}).get('full_name', '').strip()
    filename = cv_data.get('filename', '')
    if not full_name:
        logger.warning(f"No full_name in CV data for {filename}. Defaulting to outcome 0.")
        return pd.Series({'filename': filename, 'name': '', 'hiring_outcome': 0, 'matched_name': ''})

    if becarios_df is None or becarios_df.empty:
        logger.warning(f"Empty becarios_df for {filename} in {csv_name or 'becarios_df'}. Defaulting to outcome 0.")
        return pd.Series({'filename': filename, 'name': full_name, 'hiring_outcome': 0, 'matched_name': ''})

    # Aseguramos columnas normalizadas
    required_cols = {'normalized_full_name', 'normalized_first_name', 'normalized_apellidos'}
    if not required_cols.issubset(becarios_df.columns):
        logger.warning(f"becarios_df lacks normalized columns for {filename}. Defaulting to outcome 0.")
        return pd.Series({'filename': filename, 'name': full_name, 'hiring_outcome': 0, 'matched_name': ''})

    # Variantes del nombre del CV
    variants = _name_variants(full_name)
    # Prepara lista de candidatos con su nombre normalizado
    cand_names = becarios_df['normalized_full_name'].fillna("").astype(str).tolist()

    best_score = -1
    best_match = ""
    best_shared = -1

    cv_tokens_sets = [set(_tokens(v)) for v in variants]

    for v, v_tokens in zip(variants, cv_tokens_sets):
        # Regla 1: igualdad fuerte por conjunto de tokens
        for cand in cand_names:
            cand_tokens = set(_tokens(cand))
            if cand_tokens and cand_tokens == v_tokens and len(v_tokens) >= 2:
                logger.info(f"Strong token-set equality for '{full_name}' ~ '{cand}' in {csv_name or 'becarios_df'}.")
                return pd.Series({'filename': filename, 'name': full_name, 'hiring_outcome': 1, 'matched_name': cand})

        # Regla 2: token_set_ratio
        for cand in cand_names:
            score = fuzz.token_set_ratio(v, cand)
            if score > best_score:
                best_score = score
                best_match = cand
                best_shared = len(set(_tokens(cand)) & v_tokens)
            elif score == best_score:
                # desempate por tokens compartidos
                shared = len(set(_tokens(cand)) & v_tokens)
                if shared > best_shared:
                    best_match = cand
                    best_shared = shared

    # Umbral de aceptación
    if best_score >= 92:
        logger.info(f"Auto-match for '{full_name}' ~ '{best_match}' (score={best_score}) in {csv_name or 'becarios_df'}.")
        return pd.Series({'filename': filename, 'name': full_name, 'hiring_outcome': 1, 'matched_name': best_match})
    else:
        logger.info(f"No sufficient match for '{full_name}' (best score={best_score}) in {csv_name or 'becarios_df'}. Outcome 0.")
        return pd.Series({'filename': filename, 'name': full_name, 'hiring_outcome': 0, 'matched_name': ''})