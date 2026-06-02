#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature encoding for normalized CVs (TF-IDF + Dense Features).

Mejoras implementadas:
1. HIBRIDACIÓN: Combina TF-IDF (Texto) con Features Numéricas (Dense) extraídas de normalization.py:
   - degree_years
   - age_at_graduation
   - has_international_experience
   - total_work_years
   - has_master
   - n_hard_skills
   - n_languages
   - english_level_numeric (0=None, 1=A1... 6=C2/Native)
   (fit_score EXCLUIDO por petición usuario)

2. LIMPIEZA DE TEXTO:
   - Eliminación agresiva de años (19xx, 20xx) para evitar sesgo temporal.
   - Eliminación de dígitos sueltos.
   - Stop words básicos.

3. OPTIMIZACIÓN TF-IDF:
   - ngram_range=(1, 2) para capturar "Data Scientist", "Project Manager".
   - min_df=2 para reducir ruido de typos únicos.
"""

import os
import re
import json
import csv
import logging
import unicodedata
from typing import List, Dict, Tuple, Optional, Set, Any

from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

import nltk
from nltk.corpus import stopwords
try:
    STOP_WORDS = list(set(stopwords.words('english') + stopwords.words('spanish')))
except:
    STOP_WORDS = 'english'

logger = logging.getLogger(__name__)

# -------------------- helpers de normalización de nombres --------------------
def _deaccent(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")

def _beautify_name(name: str) -> str:
    if not name:
        return ""
    name = name.strip().strip('"').strip("'")
    if "," in name:
        left, right = [p.strip() for p in name.split(",", 1)]
        if left and right:
            name = f"{right} {left}"
    name = re.sub(r"\s+", " ", name).strip()
    return name

def _norm_name_key(name: str) -> str:
    if not name:
        return ""
    n = _beautify_name(name)
    n = _deaccent(n).lower()
    n = re.sub(r"[^\w\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n

# -------------------- Limpieza de Texto Mejorada --------------------
# Regex para años 1980-2029 (evitar overfit a fechas)
_YEAR_TOKEN_RE = re.compile(r"\b(19|20)\d{2}\b")
# Regex para dígitos sueltos o pegados a letras simples (e.g. "1", "2nd")
_DIGIT_RE = re.compile(r"\b\d+\w*\b")

def _clean_token_text(text: str) -> str:
    """Limpieza profunda para evitar ruido temporal y numérico."""
    if not text:
        return ""
    # 1. Deaccent y lower
    t = _deaccent(text).lower()
    # 2. Quitar años (clave para evitar sesgo MASI25 vs anteriores)
    t = _YEAR_TOKEN_RE.sub(" ", t)
    # 3. Quitar números/dígitos irrelevantes
    t = _DIGIT_RE.sub(" ", t)
    # 4. Quitar puntuación residual
    t = re.sub(r"[^\w\s]", " ", t)
    # 5. Colapsar espacios
    t = re.sub(r"\s+", " ", t).strip()
    return t

# -------------------- parsers/flatten --------------------
def _flatten_any(x: Any) -> List[str]:
    """Aplana str/list/tuple/set/dict a lista de strings."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple, set)):
        out: List[str] = []
        for el in x:
            out.extend(_flatten_any(el))
        return out
    if isinstance(x, dict):
        out: List[str] = []
        for v in x.values():
            out.extend(_flatten_any(v))
        return out
    return [str(x)]

def _join_fields(obj, keys: List[str]) -> str:
    parts: List[str] = []
    for k in keys:
        v = obj.get(k)
        if not v:
            continue
        parts.extend(_flatten_any(v))
    
    # Pre-clean antes de unir para que no queden huecos dobles
    cleaned_parts = [_clean_token_text(p) for p in parts]
    return " ".join(p for p in cleaned_parts if p).strip()

# --- extracción de SKILLS (sin knowledge_and_tools) ---
_SKILL_FIELD_CANDIDATES_TOP = [
    "skills_normalized", "skills",
    "hard_and_soft_skills_normalized", "hard_and_soft_skills",
    "hard_skills", "soft_skills",
]
_SKILL_FIELD_CANDIDATES_EXTRA = [
    "competencias", "competences", "competencies",
    "technical_skills", "tech_skills", "tech_stack", "stack",
    "programming_languages", "languages_programming",
    "certifications", "certificates", "courses"
]

def _extract_skills_text(cv: Dict) -> str:
    candidates: List[Any] = []
    # 1) campos principales
    for k in _SKILL_FIELD_CANDIDATES_TOP:
        if k in cv and cv[k]:
            candidates.append(cv[k])

    # 2) hard_and_soft_skills(_normalized) con 'hard_skills'/'soft_skills'
    hss = cv.get("hard_and_soft_skills_normalized") or cv.get("hard_and_soft_skills")
    if isinstance(hss, dict):
        for subk in ("hard_skills", "soft_skills"):
            if subk in hss and hss[subk]:
                candidates.append(hss[subk])

    # 3) alias extra frecuentes
    for k in _SKILL_FIELD_CANDIDATES_EXTRA:
        if k in cv and cv[k]:
            candidates.append(cv[k])

    tokens: List[str] = []
    for c in candidates:
        # A diferencia de antes, limpiamos cada token
        raw_list = _flatten_any(c)
        tokens.extend([_clean_token_text(t) for t in raw_list])
    
    tokens = [t for t in tokens if t]
    tokens = list(dict.fromkeys(tokens))  # dedup preservando orden
    return " ".join(tokens)

# -------------------- Extracción de Features DENSAS --------------------
def _convert_level_to_int(lvl: str) -> int:
    """Convierte CEFR/Names a escala numérica 1-6."""
    lvl = lvl.upper().strip()
    if lvl in ("C2", "NATIVE", "NATIVO", "BILINGUE"): return 6
    if lvl in ("C1", "ADVANCED"): return 5
    if lvl in ("B2", "INTERMEDIATE"): return 4
    if lvl in ("B1",): return 3
    if lvl in ("A2", "BASIC"): return 2
    if lvl in ("A1", "BEGINNER"): return 1
    return 0

def _extract_dense_features_dict(cv: Dict) -> Dict[str, float]:
    """
    Extrae variables numéricas (dense) en formato Dict para mapeo flexible.
    """
    # 1. Campos directos del normalizador
    degree_years = float(cv.get("degree_years", 5.0))
    age_at_graduation = float(cv.get("age_at_graduation", 22.0))
    total_work_years = float(cv.get("total_work_years", 0.0))
    has_intl = 1.0 if cv.get("has_international_experience") else 0.0
    has_mst = 1.0 if cv.get("has_master") else 0.0
    
    # 2. Métricas calculadas al vuelo extra
    # n_hard_skills
    hss = cv.get("hard_and_soft_skills_normalized") or {}
    n_hard = len(hss.get("hard_skills", []))
    
    # n_languages y max_english
    langs = cv.get("languages_normalized", [])
    n_langs = len(langs)
    max_eng = 0
    for litem in langs:
        if isinstance(litem, list) and len(litem) >= 2:
            name, level = litem[0], litem[1]
            if "english" in _deaccent(name.lower()) or "ingles" in _deaccent(name.lower()):
                score = _convert_level_to_int(str(level))
                if score > max_eng:
                    max_eng = score
    
    return {
        "degree_years": degree_years,
        "age_at_graduation": age_at_graduation,
        "total_work_years": total_work_years,
        "has_international_experience": has_intl,
        "has_master": has_mst,
        "n_hard_skills": float(n_hard),
        "n_languages": float(n_langs),
        "max_english_level": float(max_eng)
    }

# Mapping: Section -> List of dense keys
DENSE_SECTION_MAPPING = {
    "education": ["degree_years", "age_at_graduation", "has_master"],
    "work": ["total_work_years"],
    "international": ["has_international_experience"],
    "skills": ["n_hard_skills"],
    "languages": ["n_languages", "max_english_level"]
}

# -------------------- carga de labels --------------------
def _load_labels_map(labels_file: str) -> Dict[str, int]:
    if not os.path.isfile(labels_file):
        logger.error(f"Labels file not found: {labels_file}")
        return {}

    try:
        with open(labels_file, "r", encoding="utf-8") as f:
            sniffer = csv.Sniffer()
            sample = f.read(4096)
            f.seek(0)
            has_header = sniffer.has_header(sample)
            dialect = sniffer.sniff(sample) if sample else csv.excel
            reader = csv.reader(f, dialect)
            rows = list(reader)
        if rows:
            if has_header and len(rows[0]) >= 2 and rows[0][0].strip().lower() == "name":
                labels = {}
                for r in rows[1:]:
                    if not r: continue
                    name = (r[0] or "").strip()
                    lab  = (r[1] or "0").strip()
                    key = _norm_name_key(name)
                    # Gestión robusta de labels no enteros
                    try:
                        labels[key] = int(lab)
                    except:
                        labels[key] = 0
                return labels
            else:
                labels = {}
                for r in rows:
                    if not r: continue
                    line = " ".join(c for c in r if c is not None).strip()
                    if not line: continue
                    m = re.match(r"^(.*\S)\s+([01])\s*$", line)
                    if m:
                        name = m.group(1); lab = m.group(2)
                        labels[_norm_name_key(name)] = int(lab)
                return labels
    except Exception as e:
        logger.warning(f"CSV parse fallback for {labels_file}: {e}")

    # Fallback txt
    labels = {}
    try:
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = (line or "").strip()
                if not line: continue
                m = re.match(r"^(.*\S)\s+([01])\s*$", line)
                if m:
                    name = m.group(1); lab = m.group(2)
                    labels[_norm_name_key(name)] = int(lab)
    except Exception as e:
        logger.error(f"Failed to read labels as lines: {e}")
        return {}
    return labels

# -------------------- iterador JSON --------------------
def _iter_normalized_jsons(normalized_dir: str):
    for fn in sorted(os.listdir(normalized_dir)):
        if fn.endswith("_normalized.json"):
            path = os.path.join(normalized_dir, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                yield fn, data
            except Exception as e:
                logger.warning(f"Skip unreadable {path}: {e}")

def _get_display_name(cv: Dict, fallback: str) -> str:
    pi_norm = cv.get("personal_information_normalized", {}) or {}
    pi_raw  = cv.get("personal_information", {}) or {}
    return pi_norm.get("full_name") or pi_raw.get("full_name") or fallback

# -------------------- construcción de textos --------------------
def _cv_to_text_combined(cv: Dict) -> str:
    """
    Constructs a 'Clean & Contextual' combined text.
    1. Removes verbose 'description' fields to reduce noise.
    2. Adds prefixes (EDU_, JOB_, etc.) to allow the model to distinguish context.
    """
    
    parts = []

    # Helper to prefix tokens
    def _add_section(prefix: str, text_list: List[str]):
        # Join, clean, then prefix every token
        raw = " ".join(text_list)
        clean = _clean_token_text(raw)
        if not clean:
            return
        # Prefix each word (filtering stopwords first)
        prefixed = []
        for w in clean.split():
            if w in STOP_WORDS:
                continue
            prefixed.append(f"{prefix}{w}")
        parts.extend(prefixed)

    # Education (Exclude description)
    for e in cv.get("education_normalized", []) or cv.get("education", []):
        # Focus on Degree, Field, Uni - highly relevant signals
        _add_section("education::", _flatten_any(_join_fields(e, ["degree", "field", "university", "location"])))

    # Work (Exclude description)
    for w in cv.get("work_experience_normalized", []) or cv.get("work_experience", []):
        # Focus on Role/Company
        _add_section("work::", _flatten_any(_join_fields(w, ["company", "position"])))

    # International (Exclude description)
    for it in cv.get("international_experience_normalized", []) or cv.get("international_experience", []):
        _add_section("international::", _flatten_any(_join_fields(it, ["type", "country", "institution_or_company"])))

    # Languages (High Value)
    lang_txts = []
    for l in cv.get("languages_normalized", []) or cv.get("languages", []):
        lang_txts.extend(_flatten_any(l))
    _add_section("languages::", lang_txts)

    # Skills (Already extracted, just clean and add)
    skills_txt = _extract_skills_text(cv)
    _add_section("skills::", [skills_txt])

    # Others / volunteer (Exclude description)
    others = _flatten_any(cv.get("other_interests_normalized") or cv.get("other_interests") or [])
    _add_section("other::", others)
    
    for v in (cv.get("volunteering_normalized") or cv.get("volunteering") or []):
        _add_section("volunteer::", _flatten_any(_join_fields(v, ["organization","role"])))

    base = " ".join(parts).strip()
    
    if not base:
        # Fallback to raw if empty
        base = _clean_token_text(cv.get("raw_text") or "")
        
    return base

def _cv_to_text_separate(cv: Dict) -> Dict[str, str]:
    sections: Dict[str, str] = {}

    def join_list_dict(lst, keys):
        txt = " ".join(_join_fields(x, keys) for x in lst) if lst else ""
        return _clean_token_text(txt)

    sections["education"] = join_list_dict(
        cv.get("education_normalized", []) or cv.get("education", []),
        ["degree","field","university","location","description"] # Quitamos "year"
    )
    sections["work"] = join_list_dict(
        cv.get("work_experience_normalized", []) or cv.get("work_experience", []),
        ["company","position","description","location"]
    )
    sections["international"] = join_list_dict(
        cv.get("international_experience_normalized", []) or cv.get("international_experience", []),
        ["type","country","institution_or_company","description"]
    )

    # languages
    langs_tokens: List[str] = []
    for l in cv.get("languages_normalized", []) or cv.get("languages", []):
        langs_tokens.extend(_flatten_any(l))
    sections["languages"] = _clean_token_text(" ".join(t for t in langs_tokens if t))

    # skills
    sections["skills"] = _clean_token_text(_extract_skills_text(cv))

    # others
    sections["other"] = _clean_token_text(" ".join(_flatten_any(cv.get("other_interests_normalized") or cv.get("other_interests") or [])))

    # volunteer
    sections["volunteer"] = join_list_dict(
        (cv.get("volunteering_normalized") or cv.get("volunteering") or []),
        ["organization","role","description"]
    )

    for k, v in list(sections.items()):
        sections[k] = v.strip() if v and v.strip() else ""
    return sections

# -------------------- principal --------------------
def encode_features(
    *,
    vectorization_mode: str,
    normalized_dir: str,
    features_dir: str,
    models_dir: str,
    labels_file: str,
    allowed_name_keys: Optional[Set[str]] = None
) -> bool:
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1) Cargar labels
    labels_map = _load_labels_map(labels_file)
    if not labels_map:
        logger.error(f"Labels not loaded from {labels_file}.")
        return False

    # 2) Cargar CVs
    filenames: List[str] = []
    name_keys: List[str] = []
    texts_combined: List[str] = []
    texts_sections: Dict[str, List[str]] = {}
    
    # Updated: Store dense as list of dicts for flexible mapping
    # dense_dicts_list: List[Dict[str, float]] = [] (Not needed for logic change logic)
    dense_dicts_list: List[Dict[str, float]] = []
    
    skipped_by_filter = 0

    for fn, cv in _iter_normalized_jsons(normalized_dir):
        display = _get_display_name(cv, fn.replace("_normalized.json", ""))
        key = _norm_name_key(display)

        if allowed_name_keys is not None and key not in allowed_name_keys:
            skipped_by_filter += 1
            continue

        name_keys.append(key)
        filenames.append(fn)

        # TEXT
        if vectorization_mode == "combined":
            texts_combined.append(_cv_to_text_combined(cv))
        else:
            secs = _cv_to_text_separate(cv)
            if not texts_sections:
                for s in secs.keys(): texts_sections[s] = []
            for s in texts_sections.keys():
                texts_sections[s].append(secs.get(s, ""))
        
        # DENSE
        dense_dicts_list.append(_extract_dense_features_dict(cv))

    n_docs = len(filenames)
    if n_docs == 0:
        logger.error(f"No valid CV data loaded from {normalized_dir}.")
        return False
    logger.info(f"Loaded {n_docs} CVs (skipped={skipped_by_filter}).")

    # 3) Vectorizar Texto (TF-IDF)
    feature_names_global: List[str] = []
    
    # Config mejorada: binary=True (presence), sublinear_tf=True, min_df=2 (limpieza ruido), ngram_range=(1,2), bilingual stopwords
    base_args = dict(min_df=2, ngram_range=(1, 2), stop_words=STOP_WORDS, binary=True, sublinear_tf=True) 

    if vectorization_mode == "combined":
        # Updated Regex to allow colons '::' in tokens
        vec = TfidfVectorizer(max_features=10000, 
                              min_df=2, 
                              ngram_range=(1, 2), 
                              binary=True, 
                              sublinear_tf=True, 
                              token_pattern=r"(?u)\b[\w:]+\b")
        try:
            X_text = vec.fit_transform(texts_combined)
            joblib.dump(vec, os.path.join(models_dir, "tfidf_combined.joblib"))
            try:
                feature_names_global = list(vec.get_feature_names_out())
            except:
                feature_names_global = [f"text_{i}" for i in range(X_text.shape[1])]
        except ValueError:
             vec = TfidfVectorizer(min_df=1)
             X_text = vec.fit_transform(texts_combined)
             feature_names_global = list(vec.get_feature_names_out())

        # For combined mode, we DO NOT append dense features (User Request)
        # X_dense = np.array(X_dense_vals, dtype=float)
        # scaler = StandardScaler()
        # X_dense_scaled = scaler.fit_transform(X_dense)
        # joblib.dump(scaler, os.path.join(models_dir, "dense_scaler_combined.joblib"))

        # X_final = hstack([X_text, csr_matrix(X_dense_scaled)]).tocsr()
        # feature_names_global.extend([f"dense_{k}" for k in all_dense_keys])
        
        X_final = X_text # Just text

    else:
        # SEPARATE MODE: Integrate dense into sections
        mats = []
        
        # Helper to get dense matrix for a section
        def _get_section_dense_matrix(section_name: str) -> Optional[Tuple[csr_matrix, List[str], Any]]:
            if section_name not in DENSE_SECTION_MAPPING:
                return None
            
            keys = DENSE_SECTION_MAPPING[section_name]
            # Extract values
            vals = []
            for d in dense_dicts_list:
               vals.append([d[k] for k in keys])
            
            X_d = np.array(vals, dtype=float)
            
            # Scale
            scl = StandardScaler()
            X_d_scl = scl.fit_transform(X_d)
            # joblib.dump(scl, os.path.join(models_dir, f"dense_scaler_{section_name}.joblib")) # Deprecated
            
            feat_names = [f"{section_name}::dense_{k}" for k in keys]
            return csr_matrix(X_d_scl), feat_names, scl

        for s, docs in texts_sections.items():
            # 1. TF-IDF
            if not any(t.strip() for t in docs):
                docs = ["cv"] * len(docs)
                
            vec = TfidfVectorizer(max_features=4000, **base_args)
            try:
                Xs = vec.fit_transform(docs)
            except ValueError:
                vec = TfidfVectorizer(min_df=1)
                Xs = vec.fit_transform(docs)
            
            # 2. Append Dense (if any for this section)
            dense_res = _get_section_dense_matrix(s)
            scaler_obj = None
            
            if dense_res:
                Xd, fd_names, scaler_obj = dense_res
                Xs_combined = hstack([Xs, Xd])
                
                f_names = []
                try:
                    f_names = [f"{s}::{t}" for t in vec.get_feature_names_out()]
                except:
                    f_names = [f"{s}::text_{i}" for i in range(Xs.shape[1])]
                    
                f_names.extend(fd_names)
                mats.append(Xs_combined)
                feature_names_global.extend(f_names)
            else:
                mats.append(Xs)
                f_names = []
                try:
                    f_names = [f"{s}::{t}" for t in vec.get_feature_names_out()]
                except:
                    f_names = [f"{s}::text_{i}" for i in range(Xs.shape[1])]
                feature_names_global.extend(f_names)

            # SAVE UNIFIED PROCESSOR
            # processor_{section}.joblib
            joblib.dump({
                "section": s,
                "tfidf": vec,
                "scaler": scaler_obj
            }, os.path.join(models_dir, f"processor_{s}.joblib"))

        X_final = hstack(mats).tocsr()

    # 6) Labels
    y = np.zeros(n_docs, dtype=int)
    for i, k in enumerate(name_keys):
        if k in labels_map:
            y[i] = int(labels_map[k])
        else:
            y[i] = 0

    # 7) Guardar
    save_npz(os.path.join(features_dir, "X.npz"), X_final)
    np.save(os.path.join(features_dir, "y.npy"), y)
    with open(os.path.join(features_dir, "filenames.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(filenames))
    with open(os.path.join(features_dir, "name_keys.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(name_keys))
    with open(os.path.join(features_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feature_names_global))

    logger.info(f"Saved Hybrid features (Text+Dense-Integrated) to {features_dir} "
                f"(X: {X_final.shape}, positives: {int(y.sum())})")
    return True