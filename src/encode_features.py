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

def _extract_dense_features(cv: Dict) -> List[float]:
    """
    Extrae vector numérico (dense) ordenado:
    0: degree_years
    1: age_at_graduation
    2: total_work_years
    3: has_international_experience (0.0/1.0)
    4: has_master (0.0/1.0)
    5: n_hard_skills
    6: n_languages
    7: max_english_level
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
    
    return [
        degree_years,
        age_at_graduation,
        total_work_years,
        has_intl,
        has_mst,
        float(n_hard),
        float(n_langs),
        float(max_eng)
    ]

DENSE_FEATURE_NAMES = [
    "dense_degree_years",
    "dense_age_at_graduation",
    "dense_total_work_years",
    "dense_has_international_experience",
    "dense_has_master",
    "dense_n_hard_skills",
    "dense_n_languages",
    "dense_max_english_level"
]

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
    # Education
    edu_txts = []
    for e in cv.get("education_normalized", []) or cv.get("education", []):
        edu_txts.append(_join_fields(e, ["degree", "field", "university", "location", "description"])) # Quitamos "year"

    # Work
    work_txts = []
    for w in cv.get("work_experience_normalized", []) or cv.get("work_experience", []):
        work_txts.append(_join_fields(w, ["company", "position", "description", "location"]))

    # International
    intl_txts = []
    for it in cv.get("international_experience_normalized", []) or cv.get("international_experience", []):
        intl_txts.append(_join_fields(it, ["type", "country", "institution_or_company", "description"]))

    # Languages
    lang_txts = []
    for l in cv.get("languages_normalized", []) or cv.get("languages", []):
        lang_txts.extend(_flatten_any(l))

    # Skills
    skills_txt = _extract_skills_text(cv)

    # Others / volunteer
    others = " ".join(_flatten_any(cv.get("other_interests_normalized") or cv.get("other_interests") or []))
    volun  = " ".join(
        _join_fields(v, ["organization","role","description"])
        for v in (cv.get("volunteering_normalized") or cv.get("volunteering") or [])
    )

    chunks = edu_txts + work_txts + intl_txts + lang_txts
    base = " ".join([ " ".join(chunks), skills_txt, others, volun ]).strip()
    
    base = _clean_token_text(base) # Limpieza final global
    if not base:
        base = _clean_token_text(cv.get("raw_text") or "")
    return base if base else "cv"

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
    dense_features_list: List[List[float]] = []
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
        dense_features_list.append(_extract_dense_features(cv))

    n_docs = len(filenames)
    if n_docs == 0:
        logger.error(f"No valid CV data loaded from {normalized_dir}.")
        return False
    logger.info(f"Loaded {n_docs} CVs (skipped={skipped_by_filter}).")

    # 3) Vectorizar Texto (TF-IDF)
    feature_names_global: List[str] = []
    
    # Config mejorada: min_df=2 (limpieza ruido), ngram_range=(1,2)
    tfidf_args = dict(min_df=2, ngram_range=(1, 2), stop_words='english') 

    if vectorization_mode == "combined":
        vec = TfidfVectorizer(**tfidf_args)
        try:
            X_text = vec.fit_transform(texts_combined)
        except ValueError:
            # Fallback si vocab vacío
            vec = TfidfVectorizer(min_df=1)
            X_text = vec.fit_transform(texts_combined)
        
        joblib.dump(vec, os.path.join(models_dir, "tfidf_combined.joblib"))
        try:
            feature_names_global = list(vec.get_feature_names_out())
        except:
            feature_names_global = [f"text_{i}" for i in range(X_text.shape[1])]
    else:
        mats = []
        vocab_sizes = []
        section_order = []
        for s, docs in texts_sections.items():
            if not any(t.strip() for t in docs):
                docs = ["cv"] * len(docs)
            vec = TfidfVectorizer(**tfidf_args)
            try:
                Xs = vec.fit_transform(docs)
            except ValueError:
                vec = TfidfVectorizer(min_df=1)
                Xs = vec.fit_transform(docs)
                
            mats.append(Xs)
            vocab_sizes.append(Xs.shape[1])
            section_order.append(s)
            joblib.dump(vec, os.path.join(models_dir, f"tfidf_{s}.joblib"))
            try:
                feats = [f"{s}::{t}" for t in vec.get_feature_names_out()]
                feature_names_global.extend(feats)
            except:
                pass
        X_text = hstack(mats).tocsr()

    # 4) Procesar Dense Features
    X_dense = np.array(dense_features_list, dtype=float)
    
    # Escalar Dense Features (StandardScaler)
    scaler = StandardScaler()
    X_dense_scaled = scaler.fit_transform(X_dense)
    joblib.dump(scaler, os.path.join(models_dir, "dense_scaler.joblib"))
    
    # Nombre features dense
    feature_names_global.extend(DENSE_FEATURE_NAMES)
    
    # 5) Concatenar Híbrido
    X_final = hstack([X_text, csr_matrix(X_dense_scaled)]).tocsr()

    # 6) Etiquetas
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

    logger.info(f"Saved Hybrid features (Text+Dense) to {features_dir} "
                f"(X: {X_final.shape}, positives: {int(y.sum())})")
    
    # Check rápido
    logger.info(f"  Dense stats sample (first row): {dense_features_list[0]}")
    return True