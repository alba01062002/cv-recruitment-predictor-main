#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilidades comunes para el procesamiento y normalización de CVs.
Incluye constantes de configuración, límites y funciones auxiliares.
"""

import re
from datetime import datetime

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================

CONFIG = {
    "tools": {
        # Lista de herramientas relevantes para el cálculo del fit_score
        "relevant_tools": [
            "python", "matlab", "java", "sql", "git",
            "tensorflow", "pytorch", "scikit", "keras",
            "linux", "c++", "docker", "bash"
        ]
    },
    "languages": {
        # Niveles avanzados (para la parte de puntuación)
        "advanced_levels": ["C1", "C2"]
    }
}

# ============================================================
# CONSTANTES DE CONTROL DE DURACIÓN Y LÍMITES
# ============================================================

# Educación
EDUCATION_MIN_DURATION_YEARS = 1.0
EDUCATION_MAX_DURATION_YEARS = 10.0
EDUCATION_DEFAULT_DURATION = 5.0

# Experiencia laboral e internacional
WORK_MAX_DURATION_YEARS = 10.0
INT_MAX_DURATION_YEARS = 5.0

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def calculate_age_at_graduation(date_of_birth: str, education_list: list, filename: str = "") -> float:
    """
    Calcula la edad aproximada al finalizar los estudios universitarios,
    basándose en la fecha de nacimiento y el último año registrado en educación.

    Args:
        date_of_birth (str): Fecha de nacimiento (formato DD/MM/YYYY o YYYY)
        education_list (list): Lista de diccionarios con los campos de educación
        filename (str): (opcional) nombre del fichero de origen

    Returns:
        float: Edad aproximada al graduarse
    """
    if not date_of_birth:
        return 0.0

    # Detectar formato DD/MM/YYYY o YYYY
    try:
        if re.match(r"\d{2}/\d{2}/\d{4}", date_of_birth):
            d, m, y = map(int, date_of_birth.split("/"))
            birth = datetime(y, m, d)
        elif re.match(r"\d{4}", date_of_birth):
            birth = datetime(int(date_of_birth), 1, 1)
        else:
            return 0.0
    except Exception:
        return 0.0

    # Buscar el último año en la educación
    last_year = None
    for edu in education_list or []:
        year_field = str(edu.get("year", ""))
        years_found = re.findall(r"(19|20)\d{2}", year_field)
        if years_found:
            year_ints = [int(y[-4:]) for y in years_found]
            if not last_year:
                last_year = max(year_ints)
            else:
                last_year = max(last_year, max(year_ints))

    # Si no hay año, estimamos con el actual
    if not last_year:
        last_year = datetime.now().year

    age = last_year - birth.year
    return float(max(0.0, min(age, 80.0)))

# ============================================================
# FUNCIÓN AUXILIAR DE NORMALIZACIÓN DE TEXTOS
# ============================================================

def deaccent(text: str) -> str:
    """
    Elimina acentos y normaliza el texto a ASCII plano.
    """
    import unicodedata
    if text is None:
        return ""
    return unicodedata.normalize("NFKD", str(text)).encode("ASCII", "ignore").decode("ASCII")

# ============================================================
# DEBUG / TEST
# ============================================================

if __name__ == "__main__":
    # Pequeña prueba
    sample_edu = [
        {"degree": "Ingeniería de Telecomunicaciones", "year": "2001-2007"},
        {"degree": "Máster en IA", "year": "2010-2011"},
    ]
    print("Edad graduación:", calculate_age_at_graduation("15/07/1983", sample_edu))
    print("Sin acentos:", deaccent("Máster en Inteligencia Artificial"))