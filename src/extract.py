# src/extract.py
import pytesseract
from PIL import Image
import pdf2image
import os
import re
import tempfile
import shutil
import subprocess
from typing import Dict
from docx import Document

# ---------- helpers shell ----------
def _run_cmd(cmd):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        out = p.stdout.decode("utf-8", errors="ignore")
        err = p.stderr.decode("utf-8", errors="ignore")
        return p.returncode, out, err
    except FileNotFoundError:
        return 127, "", "not found"
    except Exception as e:
        return 1, "", str(e)

# ---------- limpieza de blobs base64 (aplicado a todo tipo de fichero) ----------
_BASE64_BLOCK = re.compile(
    r'(?:[A-Za-z0-9+/]{40,}={0,2})(?:\s+[A-Za-z0-9+/]{40,}={0,2}){2,}',  # bloques largos con saltos
    re.MULTILINE,
)
def _strip_base64_blobs(s: str) -> str:
    if not s:
        return s
    # data URIs
    s = re.sub(r'data:[^;]+;base64,[A-Za-z0-9+/=\s]+', ' ', s, flags=re.IGNORECASE)
    # firmas comunes (JPEG/PNG/PDF) + bloques genéricos
    s = re.sub(r'/9j/[A-Za-z0-9+/=\s]{80,}', ' ', s)          # JPEG
    s = re.sub(r'iVBORw0KGgo[A-Za-z0-9+/=\s]{80,}', ' ', s)   # PNG
    s = re.sub(r'JVBERi0x[A-Za-z0-9+/=\s]{80,}', ' ', s)      # PDF
    s = _BASE64_BLOCK.sub(' ', s)
    return s

# ---------- .doc best effort ----------
def _read_doc_best_effort(path_doc: str) -> tuple[str, str]:
    # 1) textutil (macOS)
    rc, out, _ = _run_cmd(["textutil", "-convert", "txt", "-stdout", path_doc])
    if rc == 0 and out.strip():
        return out, "textutil"
    # 2) antiword
    rc, out, _ = _run_cmd(["antiword", "-m", "UTF-8.txt", path_doc])
    if rc == 127:
        rc, out, _ = _run_cmd(["antiword", path_doc])
    if rc == 0 and out.strip():
        return out, "antiword"
    # 3) catdoc
    rc, out, _ = _run_cmd(["catdoc", "-w", path_doc])
    if rc == 0 and out.strip():
        return out, "catdoc"
    # 4) LibreOffice headless
    tmpdir = tempfile.mkdtemp(prefix="doc2txt_")
    try:
        rc, _, _ = _run_cmd(["soffice", "--headless", "--convert-to", "txt:Text", "--outdir", tmpdir, path_doc])
        base = os.path.splitext(os.path.basename(path_doc))[0] + ".txt"
        txt_path = os.path.join(tmpdir, base)
        if rc == 0 and os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if content.strip():
                return content, "soffice"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    # 5) strings
    rc, out, _ = _run_cmd(["strings", "-a", path_doc])
    if rc == 0 and out.strip():
        return out, "strings"
    return "", "none"

# ---------- lector XML robusto ----------
def _is_base64ish(s: str) -> bool:
    if not s or len(s) < 80:
        return False
    valid = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r\t ")
    ok = sum(ch in valid for ch in s)
    return (ok / len(s)) > 0.9

def _clean_xml_piece(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if _is_base64ish(s):
        return ""
    if len(s) > 200 and " " not in s:
        return ""
    return s

def _gather_strings_from_etree(elem):
    import xml.etree.ElementTree as ET  # stdlib
    bad_tags = {"BinaryObject", "Base64Binary", "Picture", "Image", "Photo", "Graphic", "ContentBinaryObject"}
    out = []
    tag_name = elem.tag.split('}')[-1]
    if tag_name in bad_tags:
        return out
    if elem.text:
        t = _clean_xml_piece(elem.text)
        if t:
            out.append(t)
    # solo atributos “seguros” (evita data/src/href/content …)
    skip_attr = {"data", "value", "content", "binary", "bytes", "file", "href", "src", "uri"}
    for k, v in elem.attrib.items():
        if k.lower() in skip_attr:
            continue
        t = _clean_xml_piece(str(v))
        if t:
            out.append(t)
    for child in list(elem):
        out.extend(_gather_strings_from_etree(child))
    if elem.tail:
        t = _clean_xml_piece(elem.tail)
        if t:
            out.append(t)
    return out

def _extract_text_from_xml(file_path: str) -> tuple[str, str | None]:
    import xml.etree.ElementTree as ET  # stdlib
    tree = ET.parse(file_path)
    root = tree.getroot()
    pieces = _gather_strings_from_etree(root)
    text = " ".join(p for p in pieces if p).strip()
    # nombre header (heurística ligera)
    header_name = None
    for c in pieces[:5]:
        words = c.strip().split()
        if 2 <= len(words) <= 5 and all(len(w) > 1 for w in words):
            header_name = c
            break
    return text, header_name

# ---------- extractor principal ----------
def extract_cv_information(file_path: str) -> Dict:
    extracted_data = {
        "personal_information": {
            "full_name": "",
            "date_of_birth": "", "age": "", "gender": "", "email": "", "phone": "", "address": "",
            "LinkedIn": None, "Instagram": None, "Twitter": None, "website": None,
            
        },
        "languages": [], "education": [], "work_experience": [],
        "hard_and_soft_skills": {"hard_skills": [], "soft_skills": []},
        "other_interests": [],
        "volunteering": [],
        "raw_text": "",
        "text_from_header": None,
        "detected_language": "unknown"
    }

    full_raw_text = ""
    file_extension = file_path.lower().split('.')[-1]

    try:
        if file_extension == 'pdf':
            images = pdf2image.convert_from_path(file_path, dpi=300)
            for image in images:
                full_raw_text += pytesseract.image_to_string(
                    image, lang='spa+eng', config='--oem 3 --psm 3'
                ) + "\n"

        elif file_extension == 'docx':
            if Document:
                doc = Document(file_path)
                for para in doc.paragraphs:
                    full_raw_text += para.text + "\n"
            else:
                extracted_data["raw_text"] = f"DOCX extraction failed: 'python-docx' not available for {file_path}."
                return extracted_data

        elif file_extension == 'doc':
            text, method = _read_doc_best_effort(file_path)
            print(f"[.doc extractor] method = {method}")
            if not text:
                print("Error: no se pudo extraer texto útil del .doc. Considera convertir a .docx o .pdf.")
                extracted_data["raw_text"] = ""
                return extracted_data
            full_raw_text = text

        elif file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                full_raw_text = f.read()

        elif file_extension == 'xml':
            try:
                xml_text, header_name_xml = _extract_text_from_xml(file_path)
                if not xml_text:
                    print(f"XML vacío o no legible: {file_path}")
                    extracted_data["raw_text"] = ""
                    return extracted_data
                full_raw_text = xml_text
                if header_name_xml:
                    extracted_data["text_from_header"] = header_name_xml.upper()
                print(f"[XML extractor] OK: {file_path}")
            except Exception as e:
                print(f"Error leyendo XML {file_path}: {e}")
                extracted_data["raw_text"] = ""
                return extracted_data

        else:
            print(f"Unsupported file type: {file_path}. Returning empty data.")
            extracted_data["raw_text"] = f"Unsupported file type: {file_path}"
            return extracted_data

    except Exception as e:
        print(f"Error processing {file_path}: {e}. Returning empty data with raw_text as error message.")
        extracted_data["raw_text"] = str(e)
        return extracted_data

    # --- nombre en cabecera ---
    lines = [line.strip() for line in full_raw_text.split('\n') if line.strip()]
    detected_name = None
    if lines:
        for line in lines[:5]:
            words = line.split()
            if line.isupper() and 2 <= len(words) <= 5 and all(len(w) > 1 for w in words):
                detected_name = line
                break
    if detected_name:
        extracted_data["text_from_header"] = detected_name
        print(f"Detected potential name from header: '{detected_name}'")
    else:
        print("No prominent ALL CAPS name detected in header.")

    # --- NUEVO: quitar blobs base64 ANTES de normalizar ---
    full_raw_text = _strip_base64_blobs(full_raw_text)

    # --- normalización ---
    t = full_raw_text.lower()
    t = re.sub(r'curriculum vitae|currículum vítae|\[page \d+\]', '', t)
    t = re.sub(r'[^a-záéíóúü\s@.\-/:,0-9()\[\]{}#+*&%$!?=<>\'\"]', '', t)
    t = re.sub(r'dhotmail|whotmail|9hotmail|gmai1|gmial', '@hotmail|@hotmail|@hotmail|@gmail|@gmail', t)
    t = re.sub(r'\s+', ' ', t).strip()
    extracted_data["raw_text"] = t

    # --- detección de idioma ---
    es_words = ['educación','formación','experiencia laboral','experiencia profesional','habilidades','competencias',
                'idiomas','lenguas','currículum','estudios','título','empleo','trabajo','certificados','proyectos',
                'perfil','contacto','referencias','objetivo profesional']
    en_words = ['education','training','work experience','professional experience','skills','competencies','languages',
                'resume','cv','studies','degree','employment','job','certifications','projects','profile','contact',
                'references','career objective']
    es_count = sum(len(re.findall(r'\b'+re.escape(k)+r'\b', t)) for k in es_words)
    en_count = sum(len(re.findall(r'\b'+re.escape(k)+r'\b', t)) for k in en_words)
    extracted_data["detected_language"] = 'es' if es_count > en_count and es_count > 0 \
        else ('en' if en_count > es_count and en_count > 0 else 'mixed')

    print(f"Detected language: {extracted_data['detected_language']} for {file_path}")
    print(f"Normalized raw_text extracted from {file_path} (first 500 chars): {extracted_data['raw_text'][:500]}...")
    return extracted_data