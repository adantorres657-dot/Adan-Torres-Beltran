# -*- coding: utf-8 -*-
"""
Sistema LPR - Categoria B: Reconocimiento de Matriculas Vehiculares (Mexico)
Convocatoria: 32 estados, 5000+ matriculas, tiempo real, encriptacion, metricas.

Arquitectura de 3 hilos:
  Hilo captura  ->  frame_queue  ->  Hilo procesamiento (YOLO+OCR)
                                              |
                                       result_queue
                                              |
                                    Hilo UI (Tkinter, solo renderiza)
"""

# ============================================================
# FIX CRÍTICO: Configurar OpenCV ANTES de importar cv2
# Esto fuerza el uso de MSMF en lugar de FFmpeg (más estable)
# ============================================================
import os
import sys

# Forzar uso de MSMF en lugar de FFmpeg (más estable en Windows)
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "1"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

# Suprimir warnings de FFmpeg
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# AHORA SÍ importar cv2 y el resto
# ============================================================
import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import time
import sqlite3
import threading
import queue
import hashlib
import re
import json
import math
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter, deque
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, ttk
from PIL import Image, ImageTk

# GPU: detectar CUDA disponible
try:
    import torch
    import torchvision  # noqa: F401 — requerido por CharRecognizer
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_DEVICE    = "cuda:0" if GPU_AVAILABLE else "cpu"
except ImportError:
    GPU_AVAILABLE = False
    GPU_DEVICE    = "cpu"

# TrOCR (Transformer OCR — Microsoft) — carga lazy, opcional
# pip install transformers
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False

# PaddleOCR — motor de alta precisión para texto en imágenes
# pip install paddlepaddle paddleocr
try:
    from paddleocr import PaddleOCR as _PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

# OCR Learning System — Sistema de aprendizaje de correcciones manuales
try:
    from ocr_learning import get_ocr_learning
    OCR_LEARNING_AVAILABLE = True
except ImportError:
    OCR_LEARNING_AVAILABLE = False
    print("AVISO: 'ocr_learning.py' no encontrado. Sistema de aprendizaje desactivado.")

# Encriptacion de imagenes (Fernet = AES-128-CBC + HMAC)
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("AVISO: 'cryptography' no instalada. Imagenes no encriptadas.")
    print("       Instalar con: pip install cryptography")

# Control difuso de brillo — ELIMINADO (causaba titileo)

# ============================================================
# CONTROL DIFUSO DE BRILLO — ELIMINADO
# Causa: flickering/titileo en video en vivo
# El ajuste de brillo frame-a-frame generaba cambios visuales bruscos
# ============================================================


# ============================================================
# CONSTANTES GLOBALES
# ============================================================
DB_NAME      = "lpr_system.db"
IMAGE_FOLDER = "event_images"
CONFIG_FILE  = "lpr_config.json"
KEY_FILE     = "lpr_secret.key"

os.makedirs(IMAGE_FOLDER, exist_ok=True)

VEHICLE_CLASSES = [2, 3, 5, 7]   # COCO: car, motorcycle, bus, truck

# ──────────────────────────────────────────────────────────────────────────────
# TABLAS DE CORRECCIÓN OCR — confusiones visuales documentadas
# Contexto: placas mexicanas usan MAYÚSCULAS exclusivamente.
# Todas las entradas se aplican DESPUÉS de .upper() al texto raw del OCR.
#
# Aplicación NO posicional (corrección general antes de conocer la máscara):
#   _OCR_DIGIT  → cuando se espera un DÍGITO y el OCR devuelve una letra similar
#   _OCR_LETTER → cuando se espera una LETRA y el OCR devuelve un número similar
#
# Confusiones cubiertas (referencia completa):
#   #1  0 ↔ O   #3  0 ↔ Q   #4  1 ↔ I   #7  2 ↔ Z   #9  3 ↔ E
#   #11 4 ↔ A   #13 5 ↔ S   #15 6 ↔ G   #17 7 ↔ T   #19 8 ↔ B
#   #22 9 ↔ g   #23 4 ↔ 9   #24 G ↔ C
# ──────────────────────────────────────────────────────────────────────────────
_OCR_DIGIT = {
    # Letra OCR → dígito correcto
    "O": "0",   # #1  O → 0
    "Q": "0",   # #3  Q → 0  (Q sin cola)
    "D": "0",   # extra: D puede parecer 0
    "I": "1",   # #4  I → 1
    "L": "1",   # extra: L minúscula (ya uppercase) → 1
    "Z": "2",   # #7  Z → 2
    "E": "3",   # #9  E → 3
    "A": "4",   # #11 A → 4
    "S": "5",   # #13 S → 5
    "G": "6",   # #15 G → 6
    "C": "6",   # #24 C → 6  (C y 6 visualmente similares)
    "T": "7",   # #17 T → 7
    "Y": "7",   # extra: Y puede confundirse con 7
    "B": "8",   # #19 B → 8
    "U": "0",   # extra: U puede confundirse con 0 en fuentes degradadas
}

_OCR_LETTER = {
    # Dígito OCR → letra correcta
    "0": "O",   # #1  0 → O
    "1": "I",   # #4  1 → I
    "2": "Z",   # #7  2 → Z
    "3": "E",   # #9  3 → E
    "4": "A",   # #11 4 → A  (NOTA: NO 4→9, eso es error de segmentación)
    "5": "S",   # #13 5 → S
    "6": "G",   # #15 6 → G
    "6": "C",   # #24 6 → C  (en contexto de letra)
    "7": "T",   # #17 7 → T
    "8": "B",   # #19 8 → B
    "9": "G",   # #21 9 → G  (9 y g de un piso son visualmente similares)
}

# ============================================================
# FORMATOS DE MATRICULAS MEXICANAS (32 estados + federal)
# Fuente: SICT / SCT Mexico
# Formato federal actual: 3 letras + 3 digitos + 1 letra  (ABC123D)
# Formatos estatales historicos y actuales incluidos.
# ============================================================
MX_PLATE_PATTERNS = [
    # Federal actual (2008-presente): ABC-123-D
    re.compile(r"^[A-Z]{3}\d{3}[A-Z]$"),
    # Federal anterior (1994-2008): ABC-1234
    re.compile(r"^[A-Z]{3}\d{4}$"),
    # Algunos estados: 2 letras + 4 digitos
    re.compile(r"^[A-Z]{2}\d{4}$"),
    # Algunos estados: 2 letras + 3 digitos + 1 letra
    re.compile(r"^[A-Z]{2}\d{3}[A-Z]$"),
    # Placas de servicio publico / especiales: numeros + letras
    re.compile(r"^\d{3}[A-Z]{3}$"),
    re.compile(r"^\d{4}[A-Z]{2}$"),
]

# Prefijos de estado por las primeras 2-3 letras de la placa federal
# (Basado en asignacion de series por estado de la SICT)
MX_STATE_PREFIXES: dict[str, str] = {
    "A":  "Aguascalientes",
    "B":  "Baja California",
    "BC": "Baja California",
    "BS": "Baja California Sur",
    "C":  "Campeche",
    "CH": "Chihuahua",
    "CL": "Colima",
    "CO": "Coahuila",
    "CS": "Chiapas",
    "D":  "Ciudad de Mexico",
    "DF": "Ciudad de Mexico",
    "DG": "Durango",
    "E":  "Estado de Mexico",
    "EM": "Estado de Mexico",
    "F":  "Coahuila",
    "G":  "Guanajuato",
    "GR": "Guerrero",
    "GT": "Guanajuato",
    "H":  "Hidalgo",
    "HG": "Hidalgo",
    "J":  "Jalisco",
    "JA": "Jalisco",
    "K":  "Nuevo Leon",
    "L":  "Baja California Sur",
    "M":  "Michoacan",
    "MI": "Michoacan",
    "MO": "Morelos",
    "N":  "Nayarit",
    "NA": "Nayarit",
    "NL": "Nuevo Leon",
    "O":  "Oaxaca",
    "OA": "Oaxaca",
    "P":  "Puebla",
    "PB": "Puebla",
    "Q":  "Queretaro",
    "QR": "Quintana Roo",
    "QT": "Queretaro",
    "R":  "Quintana Roo",
    "S":  "Sinaloa",
    "SI": "Sinaloa",
    "SL": "San Luis Potosi",
    "SO": "Sonora",
    "SP": "San Luis Potosi",
    "T":  "Tabasco",
    "TA": "Tamaulipas",
    "TB": "Tabasco",
    "TL": "Tlaxcala",
    "TM": "Tamaulipas",
    "TX": "Tlaxcala",
    "U":  "Sonora",
    "V":  "Veracruz",
    "VE": "Veracruz",
    "W":  "Chihuahua",
    "X":  "Guerrero",
    "Y":  "Yucatan",
    "YU": "Yucatan",
    "Z":  "Zacatecas",
    "ZA": "Zacatecas",
}

MX_STATES_ALL = [
    "Aguascalientes","Baja California","Baja California Sur","Campeche",
    "Chiapas","Chihuahua","Ciudad de Mexico","Coahuila","Colima","Durango",
    "Estado de Mexico","Guanajuato","Guerrero","Hidalgo","Jalisco",
    "Michoacan","Morelos","Nayarit","Nuevo Leon","Oaxaca","Puebla",
    "Queretaro","Quintana Roo","San Luis Potosi","Sinaloa","Sonora",
    "Tabasco","Tamaulipas","Tlaxcala","Veracruz","Yucatan","Zacatecas",
    "Desconocido",
]


def identify_mx_state(plate: str) -> str:
    """
    Identifica el estado mexicano a partir del prefijo de la matricula.
    Retorna "Desconocido" si el prefijo no está registrado o la placa es inválida.
    
    Formatos válidos:
    - Prefijo de 1-3 letras + dígitos: A123D, AB1234, ABC123D (federal)
    - El prefijo se identifica por las primeras 1-3 letras registradas en SICT
    """
    if not plate or len(plate) < 4:
        return "Desconocido"
    
    plate = plate.upper().replace("-", "").replace(" ", "")
    
    # Intentar con 3, 2, y 1 letra (en ese orden de prioridad)
    for length in (3, 2, 1):
        if len(plate) <= length:
            continue
        
        prefix = plate[:length]
        if prefix in MX_STATE_PREFIXES:
            rest = plate[length:]
            
            if not rest:
                continue
            
            # El resto debe empezar con un dígito
            if rest[0].isdigit() and any(c.isdigit() for c in rest):
                return MX_STATE_PREFIXES[prefix]
    
    # Caso especial: placas federales de 3 letras (ABC123D)
    # El prefijo de estado es solo la primera letra
    # Verificar que tenga formato federal: 3 letras + 3-4 dígitos + opcional 1 letra
    # PERO: las 3 letras NO deben formar un prefijo válido de 2 o 3 letras
    if len(plate) >= 6:
        # Patrón: LLL + NNN + opcional L
        if (plate[:3].isalpha() and 
            plate[3].isdigit() and 
            any(c.isdigit() for c in plate[3:7])):
            
            # Verificar que las primeras 2 o 3 letras NO sean un prefijo válido
            # (si lo son, ya se habría detectado arriba)
            first_two = plate[:2]
            first_three = plate[:3]
            
            if first_two not in MX_STATE_PREFIXES and first_three not in MX_STATE_PREFIXES:
                # Usar la primera letra como prefijo
                first_letter = plate[0]
                if first_letter in MX_STATE_PREFIXES:
                    return MX_STATE_PREFIXES[first_letter]
    
    return "Desconocido"


def validate_mx_plate(plate: str) -> tuple[bool, str]:
    """
    Valida si el texto OCR corresponde a un formato de matricula mexicana.
    Aplica corrección posicional antes de validar para tolerar errores OCR menores.
    Retorna (es_valida, razon_si_invalida).
    """
    clean = plate.upper().replace("-", "").replace(" ", "")
    if len(clean) < 4 or len(clean) > 9:
        return False, f"Longitud invalida: {len(clean)} caracteres"
    # Intentar corregir antes de validar
    corrected = correct_mx_plate(clean)
    for pattern in MX_PLATE_PATTERNS:
        if pattern.match(corrected):
            return True, ""
    # También probar el texto original sin corrección
    for pattern in MX_PLATE_PATTERNS:
        if pattern.match(clean):
            return True, ""
    return False, f"No coincide con ningun formato MX: {corrected}"


# ============================================================
# ENCRIPTACION DE IMAGENES (Fernet / AES)
# ============================================================

def _load_or_create_key() -> bytes | None:
    """Carga o genera la clave de encriptacion Fernet."""
    if not CRYPTO_AVAILABLE:
        return None
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read()
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    print(f"Clave de encriptacion generada: {KEY_FILE}")
    return key


_FERNET_KEY = _load_or_create_key()
_fernet = Fernet(_FERNET_KEY) if (_FERNET_KEY and CRYPTO_AVAILABLE) else None


def save_image_encrypted(img_bgr: np.ndarray, filepath: str) -> str:
    """
    Guarda imagen encriptada con Fernet (AES-128-CBC + HMAC-SHA256).
    Si la encriptacion no esta disponible, guarda en claro con advertencia.
    Retorna la ruta guardada.
    """
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    raw = buf.tobytes()
    enc_path = filepath + ".enc"
    if _fernet:
        encrypted = _fernet.encrypt(raw)
        with open(enc_path, "wb") as f:
            f.write(encrypted)
        return enc_path
    else:
        # Fallback: guardar sin encriptar
        with open(filepath, "wb") as f:
            f.write(raw)
        return filepath


def load_image_decrypted(filepath: str) -> np.ndarray | None:
    """Carga y desencripta una imagen guardada con save_image_encrypted."""
    try:
        if filepath.endswith(".enc") and _fernet:
            with open(filepath, "rb") as f:
                raw = _fernet.decrypt(f.read())
        else:
            with open(filepath, "rb") as f:
                raw = f.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error desencriptando imagen: {e}")
        return None


def image_hash(img_bgr: np.ndarray) -> str:
    """SHA-256 del contenido JPEG de la imagen (para auditoria sin guardar crudo)."""
    _, buf = cv2.imencode(".jpg", img_bgr)
    return hashlib.sha256(buf.tobytes()).hexdigest()


# ============================================================
# CONFIGURACION
# ============================================================

def load_config() -> dict:
    default = {
        "show_rectangles": True,
        "show_text": True,
        "show_fps": True,
        "resolution": "media",          # alta=1080p es demasiado para GTX 1650 en tiempo real
        "auto_save_images": True,
        "process_every_n_frames": 5,    # 5 frames = ~6fps de procesamiento a 30fps captura
        "duplicate_timeout": 10,
        "ocr_lang": "en",               # 'en' es mejor que 'es' para placas alfanuméricas
        "detection_mode": "full",
        "conf_threshold_vehicle": 0.45,
        "conf_threshold_plate": 0.40,
        "camera_index": 0,
        "validate_mx_format": True,
        "use_heavy_ocr": False,         # TrOCR/PaddleOCR: desactivar por defecto (consumen VRAM)
        "brightness_control": False,    # FIX: Desactivar control de brillo por defecto (causa parpadeo)
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                default.update(json.load(f))
        except Exception:
            pass
    return default


def save_config(cfg: dict):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error guardando config: {e}")


config = load_config()

# ============================================================
# BASE DE DATOS  (indices para busqueda <1s con 50,000 registros)
# ============================================================

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("PRAGMA journal_mode=WAL")   # escrituras concurrentes
        conn.execute("PRAGMA synchronous=NORMAL") # balance velocidad/seguridad
        c = conn.cursor()

        # Tabla de detecciones
        c.execute("""CREATE TABLE IF NOT EXISTS detections (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            plate         TEXT    NOT NULL,
            timestamp     DATETIME NOT NULL,
            image_path    TEXT,
            image_hash    TEXT,
            traffic_count INTEGER,
            is_registered BOOLEAN,
            state         TEXT,
            confidence    REAL
        )""")

        # Tabla de matriculas registradas
        c.execute("""CREATE TABLE IF NOT EXISTS registered_plates (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            plate             TEXT    NOT NULL UNIQUE,
            registration_date DATETIME NOT NULL,
            source            TEXT,
            status            TEXT DEFAULT 'active',
            state             TEXT,
            owner_ref         TEXT
        )""")

        # Tabla de registros invalidos (requerimiento convocatoria)
        c.execute("""CREATE TABLE IF NOT EXISTS invalid_registrations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_raw  TEXT    NOT NULL,
            reason     TEXT,
            timestamp  DATETIME NOT NULL,
            image_hash TEXT,
            image_path TEXT
        )""")

        # Tabla de metricas de sesion
        c.execute("""CREATE TABLE IF NOT EXISTS session_metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_start   DATETIME,
            session_end     DATETIME,
            total_detections INTEGER DEFAULT 0,
            true_positives   INTEGER DEFAULT 0,
            false_positives  INTEGER DEFAULT 0,
            avg_response_ms  REAL DEFAULT 0
        )""")

        # Migraciones PRIMERO: agregar columnas nuevas si no existen
        # (necesario antes de crear indices sobre esas columnas)
        existing = {r[1] for r in c.execute("PRAGMA table_info(detections)")}
        for col, typedef in [("image_hash","TEXT"), ("state","TEXT"), ("confidence","REAL")]:
            if col not in existing:
                c.execute(f"ALTER TABLE detections ADD COLUMN {col} {typedef}")

        existing_reg = {r[1] for r in c.execute("PRAGMA table_info(registered_plates)")}
        for col, typedef in [("state","TEXT"), ("owner_ref","TEXT")]:
            if col not in existing_reg:
                c.execute(f"ALTER TABLE registered_plates ADD COLUMN {col} {typedef}")

        # INDICES para busqueda rapida (<1ms con 50,000 registros)
        c.execute("CREATE INDEX IF NOT EXISTS idx_det_plate  ON detections(plate)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_det_ts     ON detections(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_reg_plate  ON registered_plates(plate)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_reg_status ON registered_plates(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_reg_state  ON registered_plates(state)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_inv_plate  ON invalid_registrations(plate_raw)")

        conn.commit()
    print("Base de datos inicializada con indices.")


def save_to_db(plate: str, image_path: str, img_hash: str,
               traffic_count: int, is_registered: bool,
               state: str, confidence: float):
    # Validación de entrada
    if not plate or not isinstance(plate, str):
        print("Error BD: placa inválida")
        return
    if not isinstance(traffic_count, int) or traffic_count < 0:
        print("Error BD: traffic_count inválido")
        return
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        print("Error BD: confidence inválido")
        return
    if not isinstance(is_registered, bool):
        print("Error BD: is_registered debe ser bool")
        return
    
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute(
                """INSERT INTO detections
                   (plate,timestamp,image_path,image_hash,traffic_count,
                    is_registered,state,confidence)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (plate, datetime.now(), image_path, img_hash,
                 traffic_count, is_registered, state, confidence),
            )
            conn.commit()
    except Exception as e:
        print(f"Error BD save_to_db: {e}")


def save_invalid_registration(plate_raw: str, reason: str,
                               img_hash: str = "", image_path: str = ""):
    """Guarda un registro invalido en tabla separada y notifica al sistema."""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute(
                """INSERT INTO invalid_registrations
                   (plate_raw,reason,timestamp,image_hash,image_path)
                   VALUES (?,?,?,?,?)""",
                (plate_raw, reason, datetime.now(), img_hash, image_path),
            )
            conn.commit()
        print(f"[INVALIDO] {plate_raw}: {reason}")
        # Notificar a la UI si esta disponible
        if lpr_app_instance:
            lpr_app_instance.notify_invalid(plate_raw, reason)
    except Exception as e:
        print(f"Error guardando invalido: {e}")


# ============================================================
# EXPORTACION
# ============================================================

def export_to_excel():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            df = pd.read_sql_query(
                "SELECT id,plate,timestamp,image_hash,traffic_count,"
                "is_registered,state,confidence FROM detections", conn
            )
        if df.empty:
            messagebox.showinfo("Exportar", "No hay datos.")
            return
        fn = f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(fn, index=False, engine="openpyxl")
        messagebox.showinfo("Exportar", f"Exportado: {fn}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def export_registered_plates():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            df = pd.read_sql_query(
                "SELECT id,plate,registration_date,source,status,state "
                "FROM registered_plates", conn
            )
        if df.empty:
            messagebox.showinfo("Exportar", "No hay registradas.")
            return
        fn = f"registered_plates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(fn, index=False, engine="openpyxl")
        messagebox.showinfo("Exportar", f"Exportadas {len(df)} matriculas.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def export_invalid_registrations():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            df = pd.read_sql_query(
                "SELECT id,plate_raw,reason,timestamp,image_hash "
                "FROM invalid_registrations ORDER BY timestamp DESC", conn
            )
        if df.empty:
            messagebox.showinfo("Exportar", "No hay registros invalidos.")
            return
        fn = f"invalid_plates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(fn, index=False, engine="openpyxl")
        messagebox.showinfo("Exportar", f"Exportados {len(df)} registros invalidos.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def import_registered_plates_from_csv():
    fp = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
    if not fp:
        return
    try:
        df = pd.read_csv(fp)
        if "plate" not in df.columns:
            messagebox.showerror("Error", "Columna 'plate' requerida.")
            return
        ins = skp = inv = 0
        with sqlite3.connect(DB_NAME) as conn:
            for _, row in df.iterrows():
                p = str(row["plate"]).strip().upper()
                if not p:
                    continue
                valid, reason = validate_mx_plate(p)
                if not valid:
                    save_invalid_registration(p, f"CSV import: {reason}")
                    inv += 1
                    continue
                state = identify_mx_state(p)
                try:
                    conn.execute(
                        "INSERT INTO registered_plates "
                        "(plate,registration_date,source,status,state) VALUES (?,?,?,?,?)",
                        (p, datetime.now(), "csv_import", "active", state),
                    )
                    ins += 1
                except sqlite3.IntegrityError:
                    skp += 1
            conn.commit()
        load_registered_plates()
        messagebox.showinfo("Importar",
            f"Importadas: {ins}  |  Omitidas (duplicado): {skp}  |  Invalidas: {inv}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def clear_old_detections():
    dias = simpledialog.askinteger("Limpiar",
        "Borrar detecciones con mas de X dias:", minvalue=1, maxvalue=365)
    if not dias:
        return
    limite = datetime.now() - timedelta(days=dias)
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cur = conn.execute(
                "DELETE FROM detections WHERE timestamp < ?", (limite,))
            conn.commit()
        messagebox.showinfo("Limpiar", f"Se borraron {cur.rowcount} detecciones.")
        if lpr_app_instance:
            lpr_app_instance.refresh_detections_list()
    except Exception as e:
        messagebox.showerror("Error", str(e))


# ============================================================
# VENTANAS DE GESTION
# ============================================================

def view_registered_plates():
    win = tk.Toplevel()
    win.title("Matriculas Registradas")
    win.geometry("780x460")

    # Filtros
    ff = tk.Frame(win); ff.pack(fill="x", padx=8, pady=4)
    tk.Label(ff, text="Filtrar estado:").pack(side="left")
    state_var = tk.StringVar(value="Todos")
    state_cb  = ttk.Combobox(ff, textvariable=state_var,
                              values=["Todos"] + MX_STATES_ALL, width=22)
    state_cb.pack(side="left", padx=4)
    tk.Label(ff, text="Buscar placa:").pack(side="left", padx=(12,0))
    search_var = tk.StringVar()
    tk.Entry(ff, textvariable=search_var, width=14).pack(side="left", padx=4)

    frm = tk.Frame(win); frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
    cols = ("ID","Placa","Estado","Fecha","Fuente","Status")
    tree = ttk.Treeview(frm, columns=cols, show="headings")
    widths = (40, 110, 160, 160, 90, 70)
    for col, w in zip(cols, widths):
        tree.heading(col, text=col); tree.column(col, width=w)
    sb = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    count_lbl = tk.Label(win, text="")
    count_lbl.pack()

    def reload(*_):
        tree.delete(*tree.get_children())
        q = "SELECT id,plate,state,registration_date,source,status FROM registered_plates WHERE 1=1"
        params = []
        if state_var.get() != "Todos":
            q += " AND state=?"; params.append(state_var.get())
        if search_var.get().strip():
            q += " AND plate LIKE ?"; params.append(f"%{search_var.get().strip().upper()}%")
        q += " ORDER BY registration_date DESC LIMIT 500"
        try:
            with sqlite3.connect(DB_NAME) as conn:
                rows = conn.execute(q, params).fetchall()
            for r in rows:
                tree.insert("", tk.END, values=r)
            count_lbl.config(text=f"{len(rows)} registros mostrados")
        except Exception as e:
            print(f"Error: {e}")

    state_cb.bind("<<ComboboxSelected>>", reload)
    search_var.trace_add("write", reload)

    def eliminar():
        sel = tree.selection()
        if not sel: return
        item = tree.item(sel[0])
        pid, placa = item["values"][0], item["values"][1]
        if messagebox.askyesno("Eliminar", f"Eliminar {placa}?"):
            try:
                with sqlite3.connect(DB_NAME) as conn:
                    conn.execute("DELETE FROM registered_plates WHERE id=?", (pid,))
                    conn.commit()
                reload(); load_registered_plates()
                if lpr_app_instance: lpr_app_instance.refresh_detections_list()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def toggle_estado():
        sel = tree.selection()
        if not sel: return
        item = tree.item(sel[0])
        pid = item["values"][0]
        nuevo = "inactive" if item["values"][5] == "active" else "active"
        try:
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("UPDATE registered_plates SET status=? WHERE id=?", (nuevo, pid))
                conn.commit()
            reload(); load_registered_plates()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    bf = tk.Frame(win); bf.pack(pady=5)
    tk.Button(bf, text="Eliminar",          command=eliminar).pack(side=tk.LEFT, padx=5)
    tk.Button(bf, text="Activar/Desactivar",command=toggle_estado).pack(side=tk.LEFT, padx=5)
    tk.Button(bf, text="Exportar Excel",    command=export_registered_plates).pack(side=tk.LEFT, padx=5)
    tk.Button(bf, text="Cerrar",            command=win.destroy).pack(side=tk.LEFT, padx=5)
    reload()


def view_invalid_registrations():
    win = tk.Toplevel()
    win.title("Registros Invalidos")
    win.geometry("700x400")
    frm = tk.Frame(win); frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
    cols = ("ID","Placa","Razon","Fecha","Hash")
    tree = ttk.Treeview(frm, columns=cols, show="headings")
    widths = (40, 110, 260, 160, 80)
    for col, w in zip(cols, widths):
        tree.heading(col, text=col); tree.column(col, width=w)
    sb = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    try:
        with sqlite3.connect(DB_NAME) as conn:
            rows = conn.execute(
                "SELECT id,plate_raw,reason,timestamp,image_hash "
                "FROM invalid_registrations ORDER BY timestamp DESC LIMIT 200"
            ).fetchall()
        for r in rows:
            tree.insert("", tk.END, values=r)
    except Exception as e:
        print(f"Error: {e}")
    bf = tk.Frame(win); bf.pack(pady=5)
    tk.Button(bf, text="Exportar Excel", command=export_invalid_registrations).pack(side=tk.LEFT, padx=5)
    tk.Button(bf, text="Cerrar", command=win.destroy).pack(side=tk.LEFT, padx=5)


# ============================================================
# CONTADOR, DUPLICADOS Y CACHE DE MATRICULAS
# ============================================================
traffic_counter = 0
last_detected: dict = {}
DUPLICATE_TIMEOUT = config["duplicate_timeout"]
_dup_lock     = threading.Lock()
_plates_lock  = threading.Lock()
_traffic_lock = threading.Lock()          # FIX: proteger traffic_counter de race conditions
registered_plates_set: set = set()


def should_register(plate: str) -> bool:
    now = time.time()
    with _dup_lock:
        if plate not in last_detected or (now - last_detected[plate]) > DUPLICATE_TIMEOUT:
            last_detected[plate] = now
            return True
    return False


def set_duplicate_timeout(value: int):
    global DUPLICATE_TIMEOUT
    DUPLICATE_TIMEOUT = value
    config["duplicate_timeout"] = value
    save_config(config)


def load_registered_plates():
    global registered_plates_set
    try:
        with sqlite3.connect(DB_NAME) as conn:
            rows = conn.execute(
                "SELECT plate FROM registered_plates WHERE status='active'"
            ).fetchall()
        with _plates_lock:
            registered_plates_set = {r[0] for r in rows}
        print(f"Cache: {len(registered_plates_set)} matriculas activas.")
    except Exception as e:
        print(f"Error cargando matriculas: {e}")


def is_plate_registered(plate: str) -> bool:
    with _plates_lock:
        return plate in registered_plates_set


def register_new_plate(plate_text: str, source: str = "manual",
                       image_path: str = "") -> bool:
    """Registra una matricula validando formato mexicano primero."""
    plate_text = plate_text.upper().replace("-","").replace(" ","")
    valid, reason = validate_mx_plate(plate_text)
    if not valid:
        save_invalid_registration(plate_text, f"Registro manual invalido: {reason}")
        messagebox.showwarning("Formato invalido",
            f"La matricula '{plate_text}' no cumple formato MX.\n{reason}")
        return False
    state = identify_mx_state(plate_text)
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute(
                "INSERT INTO registered_plates "
                "(plate,registration_date,source,status,state) VALUES (?,?,?,?,?)",
                (plate_text, datetime.now(), source, "active", state),
            )
            conn.commit()
        with _plates_lock:
            registered_plates_set.add(plate_text)
        print(f"Registrada: {plate_text} ({state})")
        return True
    except sqlite3.IntegrityError:
        messagebox.showinfo("Duplicado", f"'{plate_text}' ya esta registrada.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


# ============================================================
# PREPROCESAMIENTO Y OCR ESPECIALIZADO PARA PLACAS MEXICANAS
# CNN propia (char_cnn.pth) + EasyOCR como respaldo
# ============================================================

# ── Formatos MX y corrección por posición ───────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# CORRECCIÓN POSICIONAL — usada en _apply_mask_correction con máscara L/N
# Se aplica carácter por carácter según si la posición debe ser Letra o Número.
#
# _TO_LETTER: cuando la máscara dice "L" → convertir dígito a letra visual
# _TO_DIGIT : cuando la máscara dice "N" → convertir letra a dígito visual
#
# Confusiones cubiertas (referencia completa):
#   #1  0 ↔ O   #3  0 ↔ Q   #4  1 ↔ I   #7  2 ↔ Z   #9  3 ↔ E
#   #11 4 ↔ A   #13 5 ↔ S   #15 6 ↔ G   #17 7 ↔ T   #19 8 ↔ B
#   #21 9 ↔ G   (9 visual similar a g de un piso, mapeado a G mayúscula)
# ──────────────────────────────────────────────────────────────────────────────
_TO_LETTER = {
    "0": "O",   # #1  0 → O
    "1": "I",   # #4  1 → I
    "2": "Z",   # #7  2 → Z
    "3": "E",   # #9  3 → E  ← NUEVO
    "4": "A",   # #11 4 → A
    "5": "S",   # #13 5 → S
    "6": "G",   # #15 6 → G
    "7": "T",   # #17 7 → T  ← NUEVO
    "8": "B",   # #19 8 → B
    "9": "G",   # #21 9 → G  (9 ≈ g de un piso)  ← NUEVO
}

_TO_DIGIT = {
    "O": "0",   # #1  O → 0
    "Q": "0",   # #3  Q → 0  (Q sin cola)
    "D": "0",   # extra: D puede parecer 0 en fuentes degradadas
    "U": "0",   # extra: U puede confundirse con 0
    "I": "1",   # #4  I → 1
    "L": "1",   # extra: L (ya uppercase) → 1
    "Z": "2",   # #7  Z → 2
    "E": "3",   # #9  E → 3  ← NUEVO
    "A": "4",   # #11 A → 4
    "S": "5",   # #13 S → 5
    "G": "6",   # #15 G → 6
    "C": "6",   # #24 C → 6  (C y 6 visualmente similares)
    "T": "7",   # #17 T → 7
    "Y": "7",   # extra: Y puede confundirse con 7
    "B": "8",   # #19 B → 8
}

_MX_MASKS = [
    "LLLNNNL",  # Federal 2008+  CVL657B
    "LLLNNNN",  # Federal 1994   CVL6571
    "LLNNNN",   # Estatal 2L4N
    "LLNNNL",   # Estatal 2L3N1L
    "NNNLLL",   # Servicio público
    "NNNNLL",
    "LLLNNN",
    "LLNNN",
]

# Patrones de guión falso: EasyOCR lee el guión '-' como 'I' o '1'
# Ej: CVL-657-B → OCR lee CVLI657IB o CVL1657B
# Detectamos y eliminamos esos separadores falsos SOLO cuando están entre grupos
_DASH_PATTERNS = [
    # LLL-NNN-L leído como LLLINNNI o LLLINNNL (I/1 entre letras y números)
    (re.compile(r'^([A-Z]{2,3})[I1](\d{3,4})[I1]?([A-Z]?)$'), r'\1\2\3'),
    # LLL-NNNN leído como LLLINNN o LLLINNNN (I/1 entre letras y números)
    (re.compile(r'^([A-Z]{2,3})[I1](\d{3,4})$'),               r'\1\2'),
    # LL-NNN-L leído como LLINNNI o LLINNNL
    (re.compile(r'^([A-Z]{2})[I1](\d{3})[I1]?([A-Z]?)$'),     r'\1\2\3'),
    # NNN-LLL leído como NNNILLL o NNN1LLL (I/1 entre números y letras)
    (re.compile(r'^(\d{3,4})[I1]([A-Z]{2,3})$'),              r'\1\2'),
    # Guión al inicio SOLO si es claramente un error: I seguido de consonante + vocal
    # Ej: ICVL657B → CVL657B (I antes de CV es guión)
    # PERO NO: IABC1234 (I podría ser parte de la placa)
    # Patrón: I + consonante + vocal + resto válido
    (re.compile(r'^[I1]([BCDFGHJKLMNPQRSTVWXYZ][AEIOU][A-Z]?\d{3,4}[A-Z]?)$'), r'\1'),
]


def _remove_false_dashes(text: str) -> str:
    """
    Elimina 'I' y '1' que son guiones mal leídos por OCR.
    Solo aplica cuando el patrón coincide con formato MX conocido.
    
    IMPORTANTE: Solo elimina I/1 cuando están ENTRE grupos (letras→números o números→letras),
    NO cuando son parte del contenido de la placa.
    """
    original = text
    for pattern, replacement in _DASH_PATTERNS:
        m = pattern.match(text)
        if m:
            result = pattern.sub(replacement, text).strip()
            # Verificar que el resultado sea más corto (eliminamos algo)
            # y que siga siendo válido
            if result != text and len(result) >= 4:
                return result
    return original

def format_mx_plate(clean: str) -> str:
    """
    Agrega guiones al texto limpio para mostrarlo en formato visual MX.
    Ejemplos:
      CVL657B  → CVL-657-B
      CVL6571  → CVL-6571
      CV6571   → CV-6571
      NNNLLL   → NNN-LLL
    Si no coincide con ningún patrón conocido, devuelve el texto sin guiones.
    """
    n = len(clean)
    # LLL-NNN-L  (7 chars: 3L 3N 1L)
    if n == 7 and clean[:3].isalpha() and clean[3:6].isdigit() and clean[6].isalpha():
        return f"{clean[:3]}-{clean[3:6]}-{clean[6]}"
    # LLL-NNNN   (7 chars: 3L 4N)
    if n == 7 and clean[:3].isalpha() and clean[3:].isdigit():
        return f"{clean[:3]}-{clean[3:]}"
    # LLL-NNN    (6 chars: 3L 3N)
    if n == 6 and clean[:3].isalpha() and clean[3:].isdigit():
        return f"{clean[:3]}-{clean[3:]}"
    # LL-NNNN    (6 chars: 2L 4N)
    if n == 6 and clean[:2].isalpha() and clean[2:].isdigit():
        return f"{clean[:2]}-{clean[2:]}"
    # LL-NNN-L   (6 chars: 2L 3N 1L)
    if n == 6 and clean[:2].isalpha() and clean[2:5].isdigit() and clean[5].isalpha():
        return f"{clean[:2]}-{clean[2:5]}-{clean[5]}"
    # NNN-LLL    (6 chars: 3N 3L)
    if n == 6 and clean[:3].isdigit() and clean[3:].isalpha():
        return f"{clean[:3]}-{clean[3:]}"
    # LL-NNN     (5 chars: 2L 3N)
    if n == 5 and clean[:2].isalpha() and clean[2:].isdigit():
        return f"{clean[:2]}-{clean[2:]}"
    return clean


def _best_mask(text: str) -> str | None:
    """Retorna la máscara MX más probable para el texto dado."""
    n = len(text)
    for mask in _MX_MASKS:
        if len(mask) == n:
            return mask
    # Si no hay coincidencia exacta, usar la más cercana en longitud
    if _MX_MASKS:
        return sorted(_MX_MASKS, key=lambda m: abs(len(m) - n))[0]
    return None

def _apply_mask_correction(text: str, mask: str) -> str:
    result = []
    for i, ch in enumerate(text):
        if i >= len(mask):
            result.append(ch); continue
        if mask[i] == "L":
            result.append(_TO_LETTER.get(ch, ch))
        else:
            result.append(_TO_DIGIT.get(ch, ch))
    return "".join(result)

def correct_mx_plate(raw: str) -> str:
    """
    Pipeline completo de corrección para placas mexicanas:
    1. Detectar si tiene guiones reales (preservarlos)
    2. Eliminar espacios y caracteres no válidos
    3. Eliminar 'I'/'1' que son guiones mal leídos por OCR
    4. Detectar máscara más probable
    5. Aplicar corrección posicional
    6. Truncar a longitud máxima válida (8 chars)
    7. Re-aplicar guiones si estaban presentes originalmente
    """
    raw_upper = raw.upper()
    
    # Paso 1: Detectar si tiene guiones reales (formato correcto)
    # Ej: "ABC-123-D" o "AB-1234"
    has_real_dashes = '-' in raw_upper
    dash_positions = [i for i, c in enumerate(raw_upper) if c == '-']
    
    # Paso 2: limpiar — quitar guiones y todo lo que no sea letra/dígito
    clean = "".join(c for c in raw_upper if c.isalnum())
    if len(clean) < 4:
        return clean
    
    # Paso 3: eliminar guiones falsos (I/1 entre grupos de letras y números)
    # SOLO si NO había guiones reales detectados
    if not has_real_dashes:
        clean = _remove_false_dashes(clean)
        if len(clean) < 4:
            return clean
    
    # Paso 4-5: máscara y corrección posicional
    mask = _best_mask(clean)
    if mask:
        clean = _apply_mask_correction(clean, mask)
    
    # Paso 6: truncar a longitud máxima de placa MX (7 chars para federal, 8 máx)
    if len(clean) > 8:
        clean = clean[:8]
    
    # Paso 7: Re-aplicar formato con guiones si estaban presentes
    # o si la placa es válida y queremos formato visual
    if has_real_dashes and len(dash_positions) > 0:
        # Preservar el formato original con guiones
        return format_mx_plate(clean)
    
    return clean

# ── Validación geométrica de placas ──────────────────────────────────────────

def validate_plate_geometry(bbox: tuple, img_shape: tuple) -> tuple[bool, float]:
    """
    SISTEMA DE QUALITY SCORE AVANZADO para placas vehiculares.
    
    Usa múltiples señales para evaluar la calidad de una detección:
    1. Nitidez (Laplacian variance)
    2. Contraste (std)
    3. Densidad de bordes (Canny)
    4. Aspect ratio (proporción correcta)
    5. Alineación horizontal (rotación)
    6. Ocupación del frame (área relativa)
    7. Brillo (rango óptimo)
    8. Posición en frame
    
    Retorna (es_valida, score_confianza_0_1)
    """
    # Validación de entrada
    if not bbox or len(bbox) != 4:
        return False, 0.0
    if not img_shape or len(img_shape) < 2:
        return False, 0.0
    
    try:
        x1, y1, x2, y2 = bbox
        img_h, img_w = img_shape[:2]
    except (ValueError, TypeError):
        return False, 0.0
    
    # Validar que las coordenadas sean números válidos
    if not all(isinstance(v, (int, float)) for v in [x1, y1, x2, y2, img_h, img_w]):
        return False, 0.0
    
    w, h = x2 - x1, y2 - y1
    if h <= 0 or w <= 0:
        return False, 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # FILTROS BÁSICOS (rechazo temprano)
    # ═══════════════════════════════════════════════════════════════════════
    
    # Tamaño mínimo absoluto
    if w < 60 or h < 18:
        return False, 0.0
    
    # Aspect ratio (placas MX: 1.5 - 8.0, relajado para ángulos)
    aspect_ratio = w / h
    if not (1.5 <= aspect_ratio <= 8.0):
        return False, 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # CÁLCULO DE SEÑALES DE CALIDAD
    # ═══════════════════════════════════════════════════════════════════════
    
    score = 0.0
    
    # ── 1. ASPECT RATIO (20%) ─────────────────────────────────────────────
    # Placas MX típicas: 3.0 - 4.5
    if 3.0 <= aspect_ratio <= 4.5:
        aspect_score = 1.0  # Óptimo
    elif 2.5 <= aspect_ratio <= 5.5:
        aspect_score = 0.8  # Bueno
    elif 2.0 <= aspect_ratio <= 6.5:
        aspect_score = 0.6  # Aceptable
    else:
        aspect_score = 0.3  # Marginal
    score += aspect_score * 0.20
    
    # ── 2. OCUPACIÓN DEL FRAME (15%) ──────────────────────────────────────
    # Placas típicas: 0.5% - 8% del frame
    area = w * h
    img_area = img_h * img_w
    relative_area = area / img_area
    
    if 0.005 <= relative_area <= 0.08:
        area_score = 1.0  # Óptimo
    elif 0.002 <= relative_area <= 0.15:
        area_score = 0.7  # Aceptable
    else:
        area_score = 0.3  # Muy pequeña o muy grande
    score += area_score * 0.15
    
    # ── 3. POSICIÓN EN FRAME (10%) ────────────────────────────────────────
    # Placas suelen estar en mitad inferior o centro
    center_y = (y1 + y2) / 2
    center_x = (x1 + x2) / 2
    
    # Vertical: preferir mitad inferior (40%-90% de altura)
    y_ratio = center_y / img_h
    if 0.4 <= y_ratio <= 0.9:
        pos_score_y = 1.0
    elif 0.2 <= y_ratio <= 0.95:
        pos_score_y = 0.7
    else:
        pos_score_y = 0.3
    
    # Horizontal: preferir centro (20%-80% de ancho)
    x_ratio = center_x / img_w
    if 0.2 <= x_ratio <= 0.8:
        pos_score_x = 1.0
    else:
        pos_score_x = 0.7
    
    pos_score = (pos_score_y * 0.7 + pos_score_x * 0.3)
    score += pos_score * 0.10
    
    # ── 4. MÁRGENES (5%) ──────────────────────────────────────────────────
    # No muy cerca de los bordes (evita detecciones parciales)
    margin = 5
    if x1 > margin and y1 > margin and x2 < img_w - margin and y2 < img_h - margin:
        margin_score = 1.0
    else:
        margin_score = 0.5
    score += margin_score * 0.05
    
    # ═══════════════════════════════════════════════════════════════════════
    # SEÑALES AVANZADAS (requieren crop de imagen)
    # Solo calcular si tenemos acceso a la imagen completa
    # ═══════════════════════════════════════════════════════════════════════
    
    # Nota: Las siguientes señales (nitidez, contraste, densidad de bordes,
    # alineación horizontal, brillo) requieren el crop de la imagen.
    # Como validate_plate_geometry solo recibe bbox e img_shape, estas señales
    # se calculan en plate_quality_score_advanced() que se llama después.
    
    # Por ahora, asignar el 50% restante del score basado en geometría
    # (será complementado por plate_quality_score_advanced)
    
    # ── 5. TAMAÑO ABSOLUTO (10%) ──────────────────────────────────────────
    # Placas más grandes = mejor calidad de OCR
    if area > 20000:  # Grande (4K)
        size_score = 1.0
    elif area > 10000:  # Mediano (HD)
        size_score = 0.8
    elif area > 5000:  # Pequeño pero aceptable
        size_score = 0.6
    else:
        size_score = 0.4
    score += size_score * 0.10
    
    # ── 6. FORMA RECTANGULAR (10%) ────────────────────────────────────────
    # Placas deben ser rectangulares (no cuadradas ni muy alargadas)
    if 2.5 <= aspect_ratio <= 5.0:
        rect_score = 1.0
    elif 2.0 <= aspect_ratio <= 6.0:
        rect_score = 0.7
    else:
        rect_score = 0.4
    score += rect_score * 0.10
    
    # ── 7. PROPORCIÓN ALTURA/ANCHO DEL FRAME (5%) ─────────────────────────
    # Placas horizontales en frames horizontales = mejor
    frame_aspect = img_w / img_h
    if frame_aspect > 1.0 and aspect_ratio > 2.0:  # Ambos horizontales
        frame_match_score = 1.0
    else:
        frame_match_score = 0.7
    score += frame_match_score * 0.05
    
    # ── 8. DENSIDAD ESPACIAL (5%) ─────────────────────────────────────────
    # Placas no deben ser ni muy densas ni muy dispersas
    density = area / (w + h)  # Área por perímetro
    if 10 <= density <= 50:
        density_score = 1.0
    elif 5 <= density <= 80:
        density_score = 0.7
    else:
        density_score = 0.4
    score += density_score * 0.05
    
    # ── 9. ESTABILIDAD (10%) ──────────────────────────────────────────────
    # Placas con dimensiones estables (no extremas) son más confiables
    if 80 <= w <= 400 and 20 <= h <= 120:
        stability_score = 1.0
    elif 60 <= w <= 600 and 18 <= h <= 150:
        stability_score = 0.7
    else:
        stability_score = 0.4
    score += stability_score * 0.10
    
    # ── 10. BONUS POR CARACTERÍSTICAS IDEALES (10%) ───────────────────────
    bonus = 0.0
    
    # Bonus: aspect ratio ideal (3.5 - 4.0)
    if 3.5 <= aspect_ratio <= 4.0:
        bonus += 0.03
    
    # Bonus: tamaño ideal (área 8000-15000)
    if 8000 <= area <= 15000:
        bonus += 0.03
    
    # Bonus: posición ideal (centro-inferior)
    if 0.5 <= y_ratio <= 0.8 and 0.3 <= x_ratio <= 0.7:
        bonus += 0.02
    
    # Bonus: dimensiones típicas de placa MX
    if 100 <= w <= 300 and 25 <= h <= 80:
        bonus += 0.02
    
    score += bonus
    
    # Normalizar score a [0, 1]
    score = min(1.0, max(0.0, score))
    
    # Umbral de aceptación: 0.40 (más permisivo que antes)
    is_valid = score >= 0.40
    
    return is_valid, score


def plate_quality_score_advanced(crop: np.ndarray) -> tuple[float, dict]:
    """
    QUALITY SCORE AVANZADO OPTIMIZADO para crops de placas.
    
    OPTIMIZACIÓN: Reducir operaciones costosas (Canny, HoughLines) para tiempo real.
    Calcula señales rápidas pero efectivas:
    1. Nitidez (Laplacian variance) - RÁPIDO
    2. Contraste (std) - RÁPIDO
    3. Brillo (mean) - RÁPIDO
    4. Aspect ratio - RÁPIDO
    5. Tamaño - RÁPIDO
    6. Gradiente horizontal (aproximación de alineación) - RÁPIDO
    7. Densidad de píxeles oscuros (aproximación de bordes) - RÁPIDO
    
    Retorna (score_0_1, detalles_dict)
    """
    # Validación robusta de entrada
    if crop is None:
        return 0.5, {}
    
    try:
        if crop.size == 0:
            return 0.5, {}
    except Exception:
        return 0.5, {}
    
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    except Exception:
        return 0.5, {}
    
    try:
        h, w = gray.shape[:2]
    except Exception:
        return 0.5, {}
    
    if h < 10 or w < 40:
        return 0.5, {}
    
    score = 0.0
    details = {}
    
    # ── 1. NITIDEZ (30%) ──────────────────────────────────────────────────
    # Laplacian variance: mayor = más nítida (RÁPIDO: ~0.5ms)
    try:
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        details["sharpness"] = float(sharpness)
        
        if sharpness >= 150:
            sharp_score = 1.0  # Excelente
        elif sharpness >= 80:
            sharp_score = 0.8  # Buena
        elif sharpness >= 30:
            sharp_score = 0.5  # Aceptable
        else:
            sharp_score = 0.2  # Borrosa
        score += sharp_score * 0.30
    except Exception:
        details["sharpness"] = 0.0
        score += 0.15  # Score neutral
    
    # ── 2. CONTRASTE (25%) ────────────────────────────────────────────────
    # Desviación estándar: mayor = mejor separación texto/fondo (RÁPIDO: ~0.1ms)
    try:
        contrast = float(gray.std())
        details["contrast"] = contrast
        
        if contrast >= 50:
            contrast_score = 1.0  # Excelente
        elif contrast >= 35:
            contrast_score = 0.8  # Bueno
        elif contrast >= 20:
            contrast_score = 0.5  # Aceptable
        else:
            contrast_score = 0.2  # Bajo
        score += contrast_score * 0.25
    except Exception:
        details["contrast"] = 0.0
        score += 0.125  # Score neutral
    
    # ── 3. BRILLO (15%) ───────────────────────────────────────────────────
    # Mean intensity: rango óptimo 80-180 (RÁPIDO: ~0.1ms)
    try:
        brightness = float(gray.mean())
        details["brightness"] = brightness
        
        if 80 <= brightness <= 180:
            bright_score = 1.0  # Óptimo
        elif 60 <= brightness <= 200:
            bright_score = 0.7  # Aceptable
        else:
            bright_score = 0.3  # Muy oscuro o muy claro
        score += bright_score * 0.15
    except Exception:
        details["brightness"] = 0.0
        score += 0.075  # Score neutral
    
    # ── 4. DENSIDAD DE PÍXELES OSCUROS (15%) ─────────────────────────────
    # Aproximación rápida de densidad de bordes sin Canny (RÁPIDO: ~0.2ms)
    # Caracteres oscuros en fondo claro = muchos píxeles < 100
    try:
        dark_pixels = int(np.count_nonzero(gray < 100))
        dark_density = float(dark_pixels) / float(w * h)
        details["dark_density"] = dark_density
        
        # Placas típicas: 20-50% de píxeles oscuros (caracteres)
        if 0.20 <= dark_density <= 0.50:
            dark_score = 1.0  # Óptimo
        elif 0.10 <= dark_density <= 0.60:
            dark_score = 0.7  # Aceptable
        else:
            dark_score = 0.3  # Muy pocos o muchos
        score += dark_score * 0.15
    except Exception:
        details["dark_density"] = 0.0
        score += 0.075  # Score neutral
    
    # ── 5. GRADIENTE HORIZONTAL (10%) ─────────────────────────────────────
    # Aproximación rápida de alineación sin HoughLines (RÁPIDO: ~0.3ms)
    # Sobel horizontal: placas bien alineadas tienen gradiente uniforme
    try:
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Ratio de gradiente horizontal vs vertical
        grad_x_mean = float(np.abs(sobel_x).mean())
        grad_y_mean = float(np.abs(sobel_y).mean())
        
        if grad_y_mean > 0.01:  # Evitar división por cero
            grad_ratio = grad_x_mean / grad_y_mean
            details["grad_ratio"] = grad_ratio
            
            # Placas horizontales: más gradiente vertical (bordes de caracteres)
            # que horizontal (alineación)
            if 0.5 <= grad_ratio <= 2.0:
                grad_score = 1.0  # Bien alineada
            elif 0.3 <= grad_ratio <= 3.0:
                grad_score = 0.7  # Aceptable
            else:
                grad_score = 0.4  # Rotada o mal alineada
        else:
            details["grad_ratio"] = 0.0
            grad_score = 0.5
        
        score += grad_score * 0.10
    except Exception:
        details["grad_ratio"] = 0.0
        score += 0.05  # Score neutral
    
    # ── 6. ASPECT RATIO (3%) ──────────────────────────────────────────────
    try:
        aspect = float(w) / max(float(h), 1.0)
        details["aspect"] = aspect
        
        if 2.5 <= aspect <= 5.0:
            aspect_score = 1.0  # Ideal
        elif 2.0 <= aspect <= 6.0:
            aspect_score = 0.7  # Aceptable
        else:
            aspect_score = 0.3  # Fuera de rango
        score += aspect_score * 0.03
    except Exception:
        details["aspect"] = 0.0
        score += 0.015  # Score neutral
    
    # ── 7. TAMAÑO (2%) ────────────────────────────────────────────────────
    try:
        area = int(w * h)
        details["area"] = area
        
        if area > 20000:
            size_score = 1.0  # Grande (4K)
        elif area > 10000:
            size_score = 0.8  # Mediano (HD)
        elif area > 5000:
            size_score = 0.6  # Pequeño
        else:
            size_score = 0.3  # Muy pequeño
        score += size_score * 0.02
    except Exception:
        details["area"] = 0
        score += 0.01  # Score neutral
    
    # Normalizar score
    score = float(min(1.0, max(0.0, score)))
    details["score"] = score
    
    return score, details


# ── Segmentación mejorada de caracteres ──────────────────────────────────────

def _segment_by_contours(plate_bgr: np.ndarray, img_size: int = 32) -> list[np.ndarray]:
    """Segmentación por detección de contornos (fallback)."""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar por tamaño y aspect ratio
    char_candidates = []
    h_plate = plate_bgr.shape[0]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filtros geométricos para caracteres
        if h < h_plate * 0.4 or h > h_plate * 0.95:
            continue
        if w < 4 or w > h * 1.5:
            continue
        if w * h < 50:  # Área mínima
            continue
        
        char_candidates.append((x, y, w, h))
    
    # Ordenar por posición X
    char_candidates.sort(key=lambda c: c[0])
    
    # Extraer crops
    crops = []
    for (x, y, w, h) in char_candidates:
        crop = binary[y:y+h, x:x+w]
        if crop.size > 0:
            # Padding cuadrado
            ch, cw = crop.shape
            pad = max(0, ch - cw) // 2
            crop = cv2.copyMakeBorder(crop, 2, 2, pad+2, pad+2,
                                      cv2.BORDER_CONSTANT, value=255)
            crop = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_AREA)
            crops.append(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
    
    return crops


def _segment_by_watershed(plate_bgr: np.ndarray, img_size: int = 32) -> list[np.ndarray]:
    """Segmentación por watershed (para casos difíciles)."""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Operaciones morfológicas
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    
    # Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Watershed
    markers = cv2.watershed(plate_bgr, markers)
    
    # Extraer regiones
    crops = []
    h_plate = plate_bgr.shape[0]
    regions = []
    
    for label in np.unique(markers):
        if label <= 1:  # Skip background
            continue
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers == label] = 255
        
        x, y, w, h = cv2.boundingRect(mask)
        
        # Filtros geométricos
        if h < h_plate * 0.3 or w < 4:
            continue
        
        regions.append((x, y, w, h, mask))
    
    # Ordenar por X
    regions.sort(key=lambda r: r[0])
    
    for (x, y, w, h, mask) in regions:
        # Recortar imagen y máscara al mismo bbox antes de bitwise_and
        roi      = plate_bgr[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]
        if roi.size == 0 or roi_mask.shape != roi.shape[:2]:
            continue
        crop = cv2.bitwise_and(roi, roi, mask=roi_mask)
        if crop.size > 0:
            ch, cw = crop.shape[:2]
            pad = max(0, ch - cw) // 2
            crop = cv2.copyMakeBorder(crop, 2, 2, pad+2, pad+2,
                                      cv2.BORDER_CONSTANT, value=255)
            crop = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_AREA)
            crops.append(crop)
    
    return crops


# ── CNN de caracteres ────────────────────────────────────────────────────────

class _SEBlock(torch.nn.Module):
    """Squeeze-and-Excitation block — identico a train_char_model.py."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(channels, max(channels // reduction, 4)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(max(channels // reduction, 4), channels),
            torch.nn.Sigmoid(),
        )
    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class _ConvBnRelu(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class _CharCNN(torch.nn.Module):
    """Misma arquitectura que train_char_model.py — debe coincidir exactamente."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            _ConvBnRelu(1, 32), _ConvBnRelu(32, 32),
            _SEBlock(32, reduction=8),
            torch.nn.MaxPool2d(2, 2), torch.nn.Dropout2d(0.10),
        )
        self.block2 = torch.nn.Sequential(
            _ConvBnRelu(32, 64), _ConvBnRelu(64, 64),
            _SEBlock(64, reduction=8),
            torch.nn.MaxPool2d(2, 2), torch.nn.Dropout2d(0.15),
        )
        self.block3 = torch.nn.Sequential(
            _ConvBnRelu(64, 128),
            _SEBlock(128, reduction=8),
            torch.nn.MaxPool2d(2, 2), torch.nn.Dropout2d(0.20),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 4 * 4, 256), torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.40),
            torch.nn.Linear(256, 128), torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


class CharRecognizer:
    """
    Carga char_cnn.pth y expone predict_plate(plate_bgr) -> (texto, confianza).
    Segmenta caracteres por proyección de perfil y clasifica cada uno con la CNN.
    """
    CNN_PATH    = "char_cnn.pth"
    CLASSES_PATH= "char_classes.json"
    IMG_SIZE    = 32

    def __init__(self):
        self._model  = None
        self._chars  = None
        self._device = GPU_DEVICE
        self._tf     = None
        self._loaded = False
        self._last_mtime = 0   # timestamp del archivo para detectar cambios
        self._load()

    def _load(self):
        if not os.path.exists(self.CNN_PATH):
            print(f"[CharCNN] {self.CNN_PATH} no encontrado — usando solo EasyOCR")
            return
        try:
            # Detectar si el archivo cambió (reentrenamiento)
            mtime = os.path.getmtime(self.CNN_PATH)
            if self._loaded and mtime == self._last_mtime:
                return   # ya está cargado y no cambió
            self._last_mtime = mtime

            ckpt = torch.load(self.CNN_PATH, map_location=self._device,
                              weights_only=False)
            self._chars = ckpt["chars"]
            n = ckpt["num_classes"]
            self.IMG_SIZE = ckpt.get("img_size", 32)
            model = _CharCNN(n).to(self._device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            self._model = model
            from torchvision import transforms as T
            self._tf = T.Compose([
                T.Grayscale(1),
                T.Resize((self.IMG_SIZE, self.IMG_SIZE)),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ])
            acc = ckpt.get("val_acc", 0)
            self._loaded = True
            status = "Recargado" if mtime != 0 else "Cargado"
            print(f"[CharCNN] {status} — val_acc={acc:.1f}%  device={self._device}")
        except Exception as e:
            print(f"[CharCNN] Error cargando modelo: {e}")

    def reload_if_updated(self):
        """Recarga el modelo si el archivo cambió (llamar periódicamente)."""
        if os.path.exists(self.CNN_PATH):
            mtime = os.path.getmtime(self.CNN_PATH)
            if mtime != self._last_mtime:
                print("[CharCNN] Detectado modelo actualizado, recargando...")
                self._load()

    @property
    def available(self) -> bool:
        return self._loaded and self._model is not None

    # ── Segmentación de caracteres ───────────────────────────────────────────
    def _segment(self, plate_bgr: np.ndarray) -> list[np.ndarray]:
        """
        Segmentación multi-estrategia con fallback automático.
        1. Proyección vertical (rápida, funciona en la mayoría de casos)
        2. Contornos (fallback si proyección falla)
        3. Watershed (para placas muy degradadas)
        Retorna lista de crops BGR ordenados izquierda→derecha.
        """
        h, w = plate_bgr.shape[:2]
        if h < 10 or w < 30:
            return []

        # Estrategia 1: Proyección vertical (método actual optimizado)
        crops_projection = self._segment_by_projection(plate_bgr)
        
        # Estrategia 2: Contornos (fallback si proyección falla)
        if len(crops_projection) < 4:
            crops_contours = _segment_by_contours(plate_bgr, self.IMG_SIZE)
            if len(crops_contours) >= len(crops_projection):
                crops_projection = crops_contours
        
        # Estrategia 3: Watershed (para placas muy degradadas)
        if len(crops_projection) < 4:
            crops_watershed = _segment_by_watershed(plate_bgr, self.IMG_SIZE)
            if len(crops_watershed) >= len(crops_projection):
                crops_projection = crops_watershed
        
        return crops_projection

    def _segment_by_projection(self, plate_bgr: np.ndarray) -> list[np.ndarray]:
        """
        Segmenta caracteres por proyección vertical de perfil.
        Filtra automáticamente guiones y separadores de placas MX.
        Incluye corrección de inclinación (deskew) y supresión de reflejos.
        """
        h, w = plate_bgr.shape[:2]
        if h < 10 or w < 30:
            return []

        # ── Suprimir reflejos antes de segmentar ─────────────────────────────
        plate_bgr = _remove_glare(plate_bgr)

        # Normalizar altura a 48px
        scale  = 48.0 / h
        new_w  = max(int(w * scale), 60)
        scaled = cv2.resize(plate_bgr, (new_w, 48), interpolation=cv2.INTER_CUBIC)
        gray   = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        # CLAHE + binarización
        clahe  = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4,4))
        gray   = clahe.apply(gray)
        _, bin_img = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Auto-orientar: fondo debe ser blanco
        if bin_img[:, :6].mean() < 128:
            bin_img = cv2.bitwise_not(bin_img)

        # ── Corrección de inclinación (deskew) ───────────────────────────────
        # Detecta el ángulo de inclinación del texto y lo corrige
        # Solo aplica si la inclinación es significativa (>1°) y pequeña (<15°)
        try:
            coords = np.column_stack(np.where((255 - bin_img) > 0))
            if len(coords) > 50:
                angle = cv2.minAreaRect(coords)[2]
                # minAreaRect devuelve ángulos en [-90, 0), normalizar
                if angle < -45:
                    angle = 90 + angle
                if abs(angle) > 1.0 and abs(angle) < 15.0:
                    M = cv2.getRotationMatrix2D((new_w/2, 24), angle, 1.0)
                    bin_img = cv2.warpAffine(bin_img, M, (new_w, 48),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=255)
        except Exception:
            pass  # Si falla el deskew, continuar sin él

        # Proyección vertical (suma de píxeles negros por columna)
        col_sum   = (255 - bin_img).sum(axis=0).astype(float)
        threshold = col_sum.max() * 0.12
        if threshold < 1:
            return []

        # Encontrar segmentos con contenido
        in_char, start = False, 0
        segments = []
        for x, val in enumerate(col_sum):
            if not in_char and val > threshold:
                in_char, start = True, x
            elif in_char and val <= threshold:
                in_char = False
                if x - start >= 4:
                    segments.append((start, x))
        if in_char and new_w - start >= 4:
            segments.append((start, new_w))

        # ── Calcular ancho promedio de segmentos para detectar guiones ────────
        # Los guiones son mucho más delgados que los caracteres reales.
        # Un carácter MX típico ocupa ~6-14px de ancho (a 48px de alto).
        # Un guión ocupa ~2-4px. Filtramos segmentos demasiado delgados.
        if segments:
            widths = [x2 - x1 for x1, x2 in segments]
            avg_w  = sum(widths) / len(widths)
            # Umbral: segmento es guión si su ancho < 40% del promedio
            # Y además su altura de tinta es < 25% de la altura total (guión horizontal)
            MIN_CHAR_WIDTH_RATIO = 0.40
            filtered_segments = []
            for (x1, x2) in segments:
                seg_w = x2 - x1
                if seg_w < avg_w * MIN_CHAR_WIDTH_RATIO:
                    # Verificar si es un guión horizontal (tinta concentrada en el centro)
                    seg_col = col_sum[x1:x2]
                    seg_ink = bin_img[:, x1:x2]
                    # Proyección horizontal: filas con tinta
                    row_sum = (255 - seg_ink).sum(axis=1)
                    ink_rows = (row_sum > row_sum.max() * 0.3).sum()
                    # Si la tinta ocupa menos del 30% de las filas → es guión
                    if ink_rows < 48 * 0.30:
                        continue   # descartar guión
                filtered_segments.append((x1, x2))
            segments = filtered_segments

        # Extraer crops y redimensionar a IMG_SIZE×IMG_SIZE
        crops = []
        for (x1, x2) in segments:
            crop = bin_img[:, x1:x2]
            ch, cw = crop.shape
            if ch < 8 or cw < 3:
                continue
            # Padding cuadrado
            pad = max(0, ch - cw) // 2
            crop = cv2.copyMakeBorder(crop, 2, 2, pad+2, pad+2,
                                      cv2.BORDER_CONSTANT, value=255)
            crop = cv2.resize(crop, (self.IMG_SIZE, self.IMG_SIZE),
                              interpolation=cv2.INTER_AREA)
            crops.append(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
        return crops

    # ── Predicción ───────────────────────────────────────────────────────────
    def predict_plate(self, plate_bgr: np.ndarray,
                      min_chars: int = 3) -> tuple[str | None, float]:
        """
        Segmenta y clasifica cada carácter con la CNN.
        
        CAMBIO: min_chars reducido de 4 → 3 para capturar más placas.
        Razón: Placas cortas (ej: "SG734") tienen solo 5 caracteres.
        Si la segmentación falla y solo detecta 3, aún podemos intentar OCR.
        
        Mejoras:
          - Multi-escala: prueba 3 alturas de normalización y elige la que
            produce el número de caracteres más cercano al esperado (5-7 MX).
          - Top-2 por carácter: si la confianza del top-1 es baja (<0.55),
            intenta el top-2 y valida si produce formato MX correcto.
          - Filtro de longitud: descarta resultados con < min_chars o > 8 chars.
        Retorna (texto_placa, confianza_promedio) o (None, 0) si falla.
        """
        if not self.available:
            return None, 0.0

        # Validar entrada
        if plate_bgr is None or plate_bgr.size == 0:
            return None, 0.0
        
        h, w = plate_bgr.shape[:2]
        if h < 10 or w < 30:
            return None, 0.0

        # ── Multi-escala: segmentar a 3 alturas y elegir la mejor ────────────
        best_crops = []
        best_score = -1.0
        for target_h in (48, 64, 32):
            crops_candidate = self._segment_multiscale(plate_bgr, target_h)
            n = len(crops_candidate)
            if 5 <= n <= 7:
                score = 1.0 - abs(n - 6) * 0.1
            elif 3 <= n <= 8:  # CAMBIO: aceptar desde 3 chars
                score = 0.6 - abs(n - 6) * 0.05
            else:
                score = 0.0
            if score > best_score and n >= min_chars:
                best_score = score
                best_crops = crops_candidate

        if len(best_crops) < min_chars:
            # DEBUG: Informar por qué falló
            # print(f"[CharCNN] Segmentación falló: {len(best_crops)} caracteres (min={min_chars})")
            return None, 0.0

        crops = best_crops[:8]

        # ── Clasificación con top-2 por carácter ─────────────────────────────
        from PIL import Image as PILImage
        tensors = []
        for crop in crops:
            pil = PILImage.fromarray(crop)
            tensors.append(self._tf(pil))
        batch = torch.stack(tensors).to(self._device)

        with torch.no_grad():
            logits = self._model(batch)
            probs  = torch.softmax(logits, dim=1)
            top2_confs, top2_preds = probs.topk(min(2, probs.shape[1]), dim=1)

        top1_chars = [self._chars[top2_preds[i, 0].item()] for i in range(len(crops))]
        top1_confs = [top2_confs[i, 0].item() for i in range(len(crops))]
        top2_chars = [self._chars[top2_preds[i, 1].item()] for i in range(len(crops))]

        final_chars = list(top1_chars)
        avg_conf = sum(top1_confs) / len(top1_confs)

        # ── Corrección con top-2 en posiciones de baja confianza ─────────────
        LOW_CONF = 0.55
        low_positions = [i for i, c in enumerate(top1_confs) if c < LOW_CONF]
        if low_positions:
            text_top1 = correct_mx_plate("".join(final_chars))
            valid_top1, _ = validate_mx_plate(text_top1)
            if not valid_top1:
                for pos in low_positions:
                    alt = list(final_chars)
                    alt[pos] = top2_chars[pos]
                    alt_text = correct_mx_plate("".join(alt))
                    valid_alt, _ = validate_mx_plate(alt_text)
                    if valid_alt:
                        final_chars[pos] = top2_chars[pos]
                        avg_conf = min(1.0, avg_conf + 0.08)
                        break

        text = correct_mx_plate("".join(final_chars))
        return text, avg_conf

    def _segment_multiscale(self, plate_bgr: np.ndarray,
                             target_h: int = 48) -> list[np.ndarray]:
        """
        Segmentación con altura objetivo configurable para multi-escala.
        """
        h, w = plate_bgr.shape[:2]
        if h < 10 or w < 30:
            return []

        plate_bgr = _remove_glare(plate_bgr)
        scale  = target_h / h
        new_w  = max(int(w * scale), 60)
        scaled = cv2.resize(plate_bgr, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
        gray   = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        clahe  = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
        gray   = clahe.apply(gray)
        _, bin_img = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if bin_img[:, :6].mean() < 128:
            bin_img = cv2.bitwise_not(bin_img)

        try:
            coords = np.column_stack(np.where((255 - bin_img) > 0))
            if len(coords) > 50:
                angle = cv2.minAreaRect(coords)[2]
                if angle < -45:
                    angle = 90 + angle
                if abs(angle) > 1.0 and abs(angle) < 15.0:
                    M = cv2.getRotationMatrix2D((new_w / 2, target_h / 2), angle, 1.0)
                    bin_img = cv2.warpAffine(bin_img, M, (new_w, target_h),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=255)
        except Exception:
            pass

        col_sum   = (255 - bin_img).sum(axis=0).astype(float)
        threshold = col_sum.max() * 0.12
        if threshold < 1:
            return []

        in_char, start = False, 0
        segments = []
        for x, val in enumerate(col_sum):
            if not in_char and val > threshold:
                in_char, start = True, x
            elif in_char and val <= threshold:
                in_char = False
                if x - start >= 3:
                    segments.append((start, x))
        if in_char and new_w - start >= 3:
            segments.append((start, new_w))

        if segments:
            widths = [x2 - x1 for x1, x2 in segments]
            avg_w  = sum(widths) / len(widths)
            filtered = []
            for (x1, x2) in segments:
                seg_w = x2 - x1
                if seg_w < avg_w * 0.35:
                    seg_ink = bin_img[:, x1:x2]
                    row_sum = (255 - seg_ink).sum(axis=1)
                    max_row = row_sum.max()
                    ink_rows = (row_sum > max_row * 0.3).sum() if max_row > 0 else 0
                    if ink_rows < target_h * 0.28:
                        continue
                filtered.append((x1, x2))
            segments = filtered

        crops = []
        for (x1, x2) in segments:
            crop = bin_img[:, x1:x2]
            ch, cw = crop.shape
            if ch < 6 or cw < 2:
                continue
            pad = max(0, ch - cw) // 2
            crop = cv2.copyMakeBorder(crop, 2, 2, pad + 2, pad + 2,
                                      cv2.BORDER_CONSTANT, value=255)
            crop = cv2.resize(crop, (self.IMG_SIZE, self.IMG_SIZE),
                              interpolation=cv2.INTER_AREA)
            crops.append(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
        return crops


# Instancia global — se inicializa una sola vez al cargar el módulo
_char_recognizer: CharRecognizer | None = None

def get_char_recognizer() -> CharRecognizer:
    global _char_recognizer
    if _char_recognizer is None:
        _char_recognizer = CharRecognizer()
    return _char_recognizer


# ── Preprocesamiento para EasyOCR ────────────────────────────────────────────

def _is_dark_background(gray: np.ndarray) -> bool:
    h, w = gray.shape
    border = np.concatenate([gray[:4,:].flatten(), gray[-4:,:].flatten(),
                              gray[:,:4].flatten(), gray[:,-4:].flatten()])
    center = gray[h//4:3*h//4, w//4:3*w//4].flatten()
    return float(border.mean()) < float(center.mean())


def _remove_glare(img_bgr: np.ndarray) -> np.ndarray:
    """
    Suprime reflejos (glare) en imágenes de placas vehiculares.
    Estrategia:
      1. Detectar zonas saturadas (píxeles muy brillantes en todos los canales)
      2. Inpainting de esas zonas con el contexto circundante
      3. Normalización de iluminación con división por canal de valor HSV
    """
    # ── Paso 1: máscara de zonas saturadas ───────────────────────────────────
    # Un reflejo real satura los 3 canales simultáneamente
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Canal V (brillo): píxeles con V > 240 Y S < 40 son reflejos puros
    _, glare_v = cv2.threshold(hsv[:,:,2], 240, 255, cv2.THRESH_BINARY)
    _, glare_s = cv2.threshold(hsv[:,:,1],  40, 255, cv2.THRESH_BINARY_INV)
    glare_mask = cv2.bitwise_and(glare_v, glare_s)

    # Dilatar la máscara para cubrir bordes del reflejo
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    glare_mask = cv2.dilate(glare_mask, kernel, iterations=2)

    # ── Paso 2: inpainting si hay zonas saturadas significativas ─────────────
    glare_ratio = glare_mask.sum() / (255 * glare_mask.size)
    if glare_ratio > 0.01:   # más del 1% de la imagen tiene reflejo
        result = cv2.inpaint(img_bgr, glare_mask, inpaintRadius=5,
                             flags=cv2.INPAINT_TELEA)
    else:
        result = img_bgr.copy()

    # ── Paso 3: normalización de iluminación (homomorphic-like) ──────────────
    # Divide cada canal por una versión suavizada de sí mismo para
    # eliminar gradientes de iluminación no uniformes (sombras + reflejos suaves)
    result_float = result.astype(np.float32) + 1.0
    for c in range(3):
        blur = cv2.GaussianBlur(result_float[:,:,c], (0, 0), sigmaX=15)
        blur = np.maximum(blur, 1.0)
        result_float[:,:,c] = (result_float[:,:,c] / blur) * 128.0
    result_norm = np.clip(result_float, 0, 255).astype(np.uint8)

    return result_norm


def preprocess_plate_variants(plate_bgr: np.ndarray) -> list[np.ndarray]:
    """
    ESTRATEGIA OCR OPTIMIZADA: 7 variantes estratégicas mejoradas.
    
    Cada variante cubre un caso específico:
    1. CLAHE agresivo → Iluminación no uniforme
    2. Adaptive threshold → Sombras y reflejos fuertes
    3. Sharpen + CLAHE → Placas borrosas/movimiento
    4. Upscale x2.5 + denoising → Placas pequeñas o borrosas
    5. Bilateral + CLAHE → Ruido extremo
    6. Otsu threshold → Alto contraste
    7. Morfología + CLAHE → Caracteres débiles
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return []
    h, w = plate_bgr.shape[:2]
    if h < 12 or w < 40:
        return []

    # Escalar a tamaño óptimo (500px ancho - balance velocidad/precisión)
    scale = 500.0 / w
    new_h = max(int(h * scale), 50)
    img_orig = cv2.resize(plate_bgr, (500, new_h), interpolation=cv2.INTER_CUBIC)

    # Aplicar supresión de reflejos
    img = _remove_glare(img_orig)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variants = []
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))  # Más agresivo

    # ═══════════════════════════════════════════════════════════════════════
    # VARIANTE 1: CLAHE agresivo + denoising
    # ═══════════════════════════════════════════════════════════════════════
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    v1 = clahe.apply(denoised)
    variants.append(v1)

    # ═══════════════════════════════════════════════════════════════════════
    # VARIANTE 2: Adaptive threshold Gaussian
    # ═══════════════════════════════════════════════════════════════════════
    v2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 21, 11)
    if _is_dark_background(v2):
        v2 = cv2.bitwise_not(v2)
    variants.append(v2)

    # ═══════════════════════════════════════════════════════════════════════
    # VARIANTE 3: Sharpen agresivo + CLAHE
    # ═══════════════════════════════════════════════════════════════════════
    kernel_sharpen = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
    sharp = cv2.filter2D(gray, -1, kernel_sharpen)
    v3 = clahe.apply(sharp)
    variants.append(v3)

    # ═══════════════════════════════════════════════════════════════════════
    # VARIANTE 4: Upscale x2 + denoising (para placas difíciles - FASE 2)
    # ═══════════════════════════════════════════════════════════════════════
    upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, 
                         interpolation=cv2.INTER_CUBIC)
    denoised_up = cv2.fastNlMeansDenoising(upscaled, None, 10, 7, 21)
    v4 = clahe.apply(denoised_up)
    variants.append(v4)

    # ═══════════════════════════════════════════════════════════════════════
    # VARIANTE 5: Bilateral filter + CLAHE
    # ═══════════════════════════════════════════════════════════════════════
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    v5 = clahe.apply(bilateral)
    variants.append(v5)

    # ═══════════════════════════════════════════════════════════════════════
    # VARIANTE 6: Otsu threshold (binarización automática)
    # ═══════════════════════════════════════════════════════════════════════
    _, v6 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if _is_dark_background(v6):
        v6 = cv2.bitwise_not(v6)
    variants.append(v6)

    # ═══════════════════════════════════════════════════════════════════════
    # VARIANTE 7: Morfología (closing) + CLAHE
    # ═══════════════════════════════════════════════════════════════════════
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    v7 = clahe.apply(morph)
    variants.append(v7)

    return variants  # 7 variantes


def preprocess_plate(plate_bgr: np.ndarray) -> np.ndarray | None:
    v = preprocess_plate_variants(plate_bgr)
    return v[0] if v else None


# ============================================================
# TROCR — Transformer OCR (Microsoft)
# Modelo: microsoft/trocr-small-printed  (ligero, ~330MB)
# Alternativa más precisa: microsoft/trocr-base-printed (~900MB)
# Carga lazy en hilo separado para no bloquear el arranque.
# ============================================================

class TrOCRRecognizer:
    """
    Wrapper para TrOCR (Vision Encoder-Decoder Transformer).
    Especializado en texto impreso — ideal para placas vehiculares.
    Se carga en background y se activa automáticamente cuando está listo.
    """
    # trocr-base-printed: compatible con sentencepiece, mejor precisión
    MODEL_ID = "microsoft/trocr-base-printed"

    def __init__(self):
        self._processor = None
        self._model     = None
        self._device    = GPU_DEVICE
        self._ready     = False
        self._loading   = False
        self._lock      = threading.Lock()

    def load_async(self):
        """Inicia la carga del modelo en un hilo de fondo."""
        if self._loading or self._ready:
            return
        self._loading = True
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        if not TROCR_AVAILABLE:
            print("[TrOCR] 'transformers' no instalado — pip install transformers")
            self._loading = False
            return
        try:
            print(f"[TrOCR] Cargando {self.MODEL_ID} en {self._device}...")
            proc  = TrOCRProcessor.from_pretrained(self.MODEL_ID)
            model = VisionEncoderDecoderModel.from_pretrained(self.MODEL_ID)
            model.to(self._device)
            model.eval()
            with self._lock:
                self._processor = proc
                self._model     = model
                self._ready     = True
            self._loading = False   # FIX: resetear flag para permitir reintentos
            print(f"[TrOCR] Listo en {self._device}")
        except Exception as e:
            print(f"[TrOCR] Error cargando modelo: {e}")
            self._loading = False

    @property
    def available(self) -> bool:
        return self._ready

    def predict(self, plate_bgr: np.ndarray) -> tuple[str | None, float]:
        """
        Reconoce texto en la imagen de placa usando TrOCR.
        Retorna (texto_limpio, confianza_estimada) o (None, 0.0).
        La confianza se estima por la longitud del texto generado
        y si coincide con un formato MX válido.
        """
        if not self._ready:
            return None, 0.0
        try:
            # TrOCR espera imagen RGB PIL
            rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            with self._lock:
                pixel_values = self._processor(
                    images=pil, return_tensors="pt"
                ).pixel_values.to(self._device)

                with torch.no_grad():
                    generated_ids = self._model.generate(
                        pixel_values,
                        max_new_tokens=12,
                        num_beams=4,          # beam search para mayor precisión
                        early_stopping=True,
                    )

                text_raw = self._processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

            # Limpiar: solo alfanuméricos, mayúsculas
            clean = "".join(c for c in text_raw.upper() if c.isalnum())
            if len(clean) < 4:
                return None, 0.0

            clean = correct_mx_plate(clean)

            # Estimar confianza: longitud correcta + formato MX válido
            valid, _ = validate_mx_plate(clean)
            base_conf = 0.70 if valid else 0.45
            # Bonus por longitud óptima (5-7 chars para placas MX)
            if 5 <= len(clean) <= 7:
                base_conf += 0.10

            return clean, min(1.0, base_conf)

        except Exception as e:
            print(f"[TrOCR] Error en predict: {e}")
            return None, 0.0


# Instancia global — carga lazy
_trocr_recognizer: TrOCRRecognizer | None = None

def get_trocr_recognizer() -> TrOCRRecognizer:
    global _trocr_recognizer
    if _trocr_recognizer is None:
        _trocr_recognizer = TrOCRRecognizer()
    return _trocr_recognizer


# ============================================================
# PADDLEOCR — Motor 4
# Alta precisión, corrección de ángulo automática.
# pip install paddlepaddle paddleocr
# ============================================================

class PaddleOCRRecognizer:
    """
    Wrapper para PaddleOCR.
    Ventajas sobre EasyOCR:
      - Corrección de ángulo automática (use_angle_cls=True)
      - Más preciso en texto pequeño y con ruido
      - Más rápido en CPU que EasyOCR
    Carga lazy en background para no bloquear el arranque.
    """
    def __init__(self):
        self._ocr    = None
        self._ready  = False
        self._loading = False
        self._lock   = threading.Lock()

    def load_async(self):
        if self._loading or self._ready:
            return
        self._loading = True
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        if not PADDLE_AVAILABLE:
            print("[PaddleOCR] No instalado — pip install paddlepaddle paddleocr")
            self._loading = False
            return
        try:
            print("[PaddleOCR] Cargando...")
            try:
                ocr = _PaddleOCR(
                    use_textline_orientation=True,  # API PaddleOCR 2.8+
                    lang="en",
                    use_gpu=GPU_AVAILABLE,
                )
            except TypeError:
                # Fallback para versiones anteriores de PaddleOCR
                ocr = _PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=GPU_AVAILABLE,
                )
            with self._lock:
                self._ocr   = ocr
                self._ready = True
            self._loading = False   # FIX: resetear flag para permitir reintentos
            print(f"[PaddleOCR] Listo en {'GPU' if GPU_AVAILABLE else 'CPU'}")
        except Exception as e:
            print(f"[PaddleOCR] Error cargando: {e}")
            self._loading = False

    @property
    def available(self) -> bool:
        return self._ready

    def predict(self, plate_bgr: np.ndarray) -> tuple[str | None, float]:
        """
        Corre PaddleOCR sobre el crop de la placa.
        Prueba la imagen original y la imagen con glare removido,
        se queda con el resultado de mayor confianza.
        Retorna (texto_limpio, confianza) o (None, 0.0).
        """
        if not self._ready:
            return None, 0.0

        best_text, best_conf = None, 0.0

        # Probar con imagen original y con glare removido
        candidates_imgs = [plate_bgr, _remove_glare(plate_bgr)]

        for img in candidates_imgs:
            try:
                # PaddleOCR acepta numpy BGR directamente
                with self._lock:
                    res = self._ocr.ocr(img, cls=True)

                if not res or not res[0]:
                    continue

                # Recopilar todos los fragmentos de texto de la imagen
                texts, confs = [], []
                for line in res[0]:
                    if line is None:
                        continue
                    txt  = line[1][0]          # texto reconocido
                    conf = float(line[1][1])   # confianza [0,1]
                    clean = "".join(c for c in txt.upper() if c.isalnum())
                    if clean and conf > 0.3:
                        texts.append(clean)
                        confs.append(conf)

                if not texts:
                    continue

                combined  = "".join(texts)
                avg_conf  = sum(confs) / len(confs)

                if len(combined) >= 4 and avg_conf > best_conf:
                    best_conf = avg_conf
                    best_text = correct_mx_plate(combined)

            except Exception as e:
                print(f"[PaddleOCR] Error en predict: {e}")
                continue

        return best_text, best_conf


_paddle_recognizer: PaddleOCRRecognizer | None = None

def get_paddle_recognizer() -> PaddleOCRRecognizer:
    global _paddle_recognizer
    if _paddle_recognizer is None:
        _paddle_recognizer = PaddleOCRRecognizer()
    return _paddle_recognizer


def run_ocr(reader, plate_bgr: np.ndarray,
            min_conf: float = 0.30) -> tuple[str | None, float]:
    """
    Pipeline OCR OPTIMIZADO con prioridad a CharCNN:
    
    ESTRATEGIA (OPTIMIZADA):
      1. CharCNN (motor principal) → 30ms, especializado en placas MX
      2. EasyOCR (fallback) → solo si CharCNN falla (conf < 0.30)
         - Usa solo 3 variantes (no 17): CLAHE, Adaptive, Sharpen
         - Tiempo fallback: ~150ms (3 × 50ms)
    
    GANANCIA:
      - Caso típico (CharCNN exitoso): 850ms → 30ms = 28x más rápido
      - Caso fallback: 850ms → 180ms = 4.7x más rápido
      - VRAM liberada: 73% menos (sin TrOCR/PaddleOCR por defecto)
    
    Motores pesados (TrOCR, PaddleOCR) solo si use_heavy_ocr=True.
    """
    # ══════════════════════════════════════════════════════════════════════════
    # MOTOR PRINCIPAL: CharCNN (rápido, especializado, 30ms)
    # ══════════════════════════════════════════════════════════════════════════
    char_rec = get_char_recognizer()
    if char_rec.available:
        try:
            cnn_text, cnn_conf = char_rec.predict_plate(plate_bgr)
        except Exception as e:
            print(f"[OCR DEBUG] CharCNN ERROR: {e}")
            cnn_text, cnn_conf = None, 0.0
        
        # DEBUG: Imprimir resultado CharCNN SIEMPRE
        print(f"[OCR DEBUG] CharCNN: '{cnn_text}' (conf={cnn_conf:.2f})" if cnn_text else "[OCR DEBUG] CharCNN: None")
        
        # UMBRAL MÁS CONSERVADOR: 0.50 (antes 0.30)
        # Razón: CharCNN aún está aprendiendo, mejor usar EasyOCR
        # cuando CharCNN no está muy seguro
        if cnn_text and cnn_conf >= 0.50:
            # Validar formato MX
            valid, _ = validate_mx_plate(cnn_text)
            if valid:
                print(f"[OCR DEBUG] CharCNN ACEPTADO (válido)")
                return cnn_text, cnn_conf
            # Si no es válido pero confianza >0.60, intentar corrección forzada
            elif cnn_conf >= 0.60:
                corrected = correct_mx_plate(cnn_text)
                valid_corr, _ = validate_mx_plate(corrected)
                if valid_corr:
                    print(f"[OCR DEBUG] CharCNN ACEPTADO (corregido: '{corrected}')")
                    return corrected, cnn_conf * 0.95
        
        if cnn_text:
            print(f"[OCR DEBUG] CharCNN rechazado (conf={cnn_conf:.2f} < 0.50 o formato inválido), usando fallback...")
    else:
        print("[OCR DEBUG] CharCNN no disponible")
        cnn_text, cnn_conf = None, 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # FALLBACK: EasyOCR con estrategia progresiva (rápido → preciso)
    # ══════════════════════════════════════════════════════════════════════════
    all_variants = preprocess_plate_variants(plate_bgr)
    
    # ESTRATEGIA: Probar primeras 3 variantes rápidas
    # Si no obtienen resultado válido, probar las 4 restantes
    fast_variants = all_variants[:3]  # CLAHE, Adaptive, Sharpen (más comunes)
    slow_variants = all_variants[3:]  # Upscale, Bilateral, Otsu, Morph (casos especiales)
    
    variant_results: dict[str, list[float]] = {}  # texto -> [confianzas]
    
    # FASE 1: Variantes rápidas (~150-200ms)
    for img in fast_variants:
        try:
            results = reader.readtext(
                img,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                detail=1,
                width_ths=0.6,
                text_threshold=0.35,
                low_text=0.25,
                mag_ratio=1.5,
            )
        except Exception:
            continue
        
        if not results:
            continue
        
        # Ordenar por posición X (izquierda a derecha)
        results_sorted = sorted(results, key=lambda r: r[0][0][0])
        texts, confs = [], []
        
        for (_, t, c) in results_sorted:
            clean = "".join(ch for ch in t.upper() if ch.isalnum())
            if clean:
                texts.append(clean)
                confs.append(c)
        
        if not texts:
            continue
        
        combined = "".join(texts)
        avg_conf = sum(confs) / len(confs)
        
        if len(combined) >= 4 and avg_conf >= 0.20:  # Umbral bajo para fallback
            corrected = correct_mx_plate(combined)
            
            if corrected not in variant_results:
                variant_results[corrected] = []
            variant_results[corrected].append(avg_conf)
    
    # Seleccionar mejor resultado EasyOCR (FASE 1)
    ocr_text, ocr_conf = None, 0.0
    if variant_results:
        best_text = None
        best_score = 0.0
        
        for text, confs in variant_results.items():
            # Score = promedio × boost por repetición
            avg = sum(confs) / len(confs)
            repetition_boost = min(1.0 + (len(confs) - 1) * 0.1, 1.5)
            score = avg * repetition_boost
            
            if score > best_score:
                best_score = score
                best_text = text
        
        if best_text:
            ocr_text = best_text
            ocr_conf = min(1.0, best_score)
            print(f"[OCR DEBUG] EasyOCR (FASE 1): '{ocr_text}' (conf={ocr_conf:.2f}, {len(variant_results[ocr_text])} variantes)")
    
    # ══════════════════════════════════════════════════════════════════════════
    # FASE 2: Variantes lentas SOLO si FASE 1 falló o dio baja confianza
    # ══════════════════════════════════════════════════════════════════════════
    if (not ocr_text or ocr_conf < 0.40) and slow_variants:
        print("[OCR DEBUG] FASE 1 insuficiente, probando variantes lentas...")
        
        for img in slow_variants:
            try:
                results = reader.readtext(
                    img,
                    paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    detail=1,
                    width_ths=0.6,
                    text_threshold=0.35,
                    low_text=0.25,
                    mag_ratio=1.5,
                )
            except Exception:
                continue
            
            if not results:
                continue
            
            # Ordenar por posición X
            results_sorted = sorted(results, key=lambda r: r[0][0][0])
            texts, confs = [], []
            
            for (_, t, c) in results_sorted:
                clean = "".join(ch for ch in t.upper() if ch.isalnum())
                if clean:
                    texts.append(clean)
                    confs.append(c)
            
            if not texts:
                continue
            
            combined = "".join(texts)
            avg_conf = sum(confs) / len(confs)
            
            if len(combined) >= 4 and avg_conf >= 0.20:
                corrected = correct_mx_plate(combined)
                
                if corrected not in variant_results:
                    variant_results[corrected] = []
                variant_results[corrected].append(avg_conf)
        
        # Re-evaluar mejor resultado con variantes lentas incluidas
        if variant_results:
            best_text = None
            best_score = 0.0
            
            for text, confs in variant_results.items():
                avg = sum(confs) / len(confs)
                repetition_boost = min(1.0 + (len(confs) - 1) * 0.1, 1.5)
                score = avg * repetition_boost
                
                if score > best_score:
                    best_score = score
                    best_text = text
            
            if best_text:
                ocr_text = best_text
                ocr_conf = min(1.0, best_score)
                print(f"[OCR DEBUG] EasyOCR (FASE 2): '{ocr_text}' (conf={ocr_conf:.2f}, {len(variant_results[ocr_text])} variantes)")

    # ══════════════════════════════════════════════════════════════════════════
    # MOTORES PESADOS: TrOCR + PaddleOCR (solo si use_heavy_ocr=True)
    # ══════════════════════════════════════════════════════════════════════════
    use_heavy = config.get("use_heavy_ocr", False)
    trocr_text,  trocr_conf  = (get_trocr_recognizer().predict(plate_bgr)
                                 if use_heavy else (None, 0.0))
    paddle_text, paddle_conf = (get_paddle_recognizer().predict(plate_bgr)
                                 if use_heavy else (None, 0.0))

    # ══════════════════════════════════════════════════════════════════════════
    # FUSIÓN: Combinar resultados con ponderación y consenso
    # ══════════════════════════════════════════════════════════════════════════
    def _mx_bonus(text: str | None) -> float:
        """Bonus por formato MX válido."""
        if not text:
            return 0.0
        valid, _ = validate_mx_plate(text)
        return 0.10 if valid else 0.0

    candidates: list[tuple[str, float]] = []
    if cnn_text:
        candidates.append((cnn_text,    cnn_conf    + _mx_bonus(cnn_text)))
    if ocr_text:
        candidates.append((ocr_text,    ocr_conf    + _mx_bonus(ocr_text)))
    if trocr_text:
        candidates.append((trocr_text,  trocr_conf  + _mx_bonus(trocr_text)))
    if paddle_text:
        candidates.append((paddle_text, paddle_conf + _mx_bonus(paddle_text)))

    # DEBUG: Mostrar todos los candidatos
    print(f"[OCR DEBUG] Candidatos: {[(t, f'{c:.2f}') for t, c in candidates]}")

    if not candidates:
        return None, 0.0

    # Boost por consenso entre motores (lecturas similares)
    boosted: dict[str, float] = {}
    for i, (ta, sa) in enumerate(candidates):
        score = sa
        # Boost si otro motor lee algo similar (distancia de edición ≤ 1)
        for j, (tb, _) in enumerate(candidates):
            if i != j and _edit_distance(ta, tb) <= 1:
                score += 0.15
                break
        boosted[ta] = max(boosted.get(ta, 0.0), score)

    best_text  = max(boosted, key=boosted.__getitem__)
    best_score = boosted[best_text]
    final_conf = min(1.0, best_score)

    # ══════════════════════════════════════════════════════════════════════════
    # SISTEMA DE APRENDIZAJE OCR: Aplicar correcciones aprendidas
    # ══════════════════════════════════════════════════════════════════════════
    if OCR_LEARNING_AVAILABLE and best_text:
        try:
            ocr_learning = get_ocr_learning()
            corrected_text, correction_conf = ocr_learning.correct(best_text)
            
            # CRÍTICO: Solo aplicar correcciones EXACTAS (conf >= 0.99)
            # NO aplicar patrones de caracteres a placas diferentes
            if correction_conf >= 0.99 and corrected_text != best_text:
                print(f"[OCR Learning] Aplicando corrección exacta: '{best_text}' → '{corrected_text}' (conf={correction_conf:.2f})")
                best_text = corrected_text
                # Ajustar confianza: promedio entre la original y la de corrección
                final_conf = min(1.0, (final_conf + correction_conf) / 2.0)
        except Exception as e:
            print(f"[OCR Learning] Error: {e}")

    print(f"[OCR DEBUG] FINAL: '{best_text}' (conf={final_conf:.2f})")
    return best_text, final_conf



def _edit_distance(a: str, b: str) -> int:
    """Distancia de Levenshtein entre dos strings (para agrupar lecturas similares)."""
    if abs(len(a) - len(b)) > 2:
        return 99
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev[j-1] + cost)
    return dp[n]


# ============================================================
# TRACKER SIMPLE (Kalman + IoU) — sin dependencias externas
# Reemplaza DeepSORT para evitar conflictos en Windows.
# Cada vehiculo/placa detectada recibe un ID estable entre frames.
# ============================================================

def _iou(a: tuple, b: tuple) -> float:
    """Intersection over Union entre dos bboxes (x1,y1,x2,y2)."""
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1) * max(0,iy2-iy1)
    if inter == 0: return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


class KalmanBox:
    """
    Filtro de Kalman 1D por coordenada para suavizar bboxes.
    Estado: [x1,y1,x2,y2, vx1,vy1,vx2,vy2]
    """
    def __init__(self, bbox: tuple):
        x1,y1,x2,y2 = bbox
        self.x = np.array([x1,y1,x2,y2, 0.,0.,0.,0.], dtype=float)
        # Covarianza de proceso y medicion
        self.P = np.eye(8) * 10.0
        self.Q = np.eye(8) * 1.0   # ruido proceso
        self.R = np.eye(4) * 5.0   # ruido medicion
        # Matriz de transicion (velocidad constante)
        self.F = np.eye(8)
        for i in range(4): self.F[i, i+4] = 1.0
        # Matriz de observacion
        self.H = np.zeros((4,8)); np.fill_diagonal(self.H, 1.0)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return tuple(self.x[:4].astype(int))

    def update(self, bbox: tuple):
        z = np.array(bbox, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def bbox(self) -> tuple:
        return tuple(self.x[:4].astype(int))


class SimpleTracker:
    """
    Tracker multi-objeto basado en Kalman + asignacion greedy por IoU.
    Asigna IDs estables a vehiculos y placas entre frames consecutivos.
    """
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10):
        self.iou_threshold = iou_threshold
        self.max_age       = max_age          # frames sin match antes de eliminar
        self._next_id      = 1
        self._tracks: dict[int, dict] = {}    # id -> {kalman, age, plate_text}

    def update(self, detections: list[tuple]) -> list[tuple]:
        """
        detections: lista de (x1,y1,x2,y2)
        Retorna: lista de (x1,y1,x2,y2, track_id)
        """
        # Predecir posicion de cada track existente
        for tid, tr in self._tracks.items():
            tr["pred"] = tr["kalman"].predict()
            tr["matched"] = False

        results = []
        unmatched_dets = list(range(len(detections)))

        # Asignacion greedy: mayor IoU primero
        if self._tracks and detections:
            track_ids = list(self._tracks.keys())
            iou_matrix = np.zeros((len(track_ids), len(detections)))
            for i, tid in enumerate(track_ids):
                for j, det in enumerate(detections):
                    iou_matrix[i,j] = _iou(self._tracks[tid]["pred"], det)

            matched_dets = set()
            # Ordenar por IoU descendente
            pairs = sorted(
                [(i,j) for i in range(len(track_ids)) for j in range(len(detections))],
                key=lambda p: iou_matrix[p[0],p[1]], reverse=True
            )
            matched_tracks = set()
            for i, j in pairs:
                if iou_matrix[i,j] < self.iou_threshold: break
                if i in matched_tracks or j in matched_dets: continue
                tid = track_ids[i]
                self._tracks[tid]["kalman"].update(detections[j])
                self._tracks[tid]["matched"] = True
                self._tracks[tid]["age"] = 0
                matched_tracks.add(i); matched_dets.add(j)
                results.append((*detections[j], tid))
            unmatched_dets = [j for j in range(len(detections)) if j not in matched_dets]

        # Nuevos tracks para detecciones sin match
        for j in unmatched_dets:
            tid = self._next_id; self._next_id += 1
            self._tracks[tid] = {
                "kalman":  KalmanBox(detections[j]),
                "age":     0,
                "matched": True,
                "plate_text": None,
            }
            results.append((*detections[j], tid))

        # Envejecer y eliminar tracks perdidos
        dead = []
        for tid, tr in self._tracks.items():
            if not tr["matched"]:
                tr["age"] += 1
                if tr["age"] > self.max_age:
                    dead.append(tid)
        for tid in dead:
            del self._tracks[tid]

        return results

    def set_plate(self, track_id: int, text: str):
        if track_id in self._tracks:
            self._tracks[track_id]["plate_text"] = text

    def get_plate(self, track_id: int) -> str | None:
        return self._tracks.get(track_id, {}).get("plate_text")
    
    def set_parent_vehicle(self, track_id: int, parent_id: int):
        """Asocia una placa con su vehículo padre"""
        if track_id in self._tracks:
            self._tracks[track_id]["parent_vehicle_id"] = parent_id
    
    def get_parent_vehicle(self, track_id: int) -> int | None:
        """Obtiene el ID del vehículo padre de una placa"""
        return self._tracks.get(track_id, {}).get("parent_vehicle_id")


# ============================================================
# CACHE DE OCR (evita reprocesar la misma imagen)
# Clave: hash perceptual del crop de la placa
# ============================================================

class OCRCache:
    """LRU cache para resultados OCR. Evita reprocesar crops identicos."""
    def __init__(self, maxsize: int = 256):
        self._cache: dict[str, tuple[str, float]] = {}
        self._order: deque = deque()
        self._maxsize = maxsize

    def _phash(self, img: np.ndarray) -> str:
        """Hash perceptual: resize 16x8 (128 bits) — menos colisiones que 8x8."""
        small = cv2.resize(img, (16, 8), interpolation=cv2.INTER_AREA)
        if len(small.shape) == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mean = small.mean()
        bits = (small > mean).flatten()
        return "".join("1" if b else "0" for b in bits)

    def get(self, img: np.ndarray) -> tuple[str, float] | None:
        key = self._phash(img)
        return self._cache.get(key)

    def put(self, img: np.ndarray, result: tuple[str, float]):
        key = self._phash(img)
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self._maxsize:
            old = self._order.popleft()
            self._cache.pop(old, None)
        self._cache[key] = result
        self._order.append(key)


_ocr_cache = OCRCache(maxsize=512)


# ============================================================
# CORRECCIÓN DE PERSPECTIVA
# Endereza la placa si YOLO devuelve un bbox rotado (obb).
# Si el modelo no soporta OBB, hace un crop rectangular con padding.
# ============================================================

def _perspective_correct(orig: np.ndarray, box,
                          rx1: int, ry1: int) -> np.ndarray | None:
    """
    Intenta extraer la placa con corrección de perspectiva.

    Si el modelo devuelve OBB (Oriented Bounding Box, box.xyxyxyxy),
    aplica warpPerspective para enderezar la placa.
    Si no, devuelve None y el caller usa crop rectangular.

    Parámetros:
      orig  — frame completo BGR
      box   — objeto box de YOLO (puede tener .xyxyxyxy para OBB)
      rx1, ry1 — offset del ROI dentro del frame completo
    """
    h_img, w_img = orig.shape[:2]

    try:
        # ── Caso 1: OBB disponible (modelo entrenado con rotación) ───────────
        if hasattr(box, "xyxyxyxy") and box.xyxyxyxy is not None:
            pts = box.xyxyxyxy[0].cpu().numpy().reshape(4, 2)
            # Trasladar al sistema de coordenadas del frame completo
            pts[:, 0] += rx1
            pts[:, 1] += ry1
            pts = np.clip(pts, 0, [w_img-1, h_img-1]).astype(np.float32)

            # Ordenar puntos: top-left, top-right, bottom-right, bottom-left
            rect = _order_points(pts)
            tl, tr, br, bl = rect

            # Calcular dimensiones de salida
            width  = max(int(np.linalg.norm(br - bl)),
                         int(np.linalg.norm(tr - tl)))
            height = max(int(np.linalg.norm(tr - br)),
                         int(np.linalg.norm(tl - bl)))

            if width < 40 or height < 10:
                return None

            dst = np.array([
                [0,         0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0,         height - 1],
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(orig, M, (width, height),
                                         flags=cv2.INTER_CUBIC)
            return warped

        # ── Caso 2: bbox rectangular — añadir padding ─────────────────────
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 += rx1; y1 += ry1; x2 += rx1; y2 += ry1
        pw, ph = x2 - x1, y2 - y1
        pad_y = max(3, int(ph * 0.10))
        pad_x = max(3, int(pw * 0.06))
        cy1 = max(0, y1 - pad_y)
        cy2 = min(h_img, y2 + pad_y)
        cx1 = max(0, x1 - pad_x)
        cx2 = min(w_img, x2 + pad_x)
        crop = orig[cy1:cy2, cx1:cx2]
        return crop if crop.size > 0 else None

    except Exception:
        return None


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Ordena 4 puntos como: top-left, top-right, bottom-right, bottom-left.
    Necesario para getPerspectiveTransform.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left: menor suma
    rect[2] = pts[np.argmax(s)]     # bottom-right: mayor suma
    rect[1] = pts[np.argmin(diff)]  # top-right: menor diferencia
    rect[3] = pts[np.argmax(diff)]  # bottom-left: mayor diferencia
    return rect


def correct_plate_perspective(crop: np.ndarray) -> np.ndarray | None:
    """
    Detecta y corrige la perspectiva de una placa en ángulo.
    
    Algoritmo:
    1. Detectar bordes con Canny
    2. Encontrar contorno rectangular más grande
    3. Si el contorno está rotado, aplicar warpPerspective
    4. Retornar crop corregido con aspect ratio 3:1 (estándar MX)
    
    Retorna None si no se puede corregir o no es necesario.
    
    OPTIMIZADO PARA PLACAS EN ÁNGULO:
    - Detecta placas rotadas hasta 45 grados
    - Corrige perspectiva automáticamente
    - Mejora OCR en placas no frontales
    """
    if crop is None or crop.size == 0:
        return None
    
    h, w = crop.shape[:2]
    
    # Validar tamaño mínimo
    if w < 40 or h < 10:
        return None
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    
    # Aplicar blur para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordes
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilatar para conectar bordes fragmentados
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Buscar el contorno rectangular más grande
    best_contour = None
    best_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < (w * h * 0.3):  # Al menos 30% del área total
            continue
        
        # Aproximar a polígono
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Buscar cuadriláteros (4 puntos)
        if len(approx) == 4 and area > best_area:
            best_contour = approx
            best_area = area
    
    if best_contour is None:
        return None
    
    # Obtener los 4 puntos del contorno
    pts = best_contour.reshape(4, 2).astype(np.float32)
    
    # Ordenar puntos: TL, TR, BR, BL
    # Calcular centro
    center = pts.mean(axis=0)
    
    # Ordenar por ángulo desde el centro
    def angle_from_center(pt):
        return np.arctan2(pt[1] - center[1], pt[0] - center[0])
    
    sorted_pts = sorted(pts, key=angle_from_center)
    
    # Identificar TL (top-left): menor suma x+y
    sums = [pt[0] + pt[1] for pt in sorted_pts]
    tl_idx = sums.index(min(sums))
    
    # Rotar para que TL sea el primero
    ordered_pts = sorted_pts[tl_idx:] + sorted_pts[:tl_idx]
    src_pts = np.array(ordered_pts, dtype=np.float32)
    
    # Calcular dimensiones del rectángulo destino
    # Usar aspect ratio 3:1 (estándar para placas mexicanas)
    # Mantener el área similar al crop original
    target_area = w * h
    target_height = int(np.sqrt(target_area / 3.0))
    target_width = target_height * 3
    
    # Limitar tamaño máximo
    if target_width > 600:
        target_width = 600
        target_height = 200
    
    # Validar dimensiones mínimas
    if target_width < 40 or target_height < 10:
        return None
    
    # Puntos de destino (rectángulo perfecto)
    dst_pts = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype=np.float32)
    
    # Calcular matriz de transformación
    try:
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Aplicar warp
        warped = cv2.warpPerspective(crop, matrix, (target_width, target_height))
        
        # Verificar que el resultado es válido
        if warped.size == 0 or warped.mean() < 10:
            return None
        
        return warped
    
    except Exception:
        return None


class ProcessingThread(threading.Thread):
    def __init__(self, frame_queue: queue.Queue, result_queue: queue.Queue,
                 model_veh, model_plate, reader, app_ref):
        super().__init__(daemon=True)
        self.frame_queue  = frame_queue
        self.result_queue = result_queue
        self.model_veh    = model_veh
        self.model_plate  = model_plate
        self.reader       = reader
        self.app          = app_ref
        self._running     = True
        self._votes:      dict[str, Counter] = {}
        self._last_emit:  dict[str, float]   = {}
        self._last_ocr:   dict[str, float]   = {}
        # Tracker independiente para vehiculos y placas
        self._veh_tracker   = SimpleTracker(iou_threshold=0.3, max_age=3)   # 3 frames (~100ms)
        self._plate_tracker = SimpleTracker(iou_threshold=0.25, max_age=2)  # 2 frames (~67ms)

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            try:
                item = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            orig, disp = item
            try:
                self._process(orig, disp)
            except Exception as e:
                print(f"Error hilo procesamiento: {e}")

    def _process(self, orig: np.ndarray, disp: np.ndarray):
        """
        Procesa un frame para detectar vehículos y placas.
        
        OPTIMIZADO PARA MÚLTIPLES PLACAS SIMULTÁNEAS:
        - Detecta hasta 10 vehículos y 10 placas por frame
        - Cooldowns de OCR reducidos para procesamiento más rápido
        - Umbral de votos optimizado para confirmación rápida
        - Colas aumentadas para manejar más detecciones sin lag
        """
        app = self.app
        h, w = orig.shape[:2]
        t_start = time.time()

        # ── Análisis de nitidez del frame ────────────────────────────────────
        # Varianza del Laplaciano: frames borrosos tienen varianza baja.
        # Se usa en el cooldown adaptativo del OCR (dentro del bucle de placas).
        gray_full = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        frame_sharpness = cv2.Laplacian(gray_full, cv2.CV_64F).var()
        # FIX: usar frame_sharpness para saltar detección YOLO en frames muy borrosos.
        # Frames con varianza < 15 son completamente inutilizables (movimiento extremo).
        # Esto evita que YOLO procese frames negros o completamente desenfocados.
        if frame_sharpness < 15.0:
            result = {
                "display": disp, "vehicle_detected": False,
                "confirmed_plates": [], "orig": orig,
                "n_vehicles": 0, "n_plates": 0,
                "proc_ms": (time.time() - t_start) * 1000,
            }
            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                try: self.result_queue.get_nowait()
                except queue.Empty: pass
                try: self.result_queue.put_nowait(result)
                except queue.Full: pass
            return

        # ── Deteccion de vehiculos ───────────────────────────────────────────
        # GTX 1650: 640px es óptimo — 960px no aporta más detecciones pero sí +30ms
        det_w = min(w, 640)
        det_h = int(h * det_w / w)
        small = cv2.resize(orig, (det_w, det_h))
        sx, sy = w / det_w, h / det_h

        veh_results = self.model_veh(
            small, verbose=False,
            conf=app.conf_threshold_vehicle,
            max_det=10,              # Permitir hasta 10 vehículos simultáneos
            device=GPU_DEVICE,
            half=GPU_AVAILABLE,      # FP16 en GPU: ~2x más rápido en GTX 1650
        )
        raw_veh_boxes = []
        vehicle_detected = False

        for res in veh_results:
            if res.boxes is None: continue
            for box in res.boxes:
                if int(box.cls[0].item()) not in VEHICLE_CLASSES: continue
                vehicle_detected = True
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0,int(x1*sx)); y1 = max(0,int(y1*sy))
                x2 = min(w,int(x2*sx)); y2 = min(h,int(y2*sy))
                raw_veh_boxes.append((x1,y1,x2,y2))

        tracked_vehs = self._veh_tracker.update(raw_veh_boxes)
        vehicle_rois  = []
        vehicle_boxes = []   # datos para update_ui — NO se dibuja aquí
        for (x1,y1,x2,y2,tid) in tracked_vehs:
            vehicle_rois.append((x1,y1,x2,y2))
            vehicle_boxes.append((x1,y1,x2,y2,int(tid)))

        # ── Deteccion de placas ──────────────────────────────────────────────
        # CAMBIO: la placa es la prioridad — siempre buscar en frame completo.
        # Los ROIs de vehículos se usan como zonas de MAYOR CONFIANZA (se procesan
        # primero y con umbral más bajo), pero NO son condicionantes.
        # Si no hay vehículos detectados, el frame completo se procesa igual.
        #
        # Estrategia de regiones:
        #   1. ROIs de vehículos (umbral bajo: conf_plate * 0.7) — alta prioridad
        #   2. Frame completo (umbral normal) — captura placas sin vehículo visible
        #      (motos, placas parcialmente ocultas, ángulos extremos)
        #
        # Esto elimina el problema donde una placa visible no se detectaba
        # porque el modelo de vehículos no encontraba el auto.

        raw_plate_boxes = []
        plate_crops_map: dict[tuple, np.ndarray] = {}
        plate_conf_map:  dict[tuple, float]      = {}
        seen_bboxes: set = set()   # evitar duplicados entre regiones

        def _detect_plates_in_region(rx1, ry1, rx2, ry2, conf_override=None):
            """Detecta placas en una región del frame y agrega al mapa global."""
            roi = orig[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                return
            conf = conf_override if conf_override is not None else app.conf_threshold_plate
            
            # FIX: Reducir confianza para videos (más permisivo)
            # Video to training usa conf * 0.7 para capturar más placas
            if conf_override is None:
                conf = conf * 0.85  # Más permisivo que tiempo real
            
            plate_res = self.model_plate(
                roi, verbose=False,
                conf=conf,
                max_det=10,              # Permitir hasta 10 detecciones (5+ placas)
                device=GPU_DEVICE,
                half=GPU_AVAILABLE,      # FP16: ~2x más rápido en GTX 1650
            )
            for pr in plate_res:
                if pr.boxes is None:
                    continue
                for pb in pr.boxes:
                    px1, py1, px2, py2 = map(int, pb.xyxy[0].tolist())
                    yolo_conf = float(pb.conf[0].item())
                    ax1, ay1 = rx1 + px1, ry1 + py1
                    ax2, ay2 = rx1 + px2, ry1 + py2

                    bbox = (ax1, ay1, ax2, ay2)

                    # Evitar duplicados: si ya hay una bbox con IoU > 0.5, saltar
                    is_dup = False
                    for existing in seen_bboxes:
                        if _iou(bbox, existing) > 0.50:
                            is_dup = True
                            break
                    if is_dup:
                        continue

                    # FIX: Relajar validación de geometría para placas en ángulo
                    is_valid, geo_score = validate_plate_geometry(bbox, orig.shape)
                    if not is_valid and geo_score < 0.3:  # Solo rechazar si es MUY malo
                        continue

                    pw, ph = ax2 - ax1, ay2 - ay1
                    if ph == 0:
                        continue
                    
                    # FIX: Aspect ratio más permisivo para placas en ángulo
                    # Video to training usa 1.5-8.0 para capturar placas rotadas
                    aspect_ratio = pw / ph
                    if not (1.5 <= aspect_ratio <= 8.0):
                        continue

                    crop = _perspective_correct(orig, pb, rx1, ry1)
                    if crop is None:
                        pad_y = max(2, int(ph * 0.10))
                        pad_x = max(2, int(pw * 0.06))
                        cy1 = max(0, ay1 - pad_y)
                        cy2 = min(h,  ay2 + pad_y)
                        cx1 = max(0, ax1 - pad_x)
                        cx2 = min(w,  ax2 + pad_x)
                        crop = orig[cy1:cy2, cx1:cx2]

                    if crop is None or crop.size == 0:
                        continue

                    # ── QUALITY SCORE AVANZADO ────────────────────────────────
                    # Calcular señales adicionales: nitidez, contraste, densidad
                    # de bordes, alineación horizontal, brillo
                    try:
                        quality_score, quality_details = plate_quality_score_advanced(crop)
                    except Exception as e:
                        # Si falla el quality score, usar score neutral
                        quality_score = 0.5
                        quality_details = {}
                    
                    # Combinar geo_score (geometría) con quality_score (imagen)
                    # Geometría: 40%, Calidad de imagen: 60%
                    combined_score = geo_score * 0.4 + quality_score * 0.6
                    
                    # Filtrar placas de muy baja calidad
                    # Umbral: 0.30 (más permisivo para capturar más candidatos)
                    if combined_score < 0.30:
                        continue

                    seen_bboxes.add(bbox)
                    raw_plate_boxes.append(bbox)
                    plate_crops_map[bbox] = crop
                    # Usar combined_score en lugar de solo geo_score
                    plate_conf_map[bbox] = yolo_conf * 0.5 + combined_score * 0.5

        # Paso 1: buscar en ROIs de vehículos con umbral reducido (más sensible)
        veh_conf_boost = max(0.10, app.conf_threshold_plate * 0.60)  # Más permisivo: 0.60 en vez de 0.70
        for (rx1, ry1, rx2, ry2) in vehicle_rois:
            _detect_plates_in_region(rx1, ry1, rx2, ry2, conf_override=veh_conf_boost)

        # FIX CRÍTICO: SIEMPRE buscar en frame completo, no solo si no hay detecciones
        # Esto permite detectar las 10 placas, no solo las que están en ROIs de vehículos
        # Video to training hace esto y por eso detecta mejor
        _detect_plates_in_region(0, 0, w, h, conf_override=app.conf_threshold_plate * 0.85)

        tracked_plates = self._plate_tracker.update(raw_plate_boxes)

        # ═══════════════════════════════════════════════════════════════════════
        # ANCLAJE VEHÍCULO-PLACA: Asociar cada placa con su vehículo más cercano
        # ═══════════════════════════════════════════════════════════════════════
        def _find_parent_vehicle(plate_box: tuple) -> int | None:
            """
            Encuentra el vehículo padre de una placa usando:
            1. IoU (si la placa está dentro del vehículo)
            2. Distancia del centro (si la placa está cerca del vehículo)
            """
            px1, py1, px2, py2 = plate_box[:4]
            plate_cx = (px1 + px2) / 2
            plate_cy = (py1 + py2) / 2
            
            best_vehicle_id = None
            best_score = 0.0
            
            for (vx1, vy1, vx2, vy2, vid) in tracked_vehs:
                # Calcular IoU (placa dentro del vehículo)
                iou = _iou((px1, py1, px2, py2), (vx1, vy1, vx2, vy2))
                
                # Calcular distancia normalizada del centro de la placa al vehículo
                veh_cx = (vx1 + vx2) / 2
                veh_cy = (vy1 + vy2) / 2
                veh_w = vx2 - vx1
                veh_h = vy2 - vy1
                
                # Distancia normalizada (0-1, donde 0 = centro del vehículo)
                dx = abs(plate_cx - veh_cx) / max(veh_w, 1)
                dy = abs(plate_cy - veh_cy) / max(veh_h, 1)
                dist_norm = (dx**2 + dy**2) ** 0.5
                
                # Score combinado: IoU (60%) + proximidad (40%)
                # IoU alto = placa dentro del vehículo (ideal)
                # Distancia baja = placa cerca del vehículo (fallback)
                proximity_score = max(0, 1.0 - dist_norm)
                combined_score = iou * 0.6 + proximity_score * 0.4
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_vehicle_id = vid
            
            # Solo asociar si el score es razonable (>0.2)
            # Esto evita asociaciones erróneas cuando la placa está muy lejos
            return best_vehicle_id if best_score > 0.2 else None
        
        # Asociar cada placa con su vehículo padre
        for (ax1, ay1, ax2, ay2, plate_tid) in tracked_plates:
            parent_vid = _find_parent_vehicle((ax1, ay1, ax2, ay2))
            if parent_vid is not None:
                self._plate_tracker.set_parent_vehicle(plate_tid, parent_vid)

        # NO se dibuja aquí — update_ui lo hace en cada frame para evitar parpadeo

        # ── OCR con cache + voting ───────────────────────────────────────────
        now = time.time()
        confirmed_plates = []

        if app.detection_mode != "solo_deteccion":
            for (ax1,ay1,ax2,ay2,tid) in tracked_plates:
                crop = plate_crops_map.get((ax1,ay1,ax2,ay2))
                if crop is None:
                    # FIX: usar plate_conf_map para elegir el mejor crop
                    # entre los candidatos con IoU suficiente, no solo el de mayor IoU
                    best_iou, best_crop, best_bbox = 0.0, None, None
                    best_score = -1.0
                    for bbox, c in plate_crops_map.items():
                        iou_val = _iou((ax1,ay1,ax2,ay2), bbox)
                        if iou_val > best_iou:
                            best_iou = iou_val
                            # Combinar IoU con score de calidad del crop
                            combined = iou_val * 0.7 + plate_conf_map.get(bbox, 0.0) * 0.3
                            if combined > best_score:
                                best_score = combined
                                best_crop = c
                                best_bbox = bbox
                    if best_crop is None or best_iou < 0.3: continue
                    crop = best_crop

                pid = str(tid)

                # Mostrar texto conocido del track — lo hace update_ui, no aquí
                known = self._plate_tracker.get_plate(tid)

                # ── CORRECCIÓN DE PERSPECTIVA ────────────────────────────────
                # Intentar corregir perspectiva si la placa está en ángulo
                # Esto mejora significativamente el OCR en placas no frontales
                original_crop = crop.copy()
                try:
                    warped_crop = correct_plate_perspective(crop)
                    if warped_crop is not None:
                        # Calcular calidad de ambos crops
                        orig_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        warp_gray = cv2.cvtColor(warped_crop, cv2.COLOR_BGR2GRAY)
                        orig_sharp = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
                        warp_sharp = cv2.Laplacian(warp_gray, cv2.CV_64F).var()
                        
                        # Si la corrección mejora la nitidez en 10%+, usar crop corregido
                        if warp_sharp > orig_sharp * 1.1:
                            crop = warped_crop
                except Exception:
                    pass  # Si falla, usar crop original

                # ── Cooldown adaptativo ──────────────────────────────────────
                # Frame borroso → cooldown largo (no gastar tiempo en OCR malo)
                # Frame nítido + crop grande → cooldown corto
                # OPTIMIZADO para múltiples placas: cooldowns más cortos
                crop_h, crop_w = crop.shape[:2]
                crop_area = crop_h * crop_w
                crop_gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharpness  = cv2.Laplacian(crop_gray, cv2.CV_64F).var()

                if sharpness < 30:
                    ocr_cooldown = 1.5   # muy borroso: esperar más
                elif sharpness < 80:
                    ocr_cooldown = 0.6   # algo borroso: esperar un poco
                elif sharpness >= 150:
                    ocr_cooldown = 0.1   # muy nítido: leer casi cada frame
                elif crop_area > 8000:
                    ocr_cooldown = 0.15  # nítido y grande: leer muy rápido
                else:
                    ocr_cooldown = 0.25  # nítido pero pequeño: leer rápido

                if now - self._last_ocr.get(pid, 0) < ocr_cooldown:
                    continue
                self._last_ocr[pid] = now

                # ── Cache OCR ────────────────────────────────────────────────
                cached = _ocr_cache.get(crop)
                if cached:
                    text, conf = cached
                    
                    # Aplicar correcciones del OCR Learning incluso a resultados cacheados
                    correction_applied = False
                    if OCR_LEARNING_AVAILABLE and text:
                        try:
                            ocr_learning = get_ocr_learning()
                            corrected_text, correction_conf = ocr_learning.correct(text)
                            
                            # CRÍTICO: Solo aplicar correcciones EXACTAS (conf >= 0.99)
                            # NO aplicar patrones de caracteres a placas diferentes
                            if correction_conf >= 0.99 and corrected_text != text:
                                print(f"[OCR Learning] Aplicando corrección exacta a cache: '{text}' → '{corrected_text}'")
                                text = corrected_text
                                # Mantener confianza alta para correcciones exactas
                                conf = min(1.0, (conf + 1.0) / 2.0)
                                correction_applied = True
                                
                                # CRÍTICO: Si hubo corrección exacta, resetear votos
                                if pid in self._votes:
                                    print(f"[OCR Learning] Reseteando votos de plate_id={pid} por corrección exacta")
                                    self._votes[pid].clear()
                        except Exception as e:
                            print(f"[OCR Learning] Error en cache: {e}")
                else:
                    text, conf = run_ocr(self.reader, crop)
                    correction_applied = False
                    if text:
                        _ocr_cache.put(crop, (text, conf))
                        
                        # Aplicar OCR Learning a resultados nuevos
                        if OCR_LEARNING_AVAILABLE:
                            try:
                                ocr_learning = get_ocr_learning()
                                corrected_text, correction_conf = ocr_learning.correct(text)
                                
                                if correction_conf >= 0.99 and corrected_text != text:
                                    print(f"[OCR Learning] Aplicando corrección exacta: '{text}' → '{corrected_text}'")
                                    text = corrected_text
                                    conf = min(1.0, (conf + 1.0) / 2.0)
                                    correction_applied = True
                                    
                                    if pid in self._votes:
                                        self._votes[pid].clear()
                            except Exception as e:
                                print(f"[OCR Learning] Error: {e}")

                if not text:
                    continue

                # ── Voting ponderado por nitidez ─────────────────────────────
                # Lecturas de frames nítidos pesan más que las de frames borrosos
                if pid not in self._votes:
                    self._votes[pid] = Counter()

                vote_key = text
                for existing in list(self._votes[pid].keys()):
                    if _edit_distance(text, existing) <= 1:
                        vote_key = existing
                        break

                # Peso del voto: 1 (borroso) a 3 (muy nítido)
                vote_weight = 1 if sharpness < 80 else (2 if sharpness < 300 else 3)
                self._votes[pid][vote_key] += vote_weight

                best, best_n = self._votes[pid].most_common(1)[0]
                self._plate_tracker.set_plate(tid, best)
                # El texto actualizado se recoge en plate_boxes al final — update_ui lo dibuja

                # ══════════════════════════════════════════════════════════════
                # SISTEMA DE CONFIRMACIÓN INTELIGENTE POR CONFIANZA
                # ══════════════════════════════════════════════════════════════
                # Sistema híbrido de 3 niveles basado en confianza real:
                #
                # NIVEL 1: Confirmación Instantánea (1 voto)
                #   - OCR Learning aplicó corrección EXACTA (conf=1.00)
                #   - O confianza muy alta (>=0.85) + formato válido + frame nítido
                #
                # NIVEL 2: Confirmación Rápida (2 votos)  
                #   - Formato válido + confianza media (>=0.65) + frame nítido
                #
                # NIVEL 3: Confirmación Conservadora (3-5 votos)
                #   - Formato inválido o confianza baja o frame borroso
                
                valid_fmt, _ = validate_mx_plate(best)
                
                # NIVEL 1: Confirmación instantánea
                if correction_applied and valid_fmt:
                    # Corrección exacta de OCR Learning → confirmar inmediatamente
                    votes_needed = 1
                    emission_cooldown = 0.3
                elif conf >= 0.85 and valid_fmt and sharpness >= 80:
                    # Confianza muy alta + formato válido + frame nítido
                    votes_needed = 1
                    emission_cooldown = 0.3
                # NIVEL 2: Confirmación rápida
                elif conf >= 0.65 and valid_fmt and sharpness >= 80:
                    votes_needed = 2
                    emission_cooldown = 0.5
                elif valid_fmt and sharpness >= 80:
                    votes_needed = 2
                    emission_cooldown = 0.8
                # NIVEL 3: Confirmación conservadora
                elif valid_fmt:
                    votes_needed = 3
                    emission_cooldown = 1.0
                else:
                    # Formato inválido → necesita más evidencia
                    votes_needed = 4
                    emission_cooldown = 1.5

                if best_n >= votes_needed and (now - self._last_emit.get(pid,0)) > emission_cooldown:
                    self._last_emit[pid] = now
                    self._votes[pid].clear()
                    response_ms = (time.time() - t_start) * 1000
                    confirmed_plates.append({
                        "text":        best,
                        "crop":        crop.copy(),
                        "conf":        conf,
                        "response_ms": response_ms,
                        "track_id":    tid,
                        "ax1": ax1, "ay1": ay1,
                    })

        # Limpiar tracks viejos del voting
        active_pids = {str(t[4]) for t in tracked_plates}
        
        # CRÍTICO: Limpiar votos de tracks que ya no existen
        # Esto previene que se queden "pegados" a la escena anterior
        dead_pids = [k for k in self._votes.keys() if k not in active_pids]
        for k in dead_pids:
            del self._votes[k]
            if k in self._last_emit:
                del self._last_emit[k]
            if k in self._last_ocr:
                del self._last_ocr[k]
        
        # Limpieza de tracks muy viejos (>10 segundos sin actividad)
        stale = [k for k in self._last_ocr if k not in active_pids
                 and now - self._last_ocr[k] > 10]
        for k in stale:
            self._last_ocr.pop(k,None)
            self._votes.pop(k,None)
            self._last_emit.pop(k,None)

        # ── Construir plate_boxes para update_ui ─────────────────────────────
        # update_ui dibuja estas cajas en CADA frame — no solo en los procesados
        # NUEVO: Incluir parent_vehicle_id para mostrar la asociación visualmente
        plate_boxes = []
        for (ax1,ay1,ax2,ay2,tid) in tracked_plates:
            known = self._plate_tracker.get_plate(tid)
            parent_vid = self._plate_tracker.get_parent_vehicle(tid)
            
            if known:
                registered = is_plate_registered(known)
                box_color  = (0, 180, 0) if registered else (0, 60, 255)
            else:
                box_color  = (0, 0, 255)   # rojo: detectada, sin texto aún
            
            # Formato: (x1, y1, x2, y2, texto, color, parent_vehicle_id)
            plate_boxes.append((ax1, ay1, ax2, ay2, known or "", box_color, parent_vid))

        result = {
            "display":          disp,          # frame limpio, sin cajas
            "vehicle_detected": vehicle_detected,
            "vehicle_boxes":    vehicle_boxes,  # datos para dibujar en update_ui
            "plate_boxes":      plate_boxes,    # datos para dibujar en update_ui
            "confirmed_plates": confirmed_plates,
            "orig":             orig,
            "n_vehicles":       len(tracked_vehs),
            "n_plates":         len(tracked_plates),
            "proc_ms":          (time.time()-t_start)*1000,
        }
        # Nunca bloquear el hilo de procesamiento:
        # si la cola está llena, descartar el resultado más viejo y meter el nuevo
        try:
            self.result_queue.put_nowait(result)
        except queue.Full:
            try:
                self.result_queue.get_nowait()   # descartar el más viejo
            except queue.Empty:
                pass
            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                pass

# ============================================================
# HERRAMIENTA DE VALIDACION MANUAL DE DATOS DE ENTRENAMIENTO
# ============================================================

def _get_bank_info() -> str:
    """Retorna info del banco de muestras para mostrar en UI."""
    bank = os.path.join("char_cache", "sample_bank.npz")
    real = os.path.join("char_cache", "real_chars.npz")
    lines = []
    for path, label in [(bank, "sample_bank"), (real, "real_chars")]:
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True)
                n = len(data["labels"])
                from collections import Counter as _Counter
                cnt = _Counter(data["labels"].tolist())
                lines.append("%s: %d muestras, %d clases" % (label, n, len(cnt)))
            except Exception:
                lines.append("%s: error al leer" % label)
        else:
            lines.append("%s: no existe" % label)
    jpgs = len(list(Path(IMAGE_FOLDER).glob("*.jpg")))
    encs = len(list(Path(IMAGE_FOLDER).glob("*.enc")))
    lines.append("event_images: %d JPG + %d ENC" % (jpgs, encs))
    return "\n".join(lines)


VALIDATION_LOG = os.path.join("char_cache", "validation_log.json")


def _load_validation_log() -> dict:
    """Carga el log de validaciones previas."""
    if os.path.exists(VALIDATION_LOG):
        try:
            with open(VALIDATION_LOG, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"validated": {}, "stats": {"correct": 0, "incorrect": 0, "corrected": 0}}


def _save_validation_log(log: dict):
    """Guarda el log de validaciones."""
    os.makedirs("char_cache", exist_ok=True)
    try:
        with open(VALIDATION_LOG, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("[Validacion] Error guardando log:", e)


class ValidationTool:
    """
    Herramienta de validación manual de datos de entrenamiento.

    Muestra cada imagen de event_images junto con:
      - La predicción OCR actual del modelo (lo que el sistema leyó)
      - El texto del nombre del archivo (ground truth esperado)

    El usuario puede:
      ✓ Correcto   — confirma que la predicción es correcta
      ✗ Incorrecto — marca como error sin especificar corrección
      ✎ Corregir   — escribe el texto correcto manualmente

    Los resultados se guardan en char_cache/validation_log.json
    y pueden usarse para reentrenar con datos validados.
    """

    def __init__(self, parent: tk.Tk, reader, model_plate):
        self.parent      = parent
        self.reader      = reader
        self.model_plate = model_plate
        self.log         = _load_validation_log()
        self._win        = None
        self._images     = []   # lista de (filepath, plate_from_name)
        self._idx        = 0
        self._current_img_bgr = None

    def open(self):
        """Abre la ventana de validación."""
        if self._win and self._win.winfo_exists():
            self._win.lift()
            return

        # Cargar lista de imágenes
        self._load_image_list()
        if not self._images:
            messagebox.showinfo("Validacion",
                "No hay imagenes en event_images/ para validar.\n"
                "Usa el sistema para capturar algunas primero.")
            return

        self._build_ui()
        self._show_current()

    def _load_image_list(self):
        """Carga todas las imágenes disponibles (JPG y ENC)."""
        self._images = []
        folder = Path(IMAGE_FOLDER)
        
        print(f"[DEBUG] Cargando imágenes desde: {folder.absolute()}")

        # JPGs con nombre de placa
        jpg_count = 0
        for f in sorted(folder.glob("*.jpg")):
            parts = f.stem.split("_")
            print(f"[DEBUG] JPG: {f.name} -> partes: {parts} (len={len(parts)})")
            
            # Caso 1: Formato completo con timestamp (YYYYMMDD_HHMMSS_MICROSEC_PLACA.jpg)
            if len(parts) >= 3:
                plate = parts[-1].upper()
                if 3 <= len(plate) <= 9 and plate.replace(" ", "").replace("-", "").isalnum():
                    self._images.append((str(f), plate))
                    jpg_count += 1
                    print(f"[DEBUG]   ✓ Aceptado: placa '{plate}'")
                else:
                    print(f"[DEBUG]   Rechazado: placa '{plate}' no válida")
            
            # Caso 2: Formato simple (TIMESTAMP_PLACA.jpg o similar con 2 partes)
            elif len(parts) == 2:
                plate = parts[-1].upper()
                if 3 <= len(plate) <= 9 and plate.replace(" ", "").replace("-", "").isalnum():
                    self._images.append((str(f), plate))
                    jpg_count += 1
                    print(f"[DEBUG]   ✓ Aceptado (2 partes): placa '{plate}'")
                else:
                    print(f"[DEBUG]   Rechazado: placa '{plate}' no válida")
            
            # Caso 3: Archivo sin guiones bajos - intentar extraer placa del nombre completo
            else:
                # Intentar usar el nombre completo como placa si tiene formato válido
                plate = f.stem.upper()
                if 3 <= len(plate) <= 9 and plate.replace(" ", "").replace("-", "").isalnum():
                    self._images.append((str(f), plate))
                    jpg_count += 1
                    print(f"[DEBUG]   ✓ Aceptado (nombre completo): placa '{plate}'")
                else:
                    print(f"[DEBUG]   Rechazado: nombre '{plate}' no válido como placa")

        # ENCs encriptados
        enc_count = 0
        for f in sorted(folder.glob("*.enc")):
            stem = f.stem  # quita .enc
            parts = stem.split("_")
            if len(parts) >= 3:
                plate = parts[-1].upper()
                if 3 <= len(plate) <= 9 and plate.replace(" ", "").replace("-", "").isalnum():
                    self._images.append((str(f), plate))
                    enc_count += 1
            elif len(parts) == 2:
                plate = parts[-1].upper()
                if 3 <= len(plate) <= 9 and plate.replace(" ", "").replace("-", "").isalnum():
                    self._images.append((str(f), plate))
                    enc_count += 1

        print(f"[DEBUG] Total encontradas: {jpg_count} JPG + {enc_count} ENC = {len(self._images)}")

        # Recargar el log desde disco para obtener las validaciones más recientes
        self.log = _load_validation_log()
        
        # Filtrar las ya validadas si el usuario quiere
        already = set(self.log.get("validated", {}).keys())
        pending = [(p, t) for p, t in self._images
                   if os.path.basename(p) not in already]

        print(f"[DEBUG] Validadas: {len(already)}, Pendientes: {len(pending)}")

        # Mostrar pendientes primero (ordenadas por fecha, más recientes primero), luego ya validadas
        pending_sorted = sorted(pending, key=lambda x: os.path.basename(x[0]), reverse=True)
        validated = [(p, t) for p, t in self._images if os.path.basename(p) in already]
        
        self._images = pending_sorted + validated
        self._pending_count = len(pending)

    def _build_ui(self):
        """Construye la ventana de validación."""
        win = tk.Toplevel(self.parent)
        win.title("Validacion Manual de Datos de Entrenamiento")
        win.geometry("1100x700")
        win.configure(bg="#1e1e2e")
        win.protocol("WM_DELETE_WINDOW", self._on_close)
        self._win = win

        BG, FG = "#1e1e2e", "#cdd6f4"

        # ── Barra de progreso y stats ─────────────────────────────────────────
        top = tk.Frame(win, bg="#181825", pady=6)
        top.pack(fill="x", padx=10, pady=(8, 0))

        self._lbl_progress = tk.Label(top, text="", bg="#181825",
                                      fg="#89b4fa", font=("Consolas", 10))
        self._lbl_progress.pack(side="left", padx=10)

        self._lbl_stats = tk.Label(top, text="", bg="#181825",
                                   fg="#a6e3a1", font=("Consolas", 10))
        self._lbl_stats.pack(side="right", padx=10)

        # ── Imagen ────────────────────────────────────────────────────────────
        img_frame = tk.Frame(win, bg="#181825", width=700, height=450)
        img_frame.pack(side="left", fill="both", expand=True, padx=10, pady=8)
        img_frame.pack_propagate(False)

        self._img_label = tk.Label(img_frame, bg="#181825",
                                   text="Cargando...", fg="#6c7086")
        self._img_label.pack(expand=True, fill="both")

        # ── Panel derecho ─────────────────────────────────────────────────────
        right = tk.Frame(win, bg="#1e1e2e", width=300)
        right.pack(side="right", fill="y", padx=10, pady=8)
        right.pack_propagate(False)

        tk.Label(right, text="NOMBRE DEL ARCHIVO",
                 bg="#1e1e2e", fg="#6c7086",
                 font=("Consolas", 9)).pack(pady=(12, 2))
        self._lbl_filename = tk.Label(right, text="",
                                      bg="#313244", fg="#f9e2af",
                                      font=("Consolas", 14, "bold"),
                                      width=18, relief="flat", pady=6)
        self._lbl_filename.pack(padx=10)

        tk.Label(right, text="PREDICCION DEL MODELO",
                 bg="#1e1e2e", fg="#6c7086",
                 font=("Consolas", 9)).pack(pady=(16, 2))
        self._lbl_prediction = tk.Label(right, text="...",
                                        bg="#313244", fg="#89b4fa",
                                        font=("Consolas", 18, "bold"),
                                        width=18, relief="flat", pady=8)
        self._lbl_prediction.pack(padx=10)

        # Confianza
        self._lbl_conf = tk.Label(right, text="conf: --",
                                  bg="#1e1e2e", fg="#6c7086",
                                  font=("Consolas", 9))
        self._lbl_conf.pack(pady=2)

        # Coincidencia
        self._lbl_match = tk.Label(right, text="",
                                   bg="#1e1e2e",
                                   font=("Consolas", 10, "bold"))
        self._lbl_match.pack(pady=4)

        # ── Botones de acción ─────────────────────────────────────────────────
        btn_frame = tk.Frame(right, bg="#1e1e2e")
        btn_frame.pack(pady=16, fill="x", padx=8)

        tk.Button(btn_frame, text="✓  CORRECTO",
                  bg="#40a02b", fg="white",
                  font=("Consolas", 11, "bold"),
                  relief="flat", pady=8,
                  command=self._mark_correct).pack(fill="x", pady=3)

        tk.Button(btn_frame, text="✗  INCORRECTO",
                  bg="#d20f39", fg="white",
                  font=("Consolas", 11, "bold"),
                  relief="flat", pady=8,
                  command=self._mark_incorrect).pack(fill="x", pady=3)

        tk.Button(btn_frame, text="✎  CORREGIR",
                  bg="#fe640b", fg="white",
                  font=("Consolas", 11, "bold"),
                  relief="flat", pady=8,
                  command=self._mark_correct_with_edit).pack(fill="x", pady=3)

        tk.Button(btn_frame, text="🗑  DESCARTAR",
                  bg="#6c7086", fg="white",
                  font=("Consolas", 11, "bold"),
                  relief="flat", pady=8,
                  command=self._discard_image).pack(fill="x", pady=3)

        # ── Navegación ────────────────────────────────────────────────────────
        nav = tk.Frame(right, bg="#1e1e2e")
        nav.pack(fill="x", padx=8, pady=4)

        tk.Button(nav, text="◀ Anterior",
                  bg="#313244", fg="#cdd6f4",
                  font=("Consolas", 9), relief="flat",
                  command=self._prev).pack(side="left", expand=True, fill="x", padx=2)

        tk.Button(nav, text="Siguiente ▶",
                  bg="#313244", fg="#cdd6f4",
                  font=("Consolas", 9), relief="flat",
                  command=self._next).pack(side="right", expand=True, fill="x", padx=2)

        # ── Agregar imágenes ──────────────────────────────────────────────────
        tk.Button(right, text="➕ Agregar imágenes",
                  bg="#7287fd", fg="white",
                  font=("Consolas", 10, "bold"), relief="flat",
                  command=self._add_images).pack(fill="x", padx=8, pady=4)

        # ── Exportar ──────────────────────────────────────────────────────────
        tk.Button(right, text="Exportar resultados",
                  bg="#313244", fg="#cdd6f4",
                  font=("Consolas", 9), relief="flat",
                  command=self._export).pack(fill="x", padx=8, pady=4)

        # Atajos de teclado
        win.bind("<Return>",    lambda e: self._mark_correct())
        win.bind("<BackSpace>", lambda e: self._mark_incorrect())
        win.bind("<space>",     lambda e: self._mark_correct_with_edit())
        win.bind("<Delete>",    lambda e: self._discard_image())
        win.bind("<Left>",      lambda e: self._prev())
        win.bind("<Right>",     lambda e: self._next())

    def _show_current(self):
        """Muestra la imagen y predicción actual."""
        if not self._images or not (self._win and self._win.winfo_exists()):
            return

        self._idx = max(0, min(self._idx, len(self._images) - 1))
        filepath, plate_name = self._images[self._idx]
        fname = os.path.basename(filepath)

        # ── Actualizar progreso ───────────────────────────────────────────────
        stats = self.log.get("stats", {})
        total = len(self._images)
        done  = len(self.log.get("validated", {}))
        self._lbl_progress.config(
            text="Imagen %d / %d  |  Pendientes: %d" % (
                self._idx + 1, total, self._pending_count))
        self._lbl_stats.config(
            text="✓ %d  ✗ %d  ✎ %d" % (
                stats.get("correct", 0),
                stats.get("incorrect", 0),
                stats.get("corrected", 0)))

        # ── Cargar imagen ─────────────────────────────────────────────────────
        img_bgr = None
        try:
            if filepath.endswith(".enc"):
                img_bgr = load_image_decrypted(filepath)
            else:
                img_bgr = cv2.imread(filepath)
        except Exception:
            pass

        self._current_img_bgr = img_bgr

        if img_bgr is not None:
            # Mostrar imagen escalada con mejor calidad
            h, w = img_bgr.shape[:2]
            max_w, max_h = 680, 430  # Aumentado para mejor visualización
            scale = min(max_w / w, max_h / h, 1.0)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            
            # Usar INTER_AREA para reducir (mejor calidad) o INTER_CUBIC para ampliar
            if scale < 1.0:
                interp = cv2.INTER_AREA  # Mejor para reducir tamaño
            else:
                interp = cv2.INTER_CUBIC  # Mejor para mantener o ampliar
            
            disp = cv2.resize(img_bgr, (nw, nh), interpolation=interp)
            rgb  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            from PIL import Image as _PILImage
            
            # Usar LANCZOS de PIL para mejor calidad final
            pil_img = _PILImage.fromarray(rgb)
            photo = ImageTk.PhotoImage(pil_img)
            self._img_label.config(image=photo, text="")
            self._img_label.image = photo
        else:
            self._img_label.config(image="", text="No se pudo cargar la imagen")

        # ── Nombre del archivo (ground truth) ─────────────────────────────────
        self._lbl_filename.config(text=plate_name)

        # ── Predicción del modelo ─────────────────────────────────────────────
        # Verificar si ya fue validada
        validated = self.log.get("validated", {}).get(fname)
        if validated:
            pred_text = validated.get("prediction", "?")
            status    = validated.get("status", "?")
            corrected = validated.get("corrected_to", "")
            color_map = {"correct": "#40a02b", "incorrect": "#d20f39",
                         "corrected": "#fe640b"}
            self._lbl_prediction.config(
                text=pred_text,
                fg=color_map.get(status, "#89b4fa"))
            self._lbl_conf.config(
                text="Ya validada: %s%s" % (
                    status, (" → " + corrected) if corrected else ""))
            self._lbl_match.config(text="")
        else:
            # Correr OCR en hilo para no bloquear UI
            self._lbl_prediction.config(text="Analizando...", fg="#6c7086")
            self._lbl_conf.config(text="conf: --")
            self._lbl_match.config(text="")
            if img_bgr is not None:
                threading.Thread(
                    target=self._run_ocr_async,
                    args=(img_bgr, plate_name, fname),
                    daemon=True
                ).start()

    def _run_ocr_async(self, img_bgr, plate_name, fname):
        """Corre OCR en background y actualiza la UI."""
        try:
            # Intentar detectar la placa con YOLO primero
            plate_crop = None
            if self.model_plate is not None:
                try:
                    results = self.model_plate(img_bgr, verbose=False, conf=0.15,
                                               device=GPU_DEVICE)
                    for res in results:
                        if res.boxes is None:
                            continue
                        for box in res.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)
                            crop = img_bgr[y1:y2, x1:x2]
                            if crop.size > 0 and (x2 - x1) > 30:
                                plate_crop = crop
                                break
                        if plate_crop is not None:
                            break
                except Exception:
                    pass

            # Si no encontró placa, usar imagen completa
            if plate_crop is None:
                h, w = img_bgr.shape[:2]
                plate_crop = img_bgr[int(h*0.4):int(h*0.85),
                                     int(w*0.1):int(w*0.9)]

            text, conf = run_ocr(self.reader, plate_crop)
            if not text:
                text, conf = "---", 0.0

            # Actualizar UI desde hilo principal
            if self._win and self._win.winfo_exists():
                self._win.after(0, lambda t=text, c=conf, p=plate_name:
                                self._update_prediction(t, c, p))
        except Exception as e:
            if self._win and self._win.winfo_exists():
                self._win.after(0, lambda: self._lbl_prediction.config(
                    text="Error OCR", fg="#f38ba8"))

    def _update_prediction(self, text: str, conf: float, plate_name: str):
        """Actualiza la predicción en la UI (llamado desde hilo principal)."""
        if not (self._win and self._win.winfo_exists()):
            return
        self._lbl_prediction.config(text=text, fg="#89b4fa")
        self._lbl_conf.config(text="conf: %.2f" % conf)

        # Comparar con ground truth
        clean_pred = text.upper().replace("-", "").replace(" ", "")
        clean_name = plate_name.upper().replace("-", "").replace(" ", "")
        if clean_pred == clean_name:
            self._lbl_match.config(text="✓ COINCIDE", fg="#40a02b")
        else:
            # Calcular diferencia
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, clean_pred, clean_name).ratio()
            if ratio >= 0.7:
                self._lbl_match.config(
                    text="~ SIMILAR (%.0f%%)" % (ratio * 100), fg="#fe640b")
            else:
                self._lbl_match.config(text="✗ DIFERENTE", fg="#d20f39")

    def _get_current_prediction(self) -> str:
        """Obtiene el texto de predicción actual mostrado."""
        txt = self._lbl_prediction.cget("text")
        if txt in ("Analizando...", "Error OCR", "---", ""):
            return ""
        return txt

    def _record(self, status: str, corrected_to: str = ""):
        """Registra el resultado de validación actual."""
        if not self._images:
            return
        filepath, plate_name = self._images[self._idx]
        fname = os.path.basename(filepath)
        pred  = self._get_current_prediction()

        if "validated" not in self.log:
            self.log["validated"] = {}
        if "stats" not in self.log:
            self.log["stats"] = {"correct": 0, "incorrect": 0, "corrected": 0}

        # Si ya estaba validada, restar el conteo anterior
        prev = self.log["validated"].get(fname, {})
        if prev:
            prev_status = prev.get("status", "")
            if prev_status in self.log["stats"]:
                self.log["stats"][prev_status] = max(
                    0, self.log["stats"][prev_status] - 1)

        self.log["validated"][fname] = {
            "plate_name":   plate_name,
            "prediction":   pred,
            "status":       status,
            "corrected_to": corrected_to,
            "timestamp":    datetime.now().isoformat(),
        }
        self.log["stats"][status] = self.log["stats"].get(status, 0) + 1
        _save_validation_log(self.log)
        
        # ═══════════════════════════════════════════════════════════════════════
        # IMPORTANTE: También guardar en training_labels.json para el entrenamiento
        # ═══════════════════════════════════════════════════════════════════════
        training_labels = {}
        if os.path.exists("training_labels.json"):
            try:
                with open("training_labels.json", "r", encoding="utf-8") as f:
                    training_labels = json.load(f)
            except Exception:
                pass
        
        # Convertir el formato de validation_log a training_labels
        training_labels[filepath] = {
            "status": status,
            "correct_text": corrected_to if corrected_to else plate_name,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open("training_labels.json", "w", encoding="utf-8") as f:
                json.dump(training_labels, f, indent=2, ensure_ascii=False)
            
            # ═══════════════════════════════════════════════════════════════════
            # REENTRENAR OCR LEARNING EN TIEMPO REAL
            # ═══════════════════════════════════════════════════════════════════
            if OCR_LEARNING_AVAILABLE:
                try:
                    print(f"\n[OCR Learning] 🔄 Actualizando con nueva validación...")
                    ocr_learning = get_ocr_learning()
                    
                    # Recargar desde training_labels.json
                    corrections_before = ocr_learning.stats.get('unique_errors', 0)
                    ocr_learning.load_from_training_data()
                    ocr_learning.save_cache()
                    corrections_after = ocr_learning.stats.get('unique_errors', 0)
                    
                    if corrections_after > corrections_before:
                        print(f"[OCR Learning] ✅ Nueva corrección aprendida! Total: {corrections_after}")
                    else:
                        print(f"[OCR Learning] ℹ️ Validación registrada (Total: {corrections_after})")
                    
                    # Invalidar cache OCR para forzar nuevas lecturas con correcciones actualizadas
                    global _ocr_cache
                    _ocr_cache = OCRCache(maxsize=512)
                    print(f"[OCR Learning] 🗑️ Cache OCR limpiado para aplicar nuevas correcciones")
                    
                except Exception as e:
                    print(f"[OCR Learning] ⚠️ Error actualizando: {e}")
        except Exception as e:
            print(f"[ERROR] No se pudo guardar training_labels.json: {e}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # COMUNICACIÓN: Actualizar estado global de aprendizaje
        # ═══════════════════════════════════════════════════════════════════════
        try:
            learning_state = {}
            if os.path.exists("learning_state.json"):
                with open("learning_state.json", "r", encoding="utf-8") as f:
                    learning_state = json.load(f)
            
            # Actualizar estadísticas
            learning_state["total_validations"] = learning_state.get("total_validations", 0) + 1
            learning_state["stats"] = self.log.get("stats", {})
            
            # Marcar que hay datos nuevos pendientes de reentrenamiento
            # Si hay más de 20 validaciones nuevas desde el último retrain
            new_validations = learning_state["total_validations"] - learning_state.get("last_retrain_validations", 0)
            if new_validations >= 20:
                learning_state["pending_retraining"] = True
                print(f"[LEARNING] ⚠️ {new_validations} validaciones nuevas → Reentrenamiento recomendado")
            
            with open("learning_state.json", "w", encoding="utf-8") as f:
                json.dump(learning_state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[LEARNING] Error actualizando estado: {e}")

        # Actualizar pending count
        self._pending_count = sum(
            1 for p, _ in self._images
            if os.path.basename(p) not in self.log.get("validated", {}))

    def _mark_correct(self):
        self._record("correct")
        self._next()

    def _mark_incorrect(self):
        self._record("incorrect")
        self._next()

    def _discard_image(self):
        """
        Descarta la imagen actual (basura/no útil).
        Mueve la imagen a una carpeta 'discarded' y la elimina de la lista.
        """
        if not self._images:
            return
        
        filepath, plate_name = self._images[self._idx]
        fname = os.path.basename(filepath)
        
        # Crear carpeta de descartados si no existe
        discarded_dir = os.path.join(os.path.dirname(filepath), "discarded")
        os.makedirs(discarded_dir, exist_ok=True)
        
        # Mover archivo a carpeta de descartados
        try:
            dest_path = os.path.join(discarded_dir, fname)
            # Si ya existe, agregar timestamp
            if os.path.exists(dest_path):
                name, ext = os.path.splitext(fname)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_path = os.path.join(discarded_dir, f"{name}_{timestamp}{ext}")
            
            import shutil
            shutil.move(filepath, dest_path)
            print(f"[DISCARD] Imagen descartada: {fname} → discarded/")
            
            # Registrar en el log como descartada
            self._record("discarded")
            
            # Eliminar de la lista de imágenes
            self._images.pop(self._idx)
            
            # Ajustar índice si es necesario
            if self._idx >= len(self._images):
                self._idx = max(0, len(self._images) - 1)
            
            # Actualizar contador de pendientes
            self._pending_count = sum(
                1 for p, _ in self._images
                if os.path.basename(p) not in self.log.get("validated", {}))
            
            # Mostrar siguiente imagen o cerrar si no hay más
            if self._images:
                self._show_current()
            else:
                messagebox.showinfo("Validación completa", 
                                  "No hay más imágenes para validar.",
                                  parent=self._win)
                self._win.destroy()
                
        except Exception as e:
            messagebox.showerror("Error", 
                               f"No se pudo descartar la imagen:\n{e}",
                               parent=self._win)

    def _mark_correct_with_edit(self):
        """
        Abre ventana de corrección con:
          1. Campo de texto para corregir la lectura OCR
          2. Canvas interactivo para dibujar el contorno de la placa
             → guarda el recorte en char_cache/plate_crops/ para reentrenar YOLO
        """
        if not self._images:
            return
        filepath, plate_name = self._images[self._idx]
        pred = self._get_current_prediction()
        img_bgr = self._current_img_bgr

        win = tk.Toplevel(self._win)
        win.title("Corregir — Dibuja el contorno de la placa")
        win.configure(bg="#1e1e2e")
        win.grab_set()

        BG, FG = "#1e1e2e", "#cdd6f4"

        # ── Instrucciones ─────────────────────────────────────────────────────
        tk.Label(win,
                 text="1. Escribe el texto correcto   2. Click en 4 esquinas de la placa (TL→TR→BR→BL)   3. Confirmar",
                 bg="#181825", fg="#89b4fa",
                 font=("Consolas", 9), pady=6).pack(fill="x")

        # ── Fila superior: texto ──────────────────────────────────────────────
        top_row = tk.Frame(win, bg=BG)
        top_row.pack(fill="x", padx=12, pady=6)

        tk.Label(top_row, text="Nombre archivo:",
                 bg=BG, fg="#f9e2af",
                 font=("Consolas", 10, "bold")).pack(side="left")
        tk.Label(top_row, text="  %s" % plate_name,
                 bg=BG, fg="#f9e2af",
                 font=("Consolas", 12, "bold")).pack(side="left", padx=8)

        tk.Label(top_row, text="  Prediccion:",
                 bg=BG, fg="#89b4fa",
                 font=("Consolas", 10)).pack(side="left", padx=(16, 0))
        tk.Label(top_row, text="  %s" % pred,
                 bg=BG, fg="#89b4fa",
                 font=("Consolas", 12, "bold")).pack(side="left")

        tk.Label(top_row, text="  Texto correcto:",
                 bg=BG, fg="#cdd6f4",
                 font=("Consolas", 10)).pack(side="left", padx=(16, 0))
        entry = tk.Entry(top_row, font=("Consolas", 13),
                         bg="#313244", fg="#cdd6f4",
                         insertbackground="white", width=14)
        entry.insert(0, plate_name)
        entry.pack(side="left", padx=6)
        entry.select_range(0, tk.END)
        entry.focus()

        # ── Canvas con la imagen ──────────────────────────────────────────────
        # Escalar imagen para que quepa en pantalla
        canvas_w, canvas_h = 800, 450
        if img_bgr is not None:
            ih, iw = img_bgr.shape[:2]
            scale = min(canvas_w / iw, canvas_h / ih, 1.0)
            disp_w = max(1, int(iw * scale))
            disp_h = max(1, int(ih * scale))
            disp_bgr = cv2.resize(img_bgr, (disp_w, disp_h),
                                  interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
            from PIL import Image as _PILImage
            _pil = _PILImage.fromarray(rgb)
            _photo = ImageTk.PhotoImage(_pil)
        else:
            disp_w, disp_h, scale = canvas_w, canvas_h, 1.0
            _photo = None

        canvas = tk.Canvas(win, width=disp_w, height=disp_h,
                           bg="#000000", cursor="crosshair",
                           highlightthickness=1,
                           highlightbackground="#45475a")
        canvas.pack(padx=12, pady=6)

        if _photo:
            canvas.create_image(0, 0, anchor="nw", image=_photo)
            canvas._photo_ref = _photo  # evitar GC

        # ── Estado del dibujo (4 PUNTOS) ─────────────────────────────────────
        draw_state = {
            "points": [],          # Lista de (x, y) en coords del canvas
            "points_orig": [],     # Lista de (x, y) en coords de la imagen original
            "point_ids": [],       # IDs de los círculos dibujados
            "line_ids": [],        # IDs de las líneas dibujadas
            "bbox_orig": None,     # (x1,y1,x2,y2) del rectángulo envolvente
        }

        lbl_bbox = tk.Label(win,
                            text="Sin selección — click en las 4 esquinas de la placa (0/4)",
                            bg=BG, fg="#6c7086",
                            font=("Consolas", 9))
        lbl_bbox.pack()

        def on_click(event):
            """Agregar punto al hacer click."""
            if len(draw_state["points"]) >= 4:
                # Ya hay 4 puntos, reiniciar
                for pid in draw_state["point_ids"]:
                    canvas.delete(pid)
                for lid in draw_state["line_ids"]:
                    canvas.delete(lid)
                draw_state["points"] = []
                draw_state["points_orig"] = []
                draw_state["point_ids"] = []
                draw_state["line_ids"] = []
                draw_state["bbox_orig"] = None
                lbl_bbox.config(text="Selección reiniciada — click en las 4 esquinas (0/4)",
                                fg="#6c7086")
            
            # Agregar nuevo punto
            x, y = event.x, event.y
            draw_state["points"].append((x, y))
            
            # Convertir a coordenadas originales
            if img_bgr is not None:
                ox = int(x / scale)
                oy = int(y / scale)
                ih, iw = img_bgr.shape[:2]
                ox = max(0, min(ox, iw - 1))
                oy = max(0, min(oy, ih - 1))
                draw_state["points_orig"].append((ox, oy))
            else:
                draw_state["points_orig"].append((x, y))
            
            # Dibujar punto
            point_id = canvas.create_oval(
                x - 6, y - 6, x + 6, y + 6,
                fill="#40a02b", outline="#cdd6f4", width=2)
            draw_state["point_ids"].append(point_id)
            
            # Dibujar número del punto
            num_id = canvas.create_text(
                x + 15, y - 15,
                text=str(len(draw_state["points"])),
                fill="#cdd6f4", font=("Consolas", 12, "bold"))
            draw_state["point_ids"].append(num_id)
            
            # Dibujar línea al punto anterior
            if len(draw_state["points"]) > 1:
                prev_x, prev_y = draw_state["points"][-2]
                line_id = canvas.create_line(
                    prev_x, prev_y, x, y,
                    fill="#89b4fa", width=2)
                draw_state["line_ids"].append(line_id)
            
            # Si ya hay 4 puntos, cerrar el polígono
            if len(draw_state["points"]) == 4:
                first_x, first_y = draw_state["points"][0]
                line_id = canvas.create_line(
                    x, y, first_x, first_y,
                    fill="#89b4fa", width=2)
                draw_state["line_ids"].append(line_id)
                
                # Calcular bbox envolvente en coordenadas originales
                if draw_state["points_orig"]:
                    xs = [p[0] for p in draw_state["points_orig"]]
                    ys = [p[1] for p in draw_state["points_orig"]]
                    ox0, ox1 = min(xs), max(xs)
                    oy0, oy1 = min(ys), max(ys)
                    draw_state["bbox_orig"] = (ox0, oy0, ox1, oy1)
                    
                    w_px = ox1 - ox0
                    h_px = oy1 - oy0
                    ar = w_px / max(h_px, 1)
                    lbl_bbox.config(
                        text="✓ 4 puntos marcados | Envolvente: %dx%d px | AR=%.1f | Presiona Confirmar" % (
                            w_px, h_px, ar),
                        fg="#40a02b")
            else:
                lbl_bbox.config(
                    text="Puntos marcados: %d/4 — continúa marcando esquinas" % len(draw_state["points"]),
                    fg="#89b4fa")

        canvas.bind("<ButtonPress-1>", on_click)

        # ── Botones ───────────────────────────────────────────────────────────
        btn_row = tk.Frame(win, bg=BG)
        btn_row.pack(pady=10)

        def confirmar():
            corrected = entry.get().strip().upper()
            if not corrected:
                messagebox.showwarning("Falta texto",
                                       "Escribe el texto correcto de la placa.",
                                       parent=win)
                return

            bbox = draw_state.get("bbox_orig")
            crop_path = ""

            # ── Guardar recorte de placa CON CORRECCIÓN DE PERSPECTIVA ────────
            if img_bgr is not None:
                crop = None
                
                # Si hay 4 puntos, aplicar corrección de perspectiva
                if len(draw_state["points_orig"]) == 4:
                    pts_orig = np.array(draw_state["points_orig"], dtype=np.float32)
                    
                    # Ordenar puntos: TL, TR, BR, BL
                    center = pts_orig.mean(axis=0)
                    
                    def angle_from_center(pt):
                        return np.arctan2(pt[1] - center[1], pt[0] - center[0])
                    
                    sorted_pts = sorted(pts_orig, key=angle_from_center)
                    sums = [pt[0] + pt[1] for pt in sorted_pts]
                    tl_idx = sums.index(min(sums))
                    ordered_pts = sorted_pts[tl_idx:] + sorted_pts[:tl_idx]
                    src_pts = np.array(ordered_pts, dtype=np.float32)
                    
                    # Calcular dimensiones del rectángulo destino (aspect ratio 3:1)
                    if bbox:
                        x0, y0, x1, y1 = bbox
                        target_area = (x1 - x0) * (y1 - y0)
                        target_height = int(np.sqrt(target_area / 3.0))
                        target_width = target_height * 3
                    else:
                        target_width = 600
                        target_height = 200
                    
                    # Limitar tamaño máximo
                    if target_width > 600:
                        target_width = 600
                        target_height = 200
                    
                    # Puntos de destino (rectángulo perfecto)
                    dst_pts = np.array([
                        [0, 0],
                        [target_width - 1, 0],
                        [target_width - 1, target_height - 1],
                        [0, target_height - 1]
                    ], dtype=np.float32)
                    
                    # Aplicar transformación de perspectiva
                    try:
                        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        crop = cv2.warpPerspective(img_bgr, matrix, 
                                                   (target_width, target_height))
                        print(f"[Corrección] Perspectiva aplicada: {target_width}x{target_height}px")
                    except Exception as e:
                        print(f"[Error] Transformación de perspectiva falló: {e}")
                        crop = None
                
                # Si no hay 4 puntos o falló la transformación, usar bbox simple
                if crop is None and bbox:
                    x0, y0, x1, y1 = bbox
                    if (x1 - x0) > 10 and (y1 - y0) > 5:
                        crop = img_bgr[y0:y1, x0:x1]
                
                # Guardar crop si existe
                if crop is not None and crop.size > 0:
                    crops_dir = os.path.join("char_cache", "plate_crops")
                    os.makedirs(crops_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    crop_fname = "%s_%s.jpg" % (ts, corrected)
                    crop_path  = os.path.join(crops_dir, crop_fname)
                    cv2.imwrite(crop_path, crop)
                    print(f"[Guardado] Crop: {crop_path} ({crop.shape[1]}x{crop.shape[0]}px)")

                    # ── Guardar anotación YOLO (solo si hay bbox) ─────────────
                    if bbox:
                        x0, y0, x1, y1 = bbox
                        ih, iw = img_bgr.shape[:2]
                        cx = ((x0 + x1) / 2) / iw
                        cy = ((y0 + y1) / 2) / ih
                        bw = (x1 - x0) / iw
                        bh = (y1 - y0) / ih
                        yolo_dir = os.path.join("char_cache", "yolo_labels")
                        os.makedirs(yolo_dir, exist_ok=True)
                        label_path = os.path.join(
                            yolo_dir,
                            "%s_%s.txt" % (ts, corrected))
                        with open(label_path, "w") as lf:
                            lf.write("0 %.6f %.6f %.6f %.6f\n" % (cx, cy, bw, bh))

                        # ── Copiar imagen original al directorio YOLO ─────────
                        img_yolo_path = os.path.join(
                            yolo_dir,
                            "%s_%s.jpg" % (ts, corrected))
                        cv2.imwrite(img_yolo_path, img_bgr)

            # ── Registrar en el log ───────────────────────────────────────────
            self._record_with_bbox("corrected",
                                   corrected_to=corrected,
                                   bbox=bbox,
                                   crop_path=crop_path)
            win.destroy()
            self._next()

        def eliminar():
            """Elimina la imagen actual y avanza a la siguiente."""
            if not self._images:
                return
            
            filepath, plate_name = self._images[self._idx]
            
            # Confirmar eliminación
            respuesta = messagebox.askyesno(
                "Confirmar Eliminación",
                f"¿Eliminar esta imagen?\n\n{plate_name}\n\nEsta acción no se puede deshacer.",
                parent=win,
                icon="warning"
            )
            
            if not respuesta:
                return
            
            # Eliminar archivo físico
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"[Eliminado] {filepath}")
                
                # Registrar en el log como eliminada
                self._record("deleted")
                
                # Cerrar ventana y avanzar
                win.destroy()
                self._next()
                
                messagebox.showinfo(
                    "Eliminado",
                    f"Imagen eliminada: {plate_name}",
                    parent=self._win
                )
            
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"No se pudo eliminar la imagen:\n{str(e)}",
                    parent=win
                )

        tk.Button(btn_row, text="✓  Confirmar",
                  bg="#40a02b", fg="white",
                  font=("Consolas", 12, "bold"),
                  relief="flat", padx=20, pady=8,
                  command=confirmar).pack(side="left", padx=8)

        tk.Button(btn_row, text="🗑  Eliminar",
                  bg="#f38ba8", fg="white",
                  font=("Consolas", 11, "bold"),
                  relief="flat", padx=16, pady=8,
                  command=eliminar).pack(side="left", padx=8)

        tk.Button(btn_row, text="✗  Cancelar",
                  bg="#313244", fg="#cdd6f4",
                  font=("Consolas", 11),
                  relief="flat", padx=16, pady=8,
                  command=win.destroy).pack(side="left", padx=8)

        entry.bind("<Return>", lambda e: confirmar())

        # Ajustar tamaño de ventana al contenido
        win.update_idletasks()
        win.geometry("%dx%d" % (
            max(disp_w + 40, 700),
            disp_h + 200))

    def _record_with_bbox(self, status: str, corrected_to: str = "",
                          bbox=None, crop_path: str = ""):
        """Versión extendida de _record que también guarda bbox y crop_path."""
        if not self._images:
            return
        filepath, plate_name = self._images[self._idx]
        fname = os.path.basename(filepath)
        pred  = self._get_current_prediction()

        if "validated" not in self.log:
            self.log["validated"] = {}
        if "stats" not in self.log:
            self.log["stats"] = {"correct": 0, "incorrect": 0, "corrected": 0}

        prev = self.log["validated"].get(fname, {})
        if prev:
            prev_status = prev.get("status", "")
            if prev_status in self.log["stats"]:
                self.log["stats"][prev_status] = max(
                    0, self.log["stats"][prev_status] - 1)

        entry_data = {
            "plate_name":   plate_name,
            "prediction":   pred,
            "status":       status,
            "corrected_to": corrected_to,
            "timestamp":    datetime.now().isoformat(),
        }
        if bbox:
            entry_data["bbox"] = list(bbox)
        if crop_path:
            entry_data["crop_path"] = crop_path

        self.log["validated"][fname] = entry_data
        self.log["stats"][status] = self.log["stats"].get(status, 0) + 1
        _save_validation_log(self.log)
        
        # ═══════════════════════════════════════════════════════════════════════
        # IMPORTANTE: También guardar en training_labels.json para el entrenamiento
        # ═══════════════════════════════════════════════════════════════════════
        training_labels = {}
        if os.path.exists("training_labels.json"):
            try:
                with open("training_labels.json", "r", encoding="utf-8") as f:
                    training_labels = json.load(f)
            except Exception:
                pass
        
        training_labels[filepath] = {
            "status": status,
            "correct_text": corrected_to if corrected_to else plate_name,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open("training_labels.json", "w", encoding="utf-8") as f:
                json.dump(training_labels, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] No se pudo guardar training_labels.json: {e}")

        self._pending_count = sum(
            1 for p, _ in self._images
            if os.path.basename(p) not in self.log.get("validated", {}))

    def _prev(self):
        if self._idx > 0:
            self._idx -= 1
            self._show_current()

    def _next(self):
        if self._idx < len(self._images) - 1:
            self._idx += 1
            self._show_current()
        else:
            # Llegó al final
            stats = self.log.get("stats", {})
            messagebox.showinfo(
                "Validacion completa",
                "Has revisado todas las imagenes.\n\n"
                "Resultados:\n"
                "  Correctas  : %d\n"
                "  Incorrectas: %d\n"
                "  Corregidas : %d\n\n"
                "Los datos se guardaron en:\n"
                "char_cache/validation_log.json" % (
                    stats.get("correct", 0),
                    stats.get("incorrect", 0),
                    stats.get("corrected", 0)))

    def _export(self):
        """Exporta los resultados a Excel."""
        try:
            import pandas as pd
            validated = self.log.get("validated", {})
            if not validated:
                messagebox.showinfo("Exportar", "No hay datos validados aun.")
                return
            rows = []
            for fname, data in validated.items():
                rows.append({
                    "archivo":      fname,
                    "placa_nombre": data.get("plate_name", ""),
                    "prediccion":   data.get("prediction", ""),
                    "estado":       data.get("status", ""),
                    "correccion":   data.get("corrected_to", ""),
                    "fecha":        data.get("timestamp", ""),
                })
            df = pd.DataFrame(rows)
            fname_out = "validacion_%s.xlsx" % datetime.now().strftime("%Y%m%d_%H%M%S")
            df.to_excel(fname_out, index=False, engine="openpyxl")
            messagebox.showinfo("Exportar",
                "Exportado: %s\n%d registros" % (fname_out, len(rows)))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _add_images(self):
        """
        Permite agregar imágenes manualmente desde el explorador de archivos.
        El usuario puede:
          1. Seleccionar una o más imágenes (JPG, PNG)
          2. Especificar el texto de la placa para cada imagen
          3. Las imágenes se copian a event_images/ con el formato correcto
        """
        from tkinter import filedialog
        
        # Seleccionar archivos
        files = filedialog.askopenfilenames(
            parent=self._win,
            title="Seleccionar imágenes de placas",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if not files:
            return
        
        # Ventana para ingresar el texto de cada placa
        add_win = tk.Toplevel(self._win)
        add_win.title("Agregar Imágenes — Especificar Placas")
        add_win.configure(bg="#1e1e2e")
        add_win.grab_set()
        add_win.geometry("700x500")
        
        BG, FG = "#1e1e2e", "#cdd6f4"
        
        tk.Label(add_win,
                 text="Especifica el texto de la placa para cada imagen",
                 bg="#181825", fg="#89b4fa",
                 font=("Consolas", 11, "bold"), pady=8).pack(fill="x")
        
        # Frame con scroll para la lista de imágenes
        container = tk.Frame(add_win, bg=BG)
        container.pack(fill="both", expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(container, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=BG)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Lista de entries para cada imagen
        entries = []
        
        for i, filepath in enumerate(files):
            fname = os.path.basename(filepath)
            
            row = tk.Frame(scrollable_frame, bg="#313244", pady=6, padx=8)
            row.pack(fill="x", pady=3, padx=5)
            
            # Número
            tk.Label(row, text=f"{i+1}.",
                     bg="#313244", fg="#6c7086",
                     font=("Consolas", 10), width=3).pack(side="left")
            
            # Nombre del archivo
            tk.Label(row, text=fname[:40] + ("..." if len(fname) > 40 else ""),
                     bg="#313244", fg="#cdd6f4",
                     font=("Consolas", 9), width=45,
                     anchor="w").pack(side="left", padx=5)
            
            # Entry para el texto de la placa
            entry = tk.Entry(row, font=("Consolas", 11),
                           bg="#1e1e2e", fg="#f9e2af",
                           insertbackground="white", width=12)
            entry.pack(side="left", padx=5)
            entry.insert(0, "")
            entries.append((filepath, entry))
        
        # Instrucciones
        tk.Label(add_win,
                 text="Formato: ABC123 o ABC-123-D (sin espacios innecesarios)",
                 bg=BG, fg="#6c7086",
                 font=("Consolas", 8)).pack(pady=2)
        
        # Botones
        btn_frame = tk.Frame(add_win, bg=BG)
        btn_frame.pack(pady=10)
        
        def confirmar_agregar():
            added_count = 0
            skipped_count = 0
            errors = []
            added_files = []  # Para debug
            
            for filepath, entry in entries:
                plate_text = entry.get().strip().upper()
                
                # Validar que se ingresó texto
                if not plate_text:
                    skipped_count += 1
                    continue
                
                # Validar formato básico (3-9 caracteres alfanuméricos)
                clean_plate = plate_text.replace("-", "").replace(" ", "")
                if not (3 <= len(clean_plate) <= 9 and clean_plate.isalnum()):
                    errors.append(f"{os.path.basename(filepath)}: formato inválido '{plate_text}'")
                    continue
                
                try:
                    # Leer imagen original
                    img = cv2.imread(filepath)
                    if img is None:
                        errors.append(f"{os.path.basename(filepath)}: no se pudo leer")
                        continue
                    
                    # Generar nombre con timestamp (formato: YYYYMMDD_HHMMSS_MICROSEC_PLACA.jpg)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    new_fname = f"{ts}_{plate_text}.jpg"
                    dest_path = os.path.join(IMAGE_FOLDER, new_fname)
                    
                    # Asegurar que la carpeta existe
                    os.makedirs(IMAGE_FOLDER, exist_ok=True)
                    
                    # Guardar en event_images/
                    success = cv2.imwrite(dest_path, img)
                    if success:
                        added_count += 1
                        added_files.append(new_fname)
                    else:
                        errors.append(f"{os.path.basename(filepath)}: error al guardar")
                    
                except Exception as e:
                    errors.append(f"{os.path.basename(filepath)}: {str(e)}")
            
            # Mostrar resultado con nombres de archivos agregados
            msg = f"Imágenes agregadas: {added_count}\n"
            if added_files and added_count <= 5:
                msg += "\nArchivos creados:\n" + "\n".join(f"  • {f}" for f in added_files)
            if skipped_count > 0:
                msg += f"\n\nSin texto (omitidas): {skipped_count}"
            if errors:
                msg += f"\n\nErrores ({len(errors)}):\n" + "\n".join(errors[:5])
                if len(errors) > 5:
                    msg += f"\n... y {len(errors)-5} más"
            
            # IMPORTANTE: Primero actualizar la lista ANTES de cerrar el diálogo
            if added_count > 0:
                print(f"[DEBUG] === ACTUALIZANDO LISTA ===")
                print(f"[DEBUG] Archivos agregados: {added_files}")
                
                # Recargar lista de imágenes
                old_count = len(self._images)
                print(f"[DEBUG] Imágenes antes de recargar: {old_count}")
                
                self._load_image_list()
                
                new_count = len(self._images)
                print(f"[DEBUG] Imágenes después de recargar: {new_count}")
                print(f"[DEBUG] Pendientes: {self._pending_count}")
                
                # Si se agregaron imágenes, ir a la primera nueva
                if new_count > old_count:
                    self._idx = 0
                    print(f"[DEBUG] Índice reseteado a 0")
                    print(f"[DEBUG] Primera imagen: {self._images[0] if self._images else 'NINGUNA'}")
            
            # Cerrar diálogo
            add_win.destroy()
            
            # Mostrar mensaje de resultado
            messagebox.showinfo("Agregar Imágenes", msg, parent=self._win)
            
            # Actualizar interfaz DESPUÉS de cerrar todo
            if added_count > 0:
                if self._win and self._win.winfo_exists():
                    print(f"[DEBUG] Actualizando interfaz...")
                    self._win.update_idletasks()
                    self._show_current()
                    print(f"[DEBUG] Interfaz actualizada - mostrando imagen {self._idx + 1}/{len(self._images)}")
        
        tk.Button(btn_frame, text="✓  Agregar todas",
                  bg="#40a02b", fg="white",
                  font=("Consolas", 11, "bold"),
                  relief="flat", padx=20, pady=8,
                  command=confirmar_agregar).pack(side="left", padx=8)
        
        tk.Button(btn_frame, text="✗  Cancelar",
                  bg="#313244", fg="#cdd6f4",
                  font=("Consolas", 11),
                  relief="flat", padx=16, pady=8,
                  command=add_win.destroy).pack(side="left", padx=8)

    def _on_close(self):
        _save_validation_log(self.log)
        if self._win:
            self._win.destroy()
            self._win = None


# ============================================================
# CLASE PRINCIPAL GUI
# ============================================================
lpr_app_instance = None


# ══════════════════════════════════════════════════════════════════════════════
# VALIDADOR MANUAL DE DATOS DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
LABELS_FILE = "training_labels.json"

def _load_labels() -> dict:
    if os.path.exists(LABELS_FILE):
        try:
            with open(LABELS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception: pass
    return {}

def _save_labels(data: dict):
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def _collect_training_images() -> list:
    """
    Recolecta imágenes de entrenamiento de múltiples fuentes.
    Soporta: .jpg, .jpeg, .png, .enc
    """
    items = []
    sources = {IMAGE_FOLDER: "event_images", "training_plates": "training_plates"}
    
    for folder, src_label in sources.items():
        if not os.path.isdir(folder):
            print(f"[Validador] Carpeta no existe: {folder}/")
            continue
        
        files = sorted(os.listdir(folder))
        count = 0
        
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in {".jpg",".jpeg",".png",".enc"}:
                continue
            if fname.startswith("inv_"):
                continue
            
            # Extraer texto predicho del nombre del archivo
            base = fname
            for s in (".enc",".jpg",".jpeg",".png"):
                base = base.replace(s,"")
            base = base.replace("_raw","")
            
            # El texto predicho está al final del nombre
            parts = base.split("_")
            predicted = parts[-1].upper() if parts else ""
            
            # Validar que el texto predicho sea razonable
            if len(predicted) < 3 or len(predicted) > 10:
                # Si no hay predicción válida, usar "UNKNOWN"
                predicted = "UNKNOWN"
            
            items.append({
                "path": os.path.join(folder, fname),
                "source": src_label,
                "predicted": predicted,
                "is_enc": ext == ".enc",
                "fname": fname
            })
            count += 1
        
        print(f"[Validador] {src_label}: {count} imágenes encontradas")
    
    return items

def _load_plate_img(path: str, is_enc: bool):
    try:
        if is_enc and _fernet:
            with open(path,"rb") as f: raw = _fernet.decrypt(f.read())
            arr = np.frombuffer(raw, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.imread(path)
    except Exception: return None


class TrainingValidatorWindow:
    """
    Valida manualmente las imágenes de entrenamiento del sistema LPR.

    Fuentes de datos que valida:
      • event_images/*.enc  — capturas automáticas de prueba2.py (cámara/video)
      • training_plates/*.jpg — extraídas por Video_to_training.py

    Resultado guardado en training_labels.json:
      "correct"   → predicción correcta, usar en training
      "incorrect" → imagen ilegible/mala, saltar en training
      "corrected" → texto corregido manualmente
    """
    _COLORS = {"correct":"#a6e3a1","incorrect":"#f38ba8",
               "corrected":"#fab387","pending":"#6c7086"}

    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Validación de datos de entrenamiento")
        self.win.geometry("900x750")  # Aumentado de 840x700 a 900x750
        self.win.configure(bg="#1e1e2e")
        self.items  = _collect_training_images()
        self.labels = _load_labels()
        self.idx    = 0
        self._itk   = None

        if not self.items:
            messagebox.showinfo("Sin datos",
                "No hay imágenes en event_images/ ni training_plates/.\n"
                "Captura placas con la cámara o carga un video primero.",
                parent=self.win)
            self.win.destroy(); return

        # Ir al primer pendiente
        for i,it in enumerate(self.items):
            if it["path"] not in self.labels:
                self.idx = i; break

        self._build_ui()
        self._show()
        self.win.bind("<Return>", lambda e: self._mark("correct"))
        self.win.bind("n",        lambda e: self._mark("incorrect"))
        self.win.bind("c",        lambda e: self._focus_entry())
        self.win.bind("<Right>",  lambda e: self._nav(+1))
        self.win.bind("<Left>",   lambda e: self._nav(-1))
        self.win.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self):
        w = self.win
        # Barra superior
        top = tk.Frame(w, bg="#181825", pady=6); top.pack(fill="x")
        self.lbl_prog  = tk.Label(top, bg="#181825", fg="#cdd6f4",
                                  font=("Segoe UI",11,"bold"))
        self.lbl_prog.pack(side="left", padx=12)
        self.lbl_stats = tk.Label(top, bg="#181825", fg="#a6adc8",
                                  font=("Segoe UI",10))
        self.lbl_stats.pack(side="right", padx=12)
        # Imagen
        self.lbl_img = tk.Label(w, bg="#313244", width=800, height=190,
                                text="Cargando...", fg="#6c7086",
                                font=("Segoe UI",12))
        self.lbl_img.pack(padx=20, pady=8)
        # Predicción grande
        self.lbl_pred = tk.Label(w, bg="#1e1e2e", fg="#cdd6f4",
                                 font=("Segoe UI",28,"bold"))
        self.lbl_pred.pack()
        self.lbl_meta = tk.Label(w, bg="#1e1e2e", fg="#a6adc8",
                                 font=("Segoe UI",9))
        self.lbl_meta.pack()
        self.lbl_est  = tk.Label(w, bg="#1e1e2e", fg="#fab387",
                                 font=("Segoe UI",10,"italic"))
        self.lbl_est.pack(pady=(2,0))
        # Botones
        bf = tk.Frame(w, bg="#1e1e2e", pady=14); bf.pack()
        btn = {"font":("Segoe UI",13,"bold"),"relief":"flat",
               "cursor":"hand2","width":13,"height":2}
        tk.Button(bf, text="✅  CORRECTO",  bg="#a6e3a1", fg="#1e1e2e",
                  command=lambda:self._mark("correct"),   **btn).grid(row=0,column=0,padx=10)
        tk.Button(bf, text="❌  INCORRECTO",bg="#f38ba8", fg="#1e1e2e",
                  command=lambda:self._mark("incorrect"), **btn).grid(row=0,column=1,padx=10)
        tk.Button(bf, text="✏️  CORREGIR",  bg="#fab387", fg="#1e1e2e",
                  command=self._focus_entry,              **btn).grid(row=0,column=2,padx=10)
        # Campo corrección
        cf = tk.Frame(w, bg="#1e1e2e"); cf.pack(pady=(0,10))
        tk.Label(cf, text="Texto correcto:", bg="#1e1e2e", fg="#a6adc8",
                 font=("Segoe UI",10)).pack(side="left", padx=(0,6))
        self.entry = tk.Entry(cf, font=("Segoe UI",16,"bold"), width=12,
                              bg="#313244", fg="#cdd6f4",
                              insertbackground="#cdd6f4", relief="flat", bd=4)
        self.entry.pack(side="left")
        self.entry.bind("<Return>", lambda e: self._save_correction())
        tk.Button(cf, text="Guardar", font=("Segoe UI",10),
                  bg="#89b4fa", fg="#1e1e2e", relief="flat", cursor="hand2",
                  command=self._save_correction).pack(side="left", padx=8)
        # Progress bar
        self.pb = ttk.Progressbar(w, mode="determinate", length=800)
        self.pb.pack(padx=20, pady=(4,0))
        # Nav inferior
        nav = tk.Frame(w, bg="#181825", pady=8); nav.pack(fill="x", side="bottom")
        nb = {"font":("Segoe UI",10),"bg":"#313244","fg":"#cdd6f4",
              "relief":"flat","cursor":"hand2","padx":14,"pady":6}
        tk.Button(nav, text="◀ Anterior",command=lambda:self._nav(-1),**nb).pack(side="left",padx=8)
        tk.Button(nav, text="Saltar ▶",  command=lambda:self._nav(+1),**nb).pack(side="left",padx=4)
        tk.Button(nav, text="📁 Cargar imágenes",command=self._cargar_imagenes,
                  bg="#89b4fa",fg="#1e1e2e",**nb).pack(side="left",padx=4)
        tk.Label(nav, text="Enter=✅  N=❌  C=✏️  ←→=Navegar",
                 bg="#181825",fg="#6c7086",font=("Segoe UI",9)).pack(side="left",padx=16)
        tk.Button(nav, text="📊 Resumen",command=self._summary,**nb).pack(side="right",padx=4)
        tk.Button(nav, text="💾 Guardar", command=self._save_all,**nb).pack(side="right",padx=4)

    def _show(self):
        if not self.items: return
        it   = self.items[self.idx]
        path = it["path"]
        # Imagen
        img  = _load_plate_img(path, it["is_enc"])
        if img is not None:
            h,w  = img.shape[:2]
            sc   = min(800/max(w,1), 190/max(h,1))
            nw,nh = max(1,int(w*sc)), max(1,int(h*sc))
            rgb  = cv2.cvtColor(cv2.resize(img,(nw,nh)), cv2.COLOR_BGR2RGB)
            self._itk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.lbl_img.config(image=self._itk, text="", width=nw, height=nh)
        else:
            self.lbl_img.config(image="", text="⚠ No se pudo cargar", width=800, height=190)
        # Estado
        ld   = self.labels.get(path, {})
        pred = it["predicted"]
        st   = ld.get("status","pending")
        disp = format_mx_plate(ld.get("correct_text",pred)) if st=="corrected" else format_mx_plate(pred)
        self.lbl_pred.config(text=disp, fg=self._COLORS.get(st,"#cdd6f4"))
        kb   = os.path.getsize(path)//1024 if os.path.exists(path) else 0
        self.lbl_meta.config(text=f"{it['fname']}  |  {it['source']}/  |  {kb} KB")
        est_txt = {"correct":"✅ Validado como CORRECTO",
                   "incorrect":"❌ Marcado como INCORRECTO (se omitirá en training)",
                   "corrected":f"✏️  Corregido → {ld.get('correct_text','')}",
                   "pending":"⏳ Pendiente de validar"}.get(st,"")
        self.lbl_est.config(text=est_txt, fg=self._COLORS.get(st,"#6c7086"))
        # Progreso
        total    = len(self.items)
        reviewed = sum(1 for i in self.items if i["path"] in self.labels)
        correct  = sum(1 for v in self.labels.values() if v.get("status")=="correct")
        wrong    = sum(1 for v in self.labels.values() if v.get("status")=="incorrect")
        fixed    = sum(1 for v in self.labels.values() if v.get("status")=="corrected")
        self.lbl_prog.config(text=f"Imagen {self.idx+1}/{total}  —  revisadas: {reviewed}/{total}")
        self.lbl_stats.config(text=f"✅ {correct}   ❌ {wrong}   ✏️  {fixed}   ⏳ {total-reviewed}")
        self.pb["maximum"] = total; self.pb["value"] = reviewed
        self.entry.delete(0, tk.END)
        self.entry.insert(0, ld.get("correct_text", pred) if st=="corrected" else pred)

    def _nav(self, d):
        self.idx = (self.idx + d) % len(self.items); self._show()

    def _mark(self, status):
        it = self.items[self.idx]
        self.labels[it["path"]] = {
            "status": status, "predicted": it["predicted"],
            "correct_text": it["predicted"],
            "reviewed_at": datetime.now().isoformat(timespec="seconds"),
            "source": it["source"]}
        _save_labels(self.labels); self._show(); self._nav(+1)

    def _focus_entry(self):
        self.entry.focus_set(); self.entry.select_range(0, tk.END)

    def _save_correction(self):
        raw  = "".join(c for c in self.entry.get().strip().upper() if c.isalnum())
        if len(raw) < 4:
            messagebox.showwarning("Inválido","Mínimo 4 caracteres.",parent=self.win); return
        it = self.items[self.idx]
        self.labels[it["path"]] = {
            "status": "corrected", "predicted": it["predicted"],
            "correct_text": raw,
            "reviewed_at": datetime.now().isoformat(timespec="seconds"),
            "source": it["source"]}
        _save_labels(self.labels); self._show(); self._nav(+1)

    def _save_all(self):
        _save_labels(self.labels)
        messagebox.showinfo("Guardado",f"{len(self.labels)} validaciones en {LABELS_FILE}",
                            parent=self.win)

    def _cargar_imagenes(self):
        """Permite cargar imágenes desde el explorador de archivos."""
        archivos = filedialog.askopenfilenames(
            title="Seleccionar imágenes de placas",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Todos", "*.*")
            ],
            parent=self.win
        )
        
        if not archivos:
            return
        
        # Crear carpeta training_plates si no existe
        os.makedirs("training_plates", exist_ok=True)
        
        copiadas = 0
        duplicadas = 0
        
        for archivo in archivos:
            try:
                nombre_original = os.path.basename(archivo)
                destino = os.path.join("training_plates", nombre_original)
                
                # Si ya existe, agregar timestamp
                if os.path.exists(destino):
                    nombre_base, ext = os.path.splitext(nombre_original)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    nuevo_nombre = f"{nombre_base}_{timestamp}{ext}"
                    destino = os.path.join("training_plates", nuevo_nombre)
                    duplicadas += 1
                
                # Copiar archivo
                import shutil
                shutil.copy2(archivo, destino)
                copiadas += 1
                
            except Exception as e:
                print(f"Error copiando {archivo}: {e}")
        
        # Recargar lista de imágenes
        self.items = _collect_training_images()
        
        # Ir a la primera imagen nueva (al final de la lista)
        if self.items:
            self.idx = len(self.items) - copiadas
            if self.idx < 0:
                self.idx = 0
        
        # Actualizar UI
        self._show()
        
        msg = f"✅ {copiadas} imágenes copiadas a training_plates/\n\n"
        if duplicadas > 0:
            msg += f"⚠️  {duplicadas} archivos renombrados (ya existían)\n\n"
        msg += f"Total de imágenes ahora: {len(self.items)}"
        
        messagebox.showinfo("Imágenes cargadas", msg, parent=self.win)

    def _summary(self):
        total    = len(self.items)
        reviewed = sum(1 for i in self.items if i["path"] in self.labels)
        correct  = sum(1 for v in self.labels.values() if v.get("status")=="correct")
        wrong    = sum(1 for v in self.labels.values() if v.get("status")=="incorrect")
        fixed    = sum(1 for v in self.labels.values() if v.get("status")=="corrected")
        acc      = 100*correct/max(reviewed,1)
        messagebox.showinfo("Resumen",
            f"Total imágenes : {total}\n"
            f"Revisadas      : {reviewed} ({100*reviewed//max(total,1)}%)\n\n"
            f"✅ Correctas   : {correct} ({acc:.1f}%)\n"
            f"❌ Incorrectas : {wrong}\n"
            f"✏️  Corregidas  : {fixed}\n\n"
            f"Precisión OCR estimada: {acc:.1f}%\n"
            f"Archivo: {LABELS_FILE}", parent=self.win)

    def _close(self):
        _save_labels(self.labels); self.win.destroy()


class LPRApp:
    def __init__(self, root: tk.Tk):
        global lpr_app_instance
        lpr_app_instance = self
        self.root = root
        self.root.title("Sistema LPR - Categoria B | Mexico")
        self.root.geometry("1200x680")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.running         = True
        self.cap             = None
        self.model_veh       = None
        self.model_plate     = None
        self.reader          = None
        self.last_plate_text = None
        self.prev_time       = time.time()
        self.frame_count     = 0
        self._display_frame  = None

        self.frame_queue  = queue.Queue(maxsize=3)    # Aumentado para múltiples placas
        self.result_queue = queue.Queue(maxsize=5)   # Aumentado para manejar más detecciones
        self.proc_thread  = None

        # Config
        self.show_rectangles        = config["show_rectangles"]
        self.show_text              = config["show_text"]
        self.show_fps               = config["show_fps"]
        self.resolution             = config["resolution"]
        self.auto_save_images       = config["auto_save_images"]
        self.process_every_n_frames = config["process_every_n_frames"]
        self.conf_threshold_vehicle = config["conf_threshold_vehicle"]
        self.conf_threshold_plate   = config["conf_threshold_plate"]
        self.detection_mode         = config["detection_mode"]
        self.ocr_lang               = config["ocr_lang"]
        self.validate_mx_format     = config["validate_mx_format"]

        # Metricas de sesion
        self._session_start    = datetime.now()
        self._total_detections = 0
        self._response_times   = deque(maxlen=500)  # FIX: usar deque con maxlen para evitar memory leak
        # Historiales para dashboard (ultimos 120 puntos)
        self._fps_history       = deque(maxlen=120)
        self._proc_ms_history   = deque(maxlen=120)
        self._veh_count_history = deque(maxlen=120)
        self._dashboard_win     = None

        # Contadores para update_ui (inicializar aquí en lugar de lazy en update_ui)
        self._last_cnn_check    = time.time()   # FIX: tiempo real en lugar de contador de frames
        self._trocr_was_ready   = False
        self._paddle_was_ready  = False

        # ── Caché de overlays para video fluido ──────────────────────────────
        self._cached_veh_boxes   = []
        self._cached_plate_boxes = []
        self._last_box_update    = 0.0
        self._BOX_LIFETIME       = 0.5  # FIX: reducido de 2.0 a 0.5s para eliminar "fantasmas" más rápido

        # ── Estado de modo video ──────────────────────────────────────────────
        self._video_training_mode = False
        self._video_saved_count   = 0
        self._video_prev_save     = True
        self._video_prev_validate = True

        self.setup_ui()
        self.root.after(200, self.init_models)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def setup_ui(self):
        mb = tk.Menu(self.root)
        self.root.config(menu=mb)

        m = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="Archivos", menu=m)
        m.add_command(label="Exportar detecciones",          command=export_to_excel)
        m.add_command(label="Exportar matriculas registradas",command=export_registered_plates)
        m.add_command(label="Exportar registros invalidos",  command=export_invalid_registrations)
        m.add_command(label="Importar lista CSV",            command=import_registered_plates_from_csv)
        m.add_separator()
        m.add_command(label="Cargar video",    command=self.cargar_video)
        m.add_command(label="Guardar captura", command=self.guardar_captura)
        m.add_separator()
        m.add_command(label="Salir", command=self.on_closing)

        m = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="Ver", menu=m)
        self.var_rect     = tk.BooleanVar(value=self.show_rectangles)
        self.var_text     = tk.BooleanVar(value=self.show_text)
        self.var_fps      = tk.BooleanVar(value=self.show_fps)
        self.var_autosave = tk.BooleanVar(value=self.auto_save_images)
        self.var_validate = tk.BooleanVar(value=self.validate_mx_format)
        m.add_checkbutton(label="Rectangulos",    variable=self.var_rect,     command=self.toggle_rectangles)
        m.add_checkbutton(label="Texto matricula",variable=self.var_text,     command=self.toggle_text)
        m.add_checkbutton(label="FPS",            variable=self.var_fps,      command=self.toggle_fps)
        m.add_separator()
        rm = tk.Menu(m, tearoff=0)
        m.add_cascade(label="Resolucion", menu=rm)
        self.res_var = tk.StringVar(value=self.resolution)
        for r in ["baja","media","alta"]:
            rm.add_radiobutton(label=r.capitalize(), variable=self.res_var,
                               value=r, command=self.change_resolution)
        m.add_separator()
        m.add_command(label="Intervalo de procesamiento", command=self.configurar_intervalo)

        m = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="Base de datos", menu=m)
        m.add_command(label="Ver matriculas registradas", command=view_registered_plates)
        m.add_command(label="Ver registros invalidos",    command=view_invalid_registrations)
        m.add_command(label="Limpiar detecciones antiguas",command=clear_old_detections)

        m = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="Configuracion", menu=m)
        m.add_command(label="Seleccionar camara",         command=self.seleccionar_camara)
        m.add_command(label="Umbrales de confianza",      command=self.configurar_umbrales)
        m.add_command(label="Carpeta de imagenes",        command=self.cambiar_carpeta_imagenes)
        m.add_checkbutton(label="Guardado automatico",    variable=self.var_autosave, command=self.toggle_autosave)
        m.add_checkbutton(label="Validar formato MX",     variable=self.var_validate, command=self.toggle_validate)
        m.add_command(label="Tiempo anti-duplicado",      command=self.configurar_duplicate_timeout)
        m.add_command(label="Modo solo deteccion",        command=self.toggle_detection_mode)
        m.add_command(label="Registrar matricula manual", command=self.registrar_manual)
        m.add_separator()
        self.var_heavy_ocr = tk.BooleanVar(value=config.get("use_heavy_ocr", False))
        m.add_checkbutton(label="Motores OCR avanzados (TrOCR/Paddle)",
                          variable=self.var_heavy_ocr,
                          command=self.toggle_heavy_ocr)

        # ── Menú Entrenamiento (UNIFICADO) ───────────────────────────────────
        m = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="Entrenamiento", menu=m)
        m.add_command(label="Validación manual de datos",
                      command=self.abrir_validacion_manual)
        m.add_separator()
        m.add_command(label="Ver event_images/",
                      command=lambda: os.startfile(IMAGE_FOLDER)
                      if os.path.isdir(IMAGE_FOLDER) else None)
        m.add_command(label="Ver training_plates/",
                      command=lambda: os.startfile("training_plates")
                      if os.path.isdir("training_plates") else None)
        m.add_separator()
        m.add_command(label="Ver banco de muestras",
                      command=lambda: messagebox.showinfo(
                          "Banco",
                          _get_bank_info()))

        # ── Menú OCR Learning (NUEVO) ────────────────────────────────────────
        m = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="🧠 OCR Learning", menu=m)
        m.add_command(label="📊 Ver estadísticas",
                      command=self.ver_ocr_learning_stats)
        m.add_command(label="🔄 Reentrenar ahora",
                      command=self.reentrenar_ocr_learning)
        m.add_command(label="🗑️ Limpiar cache OCR",
                      command=self.limpiar_cache_ocr)
        m.add_separator()
        m.add_command(label="📖 Ver correcciones aprendidas",
                      command=self.ver_correcciones_aprendidas)
        m.add_command(label="📝 Abrir training_labels.json",
                      command=lambda: os.startfile("training_labels.json")
                      if os.path.exists("training_labels.json") else
                      messagebox.showinfo("Info", "No hay datos de entrenamiento aún"))

        m = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="Ayuda", menu=m)
        m.add_command(label="Acerca de",       command=self.acerca_de)
        m.add_command(label="Estadisticas",    command=self.estadisticas)
        m.add_command(label="Metricas sesion", command=self.ver_metricas)
        m.add_command(label="Dashboard",       command=self.abrir_dashboard)

                    # Barra superior
        top = tk.Frame(self.root, bg="#1e1e2e", height=44)
        top.pack(fill="x")
        lbl_style = {"bg":"#1e1e2e"}  # sin fg para evitar conflictos

        self.lbl_cam    = tk.Label(top, text="CAM: --",      **lbl_style, font=("Consolas",10), fg="#cdd6f4")
        self.lbl_fps    = tk.Label(top, text="FPS: --",      **lbl_style, font=("Consolas",10), fg="#cdd6f4")
        self.lbl_detect = tk.Label(top, text="VEH: No",      **lbl_style, font=("Consolas",10), fg="#cdd6f4")
        self.lbl_plate  = tk.Label(top, text="PLACA: --",    **lbl_style, font=("Consolas",12,"bold"), fg="#cdd6f4")
        self.lbl_state  = tk.Label(top, text="ESTADO: --",   **lbl_style, font=("Consolas",10), fg="#cdd6f4")
        self.lbl_status = tk.Label(top, text="Iniciando...", **lbl_style, font=("Consolas",10), fg="#fab387")
        self.lbl_invalid= tk.Label(top, text="",             **lbl_style, font=("Consolas",10), fg="#f38ba8")
        gpu_txt  = f"GPU:{torch.cuda.get_device_name(0)[:12]}" if GPU_AVAILABLE else "CPU"
        self.lbl_gpu    = tk.Label(top, text=gpu_txt,        **lbl_style, font=("Consolas",10),
                                   fg="#a6e3a1" if GPU_AVAILABLE else "#f38ba8")

        for w in (self.lbl_cam, self.lbl_fps, self.lbl_detect,
                  self.lbl_plate, self.lbl_state, self.lbl_gpu,
                  self.lbl_status, self.lbl_invalid):
            w.pack(side="left", padx=10)

        tk.Button(top, text="Dashboard", command=self.abrir_dashboard,
                  bg="#45475a", fg="white", relief="flat").pack(side="right", padx=6)
        tk.Button(top, text="Buscar", command=self.buscar_placa,
                  bg="#313244", fg="white", relief="flat").pack(side="right", padx=6)
        tk.Button(top, text="Excel", command=export_to_excel,
                  bg="#313244", fg="white", relief="flat").pack(side="right", padx=6)
        # Contenido
        main = tk.Frame(self.root, bg="#181825")
        main.pack(fill="both", expand=True)

        lf = tk.Frame(main, bg="black")
        lf.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        self.video_label = tk.Label(lf, bg="black")
        self.video_label.pack(fill="both", expand=True)

        rf = tk.Frame(main, bg="#1e1e2e", width=300)
        rf.pack(side="right", fill="y", padx=6, pady=6)
        rf.pack_propagate(False)
        tk.Label(rf, text="DETECCIONES RECIENTES",
                 font=("Consolas",9,"bold"), bg="#1e1e2e", fg="#89b4fa").pack(pady=6)
        self.listbox = tk.Listbox(rf, width=38, font=("Consolas",8),
                                  bg="#181825", fg="#cdd6f4",
                                  selectbackground="#313244")
        self.listbox.pack(fill="both", expand=True, padx=6)
        tk.Button(rf, text="Registrar ultima placa",
                  command=self.registrar_ultima_placa,
                  bg="#313244",fg="white",relief="flat").pack(pady=6)

        self.refresh_detections_list()

    # ------------------------------------------------------------------
    # Toggles
    # ------------------------------------------------------------------
    def toggle_rectangles(self):
        self.show_rectangles = self.var_rect.get()
        config["show_rectangles"] = self.show_rectangles; save_config(config)
    def toggle_text(self):
        self.show_text = self.var_text.get()
        config["show_text"] = self.show_text; save_config(config)
    def toggle_fps(self):
        self.show_fps = self.var_fps.get()
        config["show_fps"] = self.show_fps; save_config(config)
    def toggle_autosave(self):
        self.auto_save_images = self.var_autosave.get()
        config["auto_save_images"] = self.auto_save_images; save_config(config)
    def toggle_validate(self):
        self.validate_mx_format = self.var_validate.get()
        config["validate_mx_format"] = self.validate_mx_format; save_config(config)
    def change_resolution(self):
        self.resolution = self.res_var.get()
        config["resolution"] = self.resolution; save_config(config)
        self._apply_resolution()
    def _apply_resolution(self):
        if not self.cap: return
        # iPhone 11 vía Camo soporta hasta 1080p — aprovechar resolución alta
        sizes = {
            "baja":  (640,  480),
            "media": (1280, 720),
            "alta":  (1920, 1080),
        }
        cw, ch = sizes.get(self.resolution, (1280, 720))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cw)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)
        # Pedir 30fps — Camo lo soporta
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Deshabilitar autoexposición para evitar parpadeo en reflejos
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual en DSHOW
    def configurar_intervalo(self):
        v = simpledialog.askinteger("Intervalo","Procesar cada N frames (1-10):",
            initialvalue=self.process_every_n_frames,minvalue=1,maxvalue=10)
        if v:
            self.process_every_n_frames = v
            config["process_every_n_frames"] = v; save_config(config)
    def configurar_duplicate_timeout(self):
        v = simpledialog.askinteger("Anti-duplicado",
            "Segundos entre registros de la misma placa:",
            initialvalue=DUPLICATE_TIMEOUT,minvalue=1,maxvalue=120)
        if v: set_duplicate_timeout(v)
    def abrir_validador(self):
        TrainingValidatorWindow(self.root)

    def toggle_detection_mode(self):
        self.detection_mode = "solo_deteccion" if self.detection_mode=="full" else "full"
        lbl = "solo deteccion" if self.detection_mode=="solo_deteccion" else "completo (OCR)"
        messagebox.showinfo("Modo",f"Modo {lbl} activado.")
        config["detection_mode"] = self.detection_mode; save_config(config)
    def toggle_heavy_ocr(self):
        """Activa/desactiva TrOCR y PaddleOCR. Requiere reiniciar para tomar efecto."""
        val = self.var_heavy_ocr.get()
        config["use_heavy_ocr"] = val
        save_config(config)
        if val:
            # Iniciar carga si aún no están listos
            if TROCR_AVAILABLE:
                get_trocr_recognizer().load_async()
            if PADDLE_AVAILABLE:
                get_paddle_recognizer().load_async()
            messagebox.showinfo("OCR avanzado",
                "TrOCR/PaddleOCR activados.\nSe cargarán en segundo plano.\n"
                "Nota: consumen VRAM adicional — puede bajar FPS en GTX 1650.")
        else:
            messagebox.showinfo("OCR avanzado",
                "TrOCR/PaddleOCR desactivados.\nLos modelos se liberarán al reiniciar.")
    def registrar_manual(self):
        p = simpledialog.askstring("Registrar","Matricula a registrar:")
        if p:
            register_new_plate(p.strip().upper(), source="manual")
            self.refresh_detections_list()
    def configurar_umbrales(self):
        win = tk.Toplevel(self.root); win.title("Umbrales"); win.geometry("300x170")
        tk.Label(win,text="Umbral vehiculos (0-1):").pack(pady=4)
        ve = tk.Entry(win); ve.insert(0,str(self.conf_threshold_vehicle)); ve.pack()
        tk.Label(win,text="Umbral placas (0-1):").pack(pady=4)
        pe = tk.Entry(win); pe.insert(0,str(self.conf_threshold_plate)); pe.pack()
        def guardar():
            try:
                self.conf_threshold_vehicle = float(ve.get())
                self.conf_threshold_plate   = float(pe.get())
                config["conf_threshold_vehicle"] = self.conf_threshold_vehicle
                config["conf_threshold_plate"]   = self.conf_threshold_plate
                save_config(config); win.destroy()
            except ValueError:
                messagebox.showerror("Error","Valores numericos validos.")
        tk.Button(win,text="Guardar",command=guardar).pack(pady=10)
    def cambiar_carpeta_imagenes(self):
        global IMAGE_FOLDER
        d = filedialog.askdirectory()
        if d:
            IMAGE_FOLDER = d; os.makedirs(IMAGE_FOLDER,exist_ok=True)
            config["image_folder"] = d; save_config(config)
            messagebox.showinfo("Carpeta",f"Carpeta: {d}")

    def notify_invalid(self, plate_raw: str, reason: str):
        """Llamado desde cualquier hilo para notificar registro invalido en UI."""
        self.root.after(0, lambda: self.lbl_invalid.config(
            text=f"INVALIDO: {plate_raw[:12]}"))
        self.root.after(4000, lambda: self.lbl_invalid.config(text=""))

    def abrir_dashboard(self):
        """Dashboard en tiempo real: FPS, tiempo de proceso, vehiculos, detecciones por estado."""
        if self._dashboard_win and self._dashboard_win.winfo_exists():
            self._dashboard_win.lift(); return

        win = tk.Toplevel(self.root)
        win.title("Dashboard - LPR Sistema")
        win.geometry("860x540")
        win.configure(bg="#1e1e2e")
        self._dashboard_win = win

        BG, FG, GRID = "#1e1e2e", "#cdd6f4", "#313244"
        COLORS = {"fps":"#89b4fa","proc":"#fab387","veh":"#a6e3a1","det":"#f38ba8"}

        # Titulo
        tk.Label(win, text="DASHBOARD - TIEMPO REAL",
                 font=("Consolas",13,"bold"), bg=BG, fg="#89dceb").pack(pady=6)

        # Fila de KPIs
        kpi_frame = tk.Frame(win, bg=BG); kpi_frame.pack(fill="x", padx=12)
        self._kpi_fps   = self._make_kpi(kpi_frame, "FPS",       "--", COLORS["fps"])
        self._kpi_proc  = self._make_kpi(kpi_frame, "Proceso ms","--", COLORS["proc"])
        self._kpi_veh   = self._make_kpi(kpi_frame, "Vehiculos", "--", COLORS["veh"])
        self._kpi_det   = self._make_kpi(kpi_frame, "Detecciones","--",COLORS["det"])
        self._kpi_cache = self._make_kpi(kpi_frame, "Cache OCR", "--", "#cba6f7")
        gpu_lbl = "GPU" if GPU_AVAILABLE else "CPU"
        self._kpi_gpu   = self._make_kpi(kpi_frame, "Dispositivo", gpu_lbl, "#a6e3a1" if GPU_AVAILABLE else "#f38ba8")

        # Canvas para graficas
        graph_frame = tk.Frame(win, bg=BG); graph_frame.pack(fill="both", expand=True, padx=12, pady=6)

        self._canvas_fps  = self._make_graph(graph_frame, "FPS",          COLORS["fps"],  0, 0)
        self._canvas_proc = self._make_graph(graph_frame, "Proceso (ms)", COLORS["proc"], 0, 1)
        self._canvas_veh  = self._make_graph(graph_frame, "Vehiculos",    COLORS["veh"],  1, 0)
        self._canvas_state= self._make_bar_graph(graph_frame, 1, 1)

        graph_frame.columnconfigure(0, weight=1)
        graph_frame.columnconfigure(1, weight=1)
        graph_frame.rowconfigure(0, weight=1)
        graph_frame.rowconfigure(1, weight=1)

        self._dashboard_running = True
        self._update_dashboard()

    def _make_kpi(self, parent, label, value, color):
        f = tk.Frame(parent, bg="#313244", padx=10, pady=6)
        f.pack(side="left", expand=True, fill="x", padx=4)
        tk.Label(f, text=label, font=("Consolas",8), bg="#313244", fg="#6c7086").pack()
        lbl = tk.Label(f, text=value, font=("Consolas",14,"bold"), bg="#313244", fg=color)
        lbl.pack()
        return lbl

    def _make_graph(self, parent, title, color, row, col):
        f = tk.Frame(parent, bg="#181825", padx=4, pady=4)
        f.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        tk.Label(f, text=title, font=("Consolas",8), bg="#181825", fg="#6c7086").pack()
        c = tk.Canvas(f, bg="#181825", highlightthickness=0)
        c.pack(fill="both", expand=True)
        c._color = color
        c._title = title
        return c

    def _make_bar_graph(self, parent, row, col):
        f = tk.Frame(parent, bg="#181825", padx=4, pady=4)
        f.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        tk.Label(f, text="Detecciones por Estado (hoy)", font=("Consolas",8),
                 bg="#181825", fg="#6c7086").pack()
        c = tk.Canvas(f, bg="#181825", highlightthickness=0)
        c.pack(fill="both", expand=True)
        return c

    def _draw_line_graph(self, canvas, data: deque, color: str):
        canvas.delete("all")
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10 or len(data) < 2: return
        pts = list(data)
        mn, mx = min(pts), max(pts)
        rng = mx - mn if mx != mn else 1
        pad = 8
        def px(i): return pad + (i / (len(pts)-1)) * (cw - 2*pad)
        def py(v): return ch - pad - ((v - mn) / rng) * (ch - 2*pad)
        # Grid lines
        for i in range(4):
            y = pad + i * (ch - 2*pad) / 3
            canvas.create_line(pad, y, cw-pad, y, fill="#313244", dash=(2,4))
        # Line
        coords = []
        for i, v in enumerate(pts):
            coords += [px(i), py(v)]
        canvas.create_line(*coords, fill=color, width=2, smooth=True)
        # Last value label
        canvas.create_text(cw-pad, pad, text=f"{pts[-1]:.1f}",
                           fill=color, anchor="ne", font=("Consolas",8))

    def _draw_bar_graph(self, canvas):
        canvas.delete("all")
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10: return
        try:
            with sqlite3.connect(DB_NAME) as conn:
                rows = conn.execute(
                    "SELECT state, COUNT(*) as n FROM detections "
                    "WHERE DATE(timestamp)=DATE('now') AND state IS NOT NULL "
                    "GROUP BY state ORDER BY n DESC LIMIT 8"
                ).fetchall()
        except Exception:
            return
        if not rows: return
        bar_colors = ["#89b4fa","#a6e3a1","#fab387","#f38ba8",
                      "#cba6f7","#89dceb","#f9e2af","#94e2d5"]
        max_n = max(r[1] for r in rows)
        pad, gap = 8, 4
        bar_w = (cw - 2*pad - gap*(len(rows)-1)) / len(rows)
        for i, (state, n) in enumerate(rows):
            x0 = pad + i*(bar_w+gap)
            bar_h = (n/max_n) * (ch - 30)
            y0 = ch - 20 - bar_h
            color = bar_colors[i % len(bar_colors)]
            canvas.create_rectangle(x0, y0, x0+bar_w, ch-20, fill=color, outline="")
            canvas.create_text(x0+bar_w/2, ch-10, text=(state or "?")[:6],
                               fill="#cdd6f4", font=("Consolas",7), anchor="center")
            canvas.create_text(x0+bar_w/2, y0-2, text=str(n),
                               fill=color, font=("Consolas",7), anchor="s")

    def _update_dashboard(self):
        if not (self._dashboard_win and self._dashboard_win.winfo_exists()):
            return
        # KPIs
        if self._fps_history:
            self._kpi_fps.config(text=f"{list(self._fps_history)[-1]:.1f}")
        if self._proc_ms_history:
            self._kpi_proc.config(text=f"{list(self._proc_ms_history)[-1]:.0f}")
        if self._veh_count_history:
            self._kpi_veh.config(text=str(list(self._veh_count_history)[-1]))
        self._kpi_det.config(text=str(self._total_detections))
        self._kpi_cache.config(text=str(len(_ocr_cache._cache)))
        # Graficas
        self._draw_line_graph(self._canvas_fps,  self._fps_history,      "#89b4fa")
        self._draw_line_graph(self._canvas_proc, self._proc_ms_history,  "#fab387")
        self._draw_line_graph(self._canvas_veh,  self._veh_count_history,"#a6e3a1")
        self._draw_bar_graph(self._canvas_state)
        self._dashboard_win.after(500, self._update_dashboard)

    def acerca_de(self):
        enc_status    = "Fernet AES-128" if _fernet else "Sin encriptacion (instalar cryptography)"
        trocr_status  = "Activo" if get_trocr_recognizer().available else (
                        "Cargando..." if TROCR_AVAILABLE else "No instalado (pip install transformers)")
        paddle_status = "Activo" if get_paddle_recognizer().available else (
                        "Cargando..." if PADDLE_AVAILABLE else "No instalado (pip install paddlepaddle paddleocr)")
        plate_model   = "best.pt (YOLOv11/custom)" if os.path.exists("best.pt") else "plate_model.pt"
        messagebox.showinfo("Acerca de",
            f"Sistema LPR v5.0 - Categoria B\n"
            f"Mexico: 32 estados\n\n"
            f"── Deteccion ──\n"
            f"  Vehiculos : YOLOv8n\n"
            f"  Placas    : {plate_model}\n\n"
            f"── OCR (4 motores) ──\n"
            f"  CNN propia : {'Activa' if get_char_recognizer().available else 'No disponible'}\n"
            f"  EasyOCR    : Activo\n"
            f"  TrOCR      : {trocr_status}\n"
            f"  PaddleOCR  : {paddle_status}\n\n"
            f"── Sistema ──\n"
            f"  Encriptacion : {enc_status}\n"
            f"  Arquitectura : 3 hilos (captura/proceso/UI)")

    def estadisticas(self):
        try:
            with sqlite3.connect(DB_NAME) as conn:
                total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
                hoy   = conn.execute(
                    "SELECT COUNT(*) FROM detections WHERE DATE(timestamp)=DATE('now')"
                ).fetchone()[0]
                reg   = conn.execute(
                    "SELECT COUNT(*) FROM registered_plates WHERE status='active'"
                ).fetchone()[0]
                inv   = conn.execute(
                    "SELECT COUNT(*) FROM invalid_registrations"
                ).fetchone()[0]
                by_state = conn.execute(
                    "SELECT state,COUNT(*) FROM registered_plates "
                    "WHERE status='active' GROUP BY state ORDER BY COUNT(*) DESC LIMIT 5"
                ).fetchall()
            state_txt = "\n".join(f"  {s}: {n}" for s,n in by_state)
            messagebox.showinfo("Estadisticas",
                f"Total detecciones: {total}\n"
                f"Hoy: {hoy}\n"
                f"Matriculas activas: {reg}\n"
                f"Registros invalidos: {inv}\n"
                f"Conteo trafico: {traffic_counter}\n\n"
                f"Top estados registrados:\n{state_txt}")
        except Exception as e:
            messagebox.showerror("Error",str(e))

    def ver_metricas(self):
        avg_ms = (sum(self._response_times)/len(self._response_times)
                  if self._response_times else 0)
        dur = datetime.now() - self._session_start
        messagebox.showinfo("Metricas de sesion",
            f"Duracion sesion: {str(dur).split('.')[0]}\n"
            f"Detecciones sesion: {self._total_detections}\n"
            f"Tiempo respuesta promedio: {avg_ms:.1f} ms\n"
            f"Encriptacion activa: {'Si' if _fernet else 'No'}\n"
            f"Matriculas en cache: {len(registered_plates_set)}")

    # ------------------------------------------------------------------
    # OCR Learning
    # ------------------------------------------------------------------
    def ver_ocr_learning_stats(self):
        """Muestra estadísticas del sistema de aprendizaje OCR."""
        if not OCR_LEARNING_AVAILABLE:
            messagebox.showwarning("OCR Learning",
                "Sistema de aprendizaje no disponible.\n"
                "Verifica que ocr_learning.py exista.")
            return
        
        try:
            ocr_learning = get_ocr_learning()
            stats = ocr_learning.stats
            
            # Top patrones aprendidos
            all_corrections = []
            for wrong, corrections in ocr_learning.char_patterns.items():
                for right, count in corrections.items():
                    all_corrections.append((wrong, right, count))
            all_corrections.sort(key=lambda x: x[2], reverse=True)
            
            patterns_text = "\n".join(
                f"  '{w}' → '{r}': {c}x"
                for w, r, c in all_corrections[:10]
            ) if all_corrections else "  (ninguno)"
            
            # Correcciones exactas
            exact_corrections = len(ocr_learning.corrections)
            
            messagebox.showinfo("📊 OCR Learning - Estadísticas",
                f"🎯 Correcciones exactas: {exact_corrections}\n"
                f"🔤 Patrones de caracteres: {stats.get('char_corrections', 0)}\n"
                f"📚 Total validaciones: {stats.get('total_corrections', 0)}\n\n"
                f"Top 10 patrones aprendidos:\n{patterns_text}\n\n"
                f"💡 Cada validación manual mejora el sistema automáticamente.")
        except Exception as e:
            messagebox.showerror("Error", f"Error obteniendo estadísticas:\n{e}")
    
    def reentrenar_ocr_learning(self):
        """Fuerza reentrenamiento del OCR Learning desde training_labels.json."""
        if not OCR_LEARNING_AVAILABLE:
            messagebox.showwarning("OCR Learning", "Sistema no disponible")
            return
        
        if not os.path.exists("training_labels.json"):
            messagebox.showinfo("OCR Learning",
                "No hay datos de entrenamiento.\n\n"
                "Usa 'Entrenamiento → Validación manual' para crear datos.")
            return
        
        try:
            print("\n" + "="*70)
            print("  REENTRENANDO OCR LEARNING")
            print("="*70)
            
            ocr_learning = get_ocr_learning()
            corrections_before = ocr_learning.stats.get('unique_errors', 0)
            
            # Recargar desde archivo
            loaded = ocr_learning.load_from_training_data()
            ocr_learning.save_cache()
            
            corrections_after = ocr_learning.stats.get('unique_errors', 0)
            
            # Limpiar cache OCR para aplicar nuevas correcciones
            global _ocr_cache
            _ocr_cache = OCRCache(maxsize=512)
            
            print("="*70 + "\n")
            
            messagebox.showinfo("✅ Reentrenamiento completo",
                f"Datos procesados: {loaded}\n"
                f"Correcciones antes: {corrections_before}\n"
                f"Correcciones ahora: {corrections_after}\n"
                f"Nuevas: {corrections_after - corrections_before}\n\n"
                f"Cache OCR limpiado para aplicar cambios.")
        except Exception as e:
            messagebox.showerror("Error", f"Error reentrenando:\n{e}")
    
    def limpiar_cache_ocr(self):
        """Limpia el cache de OCR para forzar nuevas lecturas."""
        global _ocr_cache
        _ocr_cache = OCRCache(maxsize=512)
        messagebox.showinfo("Cache OCR",
            "Cache limpiado exitosamente.\n\n"
            "Las próximas detecciones usarán OCR actualizado.")
    
    def ver_correcciones_aprendidas(self):
        """Muestra lista detallada de correcciones aprendidas."""
        if not OCR_LEARNING_AVAILABLE:
            messagebox.showwarning("OCR Learning", "Sistema no disponible")
            return
        
        try:
            ocr_learning = get_ocr_learning()
            
            if not ocr_learning.corrections:
                messagebox.showinfo("Correcciones",
                    "No hay correcciones aprendidas aún.\n\n"
                    "Usa 'Entrenamiento → Validación manual' para entrenar.")
                return
            
            # Crear ventana con lista
            win = tk.Toplevel(self.root)
            win.title("📖 Correcciones Aprendidas")
            win.geometry("600x500")
            win.configure(bg="#1e1e2e")
            
            tk.Label(win, text="Correcciones Exactas Aprendidas",
                     font=("Consolas", 12, "bold"),
                     bg="#1e1e2e", fg="#89b4fa").pack(pady=10)
            
            # Frame con scrollbar
            frame = tk.Frame(win, bg="#1e1e2e")
            frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            scrollbar = tk.Scrollbar(frame)
            scrollbar.pack(side="right", fill="y")
            
            text_widget = tk.Text(frame, font=("Consolas", 10),
                                  bg="#181825", fg="#cdd6f4",
                                  yscrollcommand=scrollbar.set,
                                  wrap="word")
            text_widget.pack(side="left", fill="both", expand=True)
            scrollbar.config(command=text_widget.yview)
            
            # Agregar correcciones
            text_widget.insert("1.0", f"Total: {len(ocr_learning.corrections)} correcciones\n")
            text_widget.insert("end", "="*60 + "\n\n")
            
            for wrong, correct in sorted(ocr_learning.corrections.items()):
                text_widget.insert("end", f"  '{wrong}' → '{correct}'\n")
            
            text_widget.config(state="disabled")
            
            tk.Button(win, text="Cerrar", command=win.destroy,
                      bg="#313244", fg="white",
                      font=("Consolas", 10)).pack(pady=10)
        except Exception as e:
            messagebox.showerror("Error", f"Error mostrando correcciones:\n{e}")

    # ------------------------------------------------------------------
    # Modelos
    # ------------------------------------------------------------------
    def init_models(self):
        self.lbl_status.config(text="Cargando modelos...", fg="#fab387")
        threading.Thread(target=self._load_models_thread, daemon=True).start()

    def _load_models_thread(self):
        # ── Detección de placas: preferir best.pt (YOLOv11/v8 custom)
        # Orden de preferencia: best.pt → plate_model.pt
        plate_model_path = "best.pt" if os.path.exists("best.pt") else "plate_model.pt"

        try:
            mv = YOLO("yolov8n.pt")
            mp = YOLO(plate_model_path)
            if GPU_AVAILABLE:
                mv.to(GPU_DEVICE)
                mp.to(GPU_DEVICE)
                print(f"[YOLO] Modelos en GPU: {GPU_DEVICE}")
            else:
                print("[YOLO] Modelos en CPU")
            print(f"[YOLO] Modelo de placas: {plate_model_path}")
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"YOLO: {e}"))
            return

        try:
            # FIX: forzar 'en' para placas — 'es' incluye modelos de español
            # que no aportan nada para alfanumérico de placas y son más lentos.
            ocr_lang_effective = "en"
            rd = easyocr.Reader(
                [ocr_lang_effective],
                gpu=GPU_AVAILABLE,
                verbose=False,
            )
            print(f"[EasyOCR] idioma='en' en {'GPU' if GPU_AVAILABLE else 'CPU'}")
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"EasyOCR: {e}"))
            return

        self.model_veh   = mv
        self.model_plate = mp
        self.reader      = rd

        # ── TrOCR y PaddleOCR: solo cargar si use_heavy_ocr=True ─────────────
        # En GTX 1650 (4GB VRAM) compiten con YOLO y reducen FPS significativamente.
        # Se pueden activar desde Configuracion → "Motores OCR avanzados".
        use_heavy = config.get("use_heavy_ocr", False)
        if use_heavy:
            if TROCR_AVAILABLE:
                get_trocr_recognizer().load_async()
            else:
                print("[TrOCR] No disponible — pip install transformers")
            if PADDLE_AVAILABLE:
                get_paddle_recognizer().load_async()
            else:
                print("[PaddleOCR] No disponible — pip install paddlepaddle paddleocr")
        else:
            print("[OCR] TrOCR/PaddleOCR desactivados (use_heavy_ocr=False). "
                  "Activa desde Configuracion si tienes VRAM disponible.")

        self.root.after(0, self._models_ready)

    def _models_ready(self):
        self.lbl_status.config(text="Listo", fg="#a6e3a1")
        self.abrir_camara(config["camera_index"])
        # Iniciar update_ui después de que la cámara esté lista
        self.root.after(100, self.update_ui)

    # ------------------------------------------------------------------
    # Camara
    # ------------------------------------------------------------------
    def abrir_camara(self, indice: int = 0):
        if self.cap:
            self.cap.release(); self.cap = None

        cap = None
        # Probar backends en orden: sin backend → DSHOW → MSMF
        backends = [
            (cv2.CAP_ANY,   "AUTO"),
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_MSMF,  "MSMF"),
        ]
        used_backend = "?"
        for backend, name in backends:
            try:
                c = cv2.VideoCapture(indice, backend)
                if not c.isOpened():
                    c.release(); continue
                # Verificar que realmente entrega frames
                try:
                    ret, frame = c.read()
                    if ret and frame is not None and frame.size > 0:
                        cap = c
                        used_backend = name
                        break
                except Exception as e:
                    print(f"[Camara] Error leyendo frame con {name}: {e}")
                c.release()
            except Exception as e:
                print(f"[Camara] Error abriendo con {name}: {e}")
                continue

        if cap is None:
            print(f"[Camara] No se pudo abrir cámara {indice} con ningún backend")
            messagebox.showwarning("Cámara no disponible",
                f"No se pudo abrir cámara {indice}.\n\n"
                f"Puedes:\n"
                f"  • Cargar un video desde el menú\n"
                f"  • Verificar que la cámara esté conectada\n"
                f"  • Probar otro índice de cámara\n\n"
                f"Si usas Camo Studio:\n"
                f"  1. Abre Camo Studio en PC\n"
                f"  2. Conecta iPhone por USB o WiFi\n"
                f"  3. Prueba índices 0, 1 o 2")
            return

        self.cap = cap
        self._apply_resolution()

        # Descartar primeros frames (Camo tarda en estabilizar)
        # FIX: Agregar try-catch para evitar crash si la cámara no está lista
        for _ in range(10):
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break  # Cámara no lista, salir del loop
            except Exception as e:
                print(f"[Camara] Error descartando frames iniciales: {e}")
                break

        real_w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = self.cap.get(cv2.CAP_PROP_FPS)

        cam_label = f"CAM:{indice} {real_w}x{real_h} @{real_fps:.0f}fps [{used_backend}]"
        self.lbl_cam.config(text=cam_label)
        print(f"[Camara] {cam_label}")

        self._start_processing_thread()
        self._start_capture_loop()

    def _start_processing_thread(self):
        if self.proc_thread and self.proc_thread.is_alive():
            self.proc_thread.stop()
            # FIX: enviar sentinel None para desbloquear frame_queue.get()
            # y esperar a que el hilo termine antes de crear uno nuevo.
            # Sin join(), el hilo viejo puede seguir procesando frames del
            # nuevo hilo, causando condiciones de carrera en los trackers.
            try: self.frame_queue.put_nowait(None)
            except queue.Full: pass
            self.proc_thread.join(timeout=2.0)  # esperar máx 2s
        for q in [self.frame_queue, self.result_queue]:
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break
        self.proc_thread = ProcessingThread(
            self.frame_queue, self.result_queue,
            self.model_veh, self.model_plate, self.reader, self)
        self.proc_thread.start()

    def cargar_video(self):
        fp = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v")])
        if not fp:
            return

        # FIX CRÍTICO: Liberar video anterior ANTES de abrir el nuevo
        # Esto evita el crash "Assertion fctx->async_lock failed"
        if self.cap:
            try:
                print("[Video] Liberando video anterior...")
                self.cap.release()
                self.cap = None
                time.sleep(0.3)  # Dar tiempo a FFmpeg para limpiar completamente
            except Exception as e:
                print(f"[Video] Error liberando video anterior: {e}")
                self.cap = None

        # Preguntar si es para entrenamiento o solo visualización
        modo = messagebox.askyesno(
            "Modo de video",
            "¿Procesar video en modo ENTRENAMIENTO?\n\n"
            "SÍ  → Guarda todas las placas detectadas en event_images/\n"
            "       (alimenta el reentrenamiento automático)\n\n"
            "NO  → Solo visualizar detecciones, sin guardar\n"
        )

        # FIX CRÍTICO: Probar múltiples backends y estrategias
        # Algunos videos H.264 tienen problemas con MSMF y FFmpeg
        # Estrategia: MSMF → FFmpeg con threads=1 → FFmpeg normal → Auto
        
        cap = None
        backends = [
            (cv2.CAP_MSMF, "Windows Media Foundation (MSMF)", None),
            (cv2.CAP_FFMPEG, "FFmpeg (single thread)", "threads;1"),
            (cv2.CAP_FFMPEG, "FFmpeg (multi thread)", None),
            (cv2.CAP_ANY, "Auto", None),
        ]
        
        for backend, name, ffmpeg_opts in backends:
            try:
                # Configurar opciones específicas por backend
                if backend == cv2.CAP_FFMPEG and ffmpeg_opts:
                    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = ffmpeg_opts
                elif backend == cv2.CAP_FFMPEG:
                    # Limpiar opciones previas
                    os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS', None)
                
                test_cap = cv2.VideoCapture(fp, backend)
                if not test_cap.isOpened():
                    test_cap.release()
                    continue
                
                # Verificar que puede leer múltiples frames (no solo el primero)
                success_count = 0
                for i in range(5):  # Intentar leer 5 frames
                    try:
                        ret, test_frame = test_cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            success_count += 1
                        else:
                            break
                    except Exception as e:
                        print(f"[Video] Error leyendo frame {i} con {name}: {e}")
                        break
                
                # Si pudo leer al menos 3 de 5 frames, considerar exitoso
                if success_count >= 3:
                    test_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Volver al inicio
                    cap = test_cap
                    print(f"[Video] Backend: {name} ({success_count}/5 frames OK)")
                    break
                else:
                    print(f"[Video] Backend {name} falló: solo {success_count}/5 frames legibles")
                    test_cap.release()
                    
            except Exception as e:
                print(f"[Video] Backend {name} falló: {e}")
                try:
                    if test_cap:
                        test_cap.release()
                except:
                    pass
        
        if cap is None or not cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir el video.")
            return

        # Metadatos del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_s   = total_frames / fps_video
        mins, secs   = divmod(int(duration_s), 60)

        # Activar/desactivar guardado según modo
        self._video_training_mode = modo
        _prev_save     = self.auto_save_images
        _prev_validate = self.validate_mx_format
        if modo:
            self.auto_save_images  = True
            # En modo entrenamiento NO filtrar por formato MX —
            # queremos guardar TODAS las detecciones para aprender
            self.validate_mx_format = False
            print("[Video] Modo ENTRENAMIENTO activado — guardando en event_images/")
        else:
            self.auto_save_images = False

        # Guardar estado de video para restaurar al terminar
        self._video_prev_save     = _prev_save
        self._video_prev_validate = _prev_validate
        self._video_total_frames = total_frames
        self._video_fps          = fps_video
        self._video_saved_count  = 0

        # Asignar el nuevo VideoCapture
        self.cap = cap

        nombre = os.path.basename(fp)
        modo_str = "ENTRENAMIENTO" if modo else "VISUALIZACIÓN"
        self.lbl_cam.config(
            text=f"VIDEO [{modo_str}]: {nombre} ({mins}:{secs:02d})")
        print(f"[Video] {nombre} | {total_frames:,} frames | "
              f"{fps_video:.1f}fps | {mins}:{secs:02d} | modo={modo_str}")

        self._start_processing_thread()
        self._start_capture_loop()

    def guardar_captura(self):
        if self._display_frame is not None:
            fn = f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fn, self._display_frame)
            messagebox.showinfo("Guardar",f"Guardado: {fn}")
        else:
            messagebox.showwarning("Guardar","No hay frame disponible.")

    def seleccionar_camara(self):
        """
        Escanea cámaras disponibles (índices 0-5) y muestra un diálogo
        con las que están activas, para facilitar seleccionar Camo Studio.
        """
        self.lbl_status.config(text="Escaneando camaras...", fg="#fab387")
        self.root.update()

        disponibles = []
        for i in range(6):
            for backend, bname in [(cv2.CAP_ANY,""), (cv2.CAP_DSHOW," DSHOW")]:
                try:
                    c = cv2.VideoCapture(i, backend)
                    if not c.isOpened():
                        c.release(); continue
                    ret, frame = c.read()
                    if ret and frame is not None:
                        w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        disponibles.append(f"[{i}]  {w}x{h}{bname}")
                        c.release()
                        break
                    c.release()
                except Exception:
                    continue

        self.lbl_status.config(text="Listo", fg="#a6e3a1")

        if not disponibles:
            messagebox.showwarning("Camaras", "No se encontraron camaras.")
            return

        lista = "\n".join(disponibles)
        idx = simpledialog.askinteger(
            "Seleccionar camara",
            f"Camaras disponibles:\n{lista}\n\n"
            f"Camo Studio suele ser indice 1 o 2.\n\n"
            f"Ingresa el indice:",
            initialvalue=config["camera_index"],
            minvalue=0, maxvalue=9,
        )
        if idx is not None:
            config["camera_index"] = idx
            save_config(config)
            self.abrir_camara(idx)

    # ------------------------------------------------------------------
    # Bucle de captura (hilo separado)
    # ------------------------------------------------------------------
    def _start_capture_loop(self):
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        is_video    = False
        fps_video   = 30.0
        total_frames = 0
        consecutive_errors = 0  # Contador de errores consecutivos
        max_consecutive_errors = 10  # Máximo de errores antes de abortar

        if self.cap:
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_video    = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            # VideoCapture de archivo devuelve FRAME_COUNT > 0; cámara devuelve 0
            is_video = total_frames > 0

        # Para videos: respetar FPS original (no procesar más rápido de lo necesario)
        # Para cámara: sin delay (el hardware ya controla el FPS)
        frame_delay = (1.0 / fps_video) if is_video else 0.0
        last_frame_t = time.time()

        while self.running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                consecutive_errors = 0  # Reset contador si lectura exitosa
            except Exception as e:
                consecutive_errors += 1
                print(f"[Captura] Error leyendo frame ({consecutive_errors}/{max_consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print(f"[Captura] Demasiados errores consecutivos, terminando video")
                    if is_video:
                        self.root.after(0, self._on_video_finished)
                    break
                
                # Intentar saltar al siguiente frame
                if is_video and self.cap:
                    try:
                        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos + 1)
                        print(f"[Captura] Saltando frame corrupto en posición {int(current_pos)}")
                    except Exception:
                        pass
                
                time.sleep(0.05)
                continue

            if not ret or frame is None:
                consecutive_errors += 1
                
                if is_video:
                    # ── FIX: fin de video — antes hacía bucle infinito ────────
                    pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if total_frames > 0 and pos >= total_frames - 2:
                        # Video terminó normalmente
                        print(f"[Captura] Video terminado en frame {int(pos)}/{total_frames}")
                        self.root.after(0, self._on_video_finished)
                        break
                    
                    # Error de lectura transitorio (no fin)
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[Captura] Demasiados errores, terminando video")
                        self.root.after(0, self._on_video_finished)
                        break
                    
                    # Intentar saltar frame corrupto
                    try:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos + 1)
                        print(f"[Captura] Saltando frame no legible en posición {int(pos)}")
                    except Exception:
                        pass
                    
                    time.sleep(0.05)
                    continue
                else:
                    # Cámara: error transitorio
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[Captura] Cámara desconectada")
                        break
                    time.sleep(0.05)
                    continue

            # ── Control de velocidad para videos ─────────────────────────────
            # Sin este control, procesamos a velocidad de GPU (~200fps),
            # lo que no tiene sentido para videos de 30fps y satura las colas.
            if is_video and frame_delay > 0:
                elapsed = time.time() - last_frame_t
                wait    = frame_delay - elapsed
                if wait > 0.002:
                    time.sleep(wait)
                last_frame_t = time.time()

            # ── Progreso del video (cada 5%) ──────────────────────────────────
            if is_video and total_frames > 0:
                pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                pct = int(100 * pos / total_frames)
                if pos % max(1, total_frames // 20) == 0:
                    saved = getattr(self, "_video_saved_count", 0)
                    self.root.after(0, lambda p=pct, s=saved:
                        self.lbl_cam.config(
                            text=self.lbl_cam.cget("text").split("|")[0].strip()
                            + f" | {p}% | guardadas: {s}"))

            self.frame_count += 1

            # ── Control de brillo neuro-difuso ───────────────────────────────
            # FIX: Aplicar ajuste de brillo SOLO al frame de procesamiento,
            # FIX: Control de brillo eliminado — causaba titileo (flickering)
            # Usar frame original directamente
            proc_frame = frame

            if self.frame_count % self.process_every_n_frames != 0:
                # FIX: Usar frame original sin ajuste de brillo para consistencia visual
                display_pkt = {
                    "display":          frame.copy(),
                    "vehicle_detected": None,
                    "confirmed_plates": [],
                    "orig":             None,
                }
                try:
                    self.result_queue.put_nowait(display_pkt)
                except queue.Full:
                    try:    self.result_queue.get_nowait()
                    except queue.Empty: pass
                    try:    self.result_queue.put_nowait(display_pkt)
                    except queue.Full:  pass
                continue

            try:
                self.frame_queue.put_nowait((proc_frame.copy(), frame.copy()))
            except queue.Full:
                # FIX: Usar frame original sin ajuste de brillo para consistencia visual
                display_pkt = {
                    "display":          frame.copy(),
                    "vehicle_detected": None,
                    "confirmed_plates": [],
                    "orig":             None,
                }
                try:
                    self.result_queue.put_nowait(display_pkt)
                except queue.Full:
                    try:    self.result_queue.get_nowait()
                    except queue.Empty: pass
                    try:    self.result_queue.put_nowait(display_pkt)
                    except queue.Full:  pass

    def _on_video_finished(self):
        """Llamado desde el hilo de captura cuando el video termina."""
        saved = getattr(self, "_video_saved_count", 0)
        modo  = getattr(self, "_video_training_mode", False)

        # FIX: Liberar VideoCapture inmediatamente para evitar crash de FFmpeg
        if self.cap:
            try:
                self.cap.release()
                self.cap = None
                time.sleep(0.2)  # Dar tiempo a FFmpeg para limpiar threads
            except Exception as e:
                print(f"[Video] Error liberando capture: {e}")
                self.cap = None

        # Restaurar configuración anterior
        if hasattr(self, "_video_prev_save"):
            self.auto_save_images = self._video_prev_save
        if hasattr(self, "_video_prev_validate"):
            self.validate_mx_format = self._video_prev_validate

        msg = "Video terminado.\n\n"
        if modo:
            msg += ("Imagenes guardadas para entrenamiento: %d\n\n"
                    "Carpeta: event_images/\n\n"
                    "Puedes ejecutar train_char_model.py o\n"
                    "dejar que retrain_loop.py las detecte automaticamente." % saved)
        else:
            msg += "Modo visualizacion — no se guardaron imagenes."

        print("[Video] Terminado. Guardadas: %d | modo_entrenamiento=%s" % (saved, modo))
        messagebox.showinfo("Video terminado", msg)
        self.lbl_cam.config(text="Video terminado — Selecciona camara o nuevo video")

    def _draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        """
        Dibuja los marcadores de vehículo (verde) y placa (rojo/naranja/verde)
        sobre cada frame — sean procesados o no.

        Las cajas se ocultan automáticamente si no hubo detección en
        _BOX_LIFETIME segundos (el vehículo salió del encuadre).
        
        NUEVO: Dibuja líneas conectando cada placa con su vehículo padre.
        """
        # FIX: Si los caches están vacíos, retornar inmediatamente sin verificar timeout
        if not self._cached_veh_boxes and not self._cached_plate_boxes:
            return frame
        
        now = time.time()
        if now - self._last_box_update > self._BOX_LIFETIME:
            # Timeout alcanzado: limpiar caches y retornar frame limpio
            self._cached_veh_boxes   = []
            self._cached_plate_boxes = []
            return frame

        if self.show_rectangles:
            # ── Crear mapa de vehículos por ID para búsqueda rápida ──────────
            vehicle_map = {tid: (x1, y1, x2, y2) for (x1, y1, x2, y2, tid) in self._cached_veh_boxes}
            
            # ── Vehículos — rectángulo verde ─────────────────────────────────
            for (x1, y1, x2, y2, tid) in self._cached_veh_boxes:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
                cv2.putText(frame, f"V#{tid}", (x1, max(0,y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)

            # ── Placas — rectángulo según estado + línea al vehículo padre ───
            # Rojo    (0,0,255)   → detectada, sin texto OCR aún
            # Naranja (0,60,255)  → texto leído, placa no registrada
            # Verde   (0,180,0)   → placa registrada en BD
            for plate_data in self._cached_plate_boxes:
                # Desempaquetar con soporte para formato antiguo y nuevo
                if len(plate_data) == 6:
                    # Formato antiguo sin parent_vehicle_id
                    x1, y1, x2, y2, text, color = plate_data
                    parent_vid = None
                else:
                    # Formato nuevo con parent_vehicle_id
                    x1, y1, x2, y2, text, color, parent_vid = plate_data
                
                # Dibujar rectángulo de la placa
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                
                # ═══════════════════════════════════════════════════════════════
                # ANCLAJE VISUAL: Línea conectando placa con su vehículo padre
                # ═══════════════════════════════════════════════════════════════
                if parent_vid is not None and parent_vid in vehicle_map:
                    vx1, vy1, vx2, vy2 = vehicle_map[parent_vid]
                    
                    # Centro de la placa
                    plate_cx = (x1 + x2) // 2
                    plate_cy = (y1 + y2) // 2
                    
                    # Centro del vehículo
                    veh_cx = (vx1 + vx2) // 2
                    veh_cy = (vy1 + vy2) // 2
                    
                    # Dibujar línea punteada conectando placa → vehículo
                    # Color de la línea coincide con el color de la placa
                    self._draw_dashed_line(frame, 
                                          (plate_cx, plate_cy), 
                                          (veh_cx, veh_cy), 
                                          color, 
                                          thickness=2, 
                                          dash_length=10)
                
                # Dibujar texto de la placa
                if self.show_text and text:
                    # Fondo semitransparente para legibilidad
                    label     = format_mx_plate(text)
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                    tx, ty = x1, max(th + 4, y1 - 4)
                    cv2.rectangle(frame,
                                  (tx, ty - th - 4), (tx + tw + 4, ty + 2),
                                  (0, 0, 0), -1)
                    cv2.putText(frame, label, (tx + 2, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        return frame
    
    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=1, dash_length=10):
        """Dibuja una línea punteada entre dos puntos"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Calcular distancia total
        dist = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
        if dist < 1:
            return
        
        # Número de segmentos
        dashes = int(dist / dash_length)
        if dashes < 1:
            cv2.line(img, pt1, pt2, color, thickness)
            return
        
        # Dibujar segmentos alternados
        for i in range(dashes):
            if i % 2 == 0:  # Solo dibujar segmentos pares
                start_ratio = i / dashes
                end_ratio = min((i + 1) / dashes, 1.0)
                
                start_x = int(x1 + (x2 - x1) * start_ratio)
                start_y = int(y1 + (y2 - y1) * start_ratio)
                end_x = int(x1 + (x2 - x1) * end_ratio)
                end_y = int(y1 + (y2 - y1) * end_ratio)
                
                cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)

    # ------------------------------------------------------------------
    # Bucle UI (hilo principal via root.after)
    # ------------------------------------------------------------------
    def update_ui(self):
        if not self.running: return

        latest = None
        # FIX: acumular confirmed_plates de TODOS los resultados en cola,
        # no solo del último. Si solo tomamos el último, perdemos detecciones
        # confirmadas de frames intermedios que el hilo de procesamiento ya emitió.
        accumulated_plates = []
        while True:
            try:
                pkt = self.result_queue.get_nowait()
                # Acumular placas confirmadas de todos los paquetes
                if pkt.get("confirmed_plates"):
                    accumulated_plates.extend(pkt["confirmed_plates"])
                latest = pkt
            except queue.Empty:
                break
        # Inyectar las placas acumuladas en el paquete más reciente
        if latest is not None and accumulated_plates:
            latest["confirmed_plates"] = accumulated_plates

        if latest is not None:
            disp = latest["display"].copy()
            self._display_frame = disp

            # Actualizar caché de cajas (solo paquetes del ProcessingThread)
            if "vehicle_boxes" in latest:
                self._cached_veh_boxes   = latest["vehicle_boxes"]
                self._cached_plate_boxes = latest["plate_boxes"]
                self._last_box_update    = time.time()
                
                # FIX: Si no hay detecciones, limpiar inmediatamente los caches
                # para evitar "fantasmas" de recuadros cuando el vehículo ya pasó
                if not latest["vehicle_boxes"] and not latest["plate_boxes"]:
                    self._cached_veh_boxes   = []
                    self._cached_plate_boxes = []

            disp = self._draw_overlays(disp)

            # ══════════════════════════════════════════════════════════════
            # PASO 1 — RENDERIZAR SIEMPRE PRIMERO, SIN EXCEPCIÓN
            # validate_mx_plate, is_plate_registered, save_image_encrypted
            # y save_to_db corren en _save_plate_async (hilo daemon).
            # El hilo UI nunca espera I/O → la cámara nunca se congela.
            # ══════════════════════════════════════════════════════════════
            try:
                rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                lw = self.video_label.winfo_width()
                lh = self.video_label.winfo_height()
                
                # Validar que tenemos dimensiones válidas
                if lw > 10 and lh > 10:
                    fh, fw = rgb.shape[:2]
                    
                    # Validar dimensiones del frame
                    if fw > 0 and fh > 0:
                        # FIX: Cachear dimensiones para evitar recalcular cada frame
                        cache_key = (fw, fh, lw, lh)
                        if not hasattr(self, '_render_cache') or self._render_cache.get('key') != cache_key:
                            # Calcular aspect ratio correcto (mantener proporciones)
                            scale = min(lw/fw, lh/fh)
                            nw = max(1, int(fw * scale))
                            nh = max(1, int(fh * scale))
                            
                            # Limitar tamaño máximo para evitar problemas de memoria
                            max_dim = 1920
                            if nw > max_dim or nh > max_dim:
                                scale_down = min(max_dim/nw, max_dim/nh)
                                nw = max(1, int(nw * scale_down))
                                nh = max(1, int(nh * scale_down))
                            
                            # Limitar tamaño del canvas
                            canvas_h = min(lh, 1080)
                            canvas_w = min(lw, 1920)
                            
                            # Calcular offsets para centrar
                            y_offset = max(0, (canvas_h - nh) // 2)
                            x_offset = max(0, (canvas_w - nw) // 2)
                            
                            self._render_cache = {
                                'key': cache_key,
                                'nw': nw,
                                'nh': nh,
                                'canvas_h': canvas_h,
                                'canvas_w': canvas_w,
                                'y_offset': y_offset,
                                'x_offset': x_offset,
                            }
                        
                        # Usar valores cacheados
                        cache = self._render_cache
                        nw, nh = cache['nw'], cache['nh']
                        canvas_h, canvas_w = cache['canvas_h'], cache['canvas_w']
                        y_offset, x_offset = cache['y_offset'], cache['x_offset']
                        
                        # Redimensionar frame manteniendo aspect ratio
                        rgb_resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
                        
                        # Crear canvas negro del tamaño del widget (letterbox/pillarbox)
                        # Limitar tamaño del canvas para evitar crashes
                        canvas_h = min(lh, 1080)
                        canvas_w = min(lw, 1920)
                        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                        
                        # Centrar la imagen redimensionada en el canvas
                        y_offset = max(0, (canvas_h - nh) // 2)
                        x_offset = max(0, (canvas_w - nw) // 2)
                        
                        # Asegurar que no excedemos los límites del canvas
                        y_end = min(canvas_h, y_offset + nh)
                        x_end = min(canvas_w, x_offset + nw)
                        nh_actual = y_end - y_offset
                        nw_actual = x_end - x_offset
                        
                        if nh_actual > 0 and nw_actual > 0:
                            canvas[y_offset:y_end, x_offset:x_end] = rgb_resized[:nh_actual, :nw_actual]
                        
                        img = ImageTk.PhotoImage(Image.fromarray(canvas))
                        self.video_label.config(image=img)
                        self.video_label.image = img
            except Exception as e:
                print(f"[Render] Error: {e}")
                import traceback
                traceback.print_exc()

            # PASO 2 — labels ligeros (microsegundos)
            now = time.time()
            fps = 1.0 / max(now - self.prev_time, 1e-6)
            self.prev_time = now
            if self.show_fps:
                self.lbl_fps.config(text=f"FPS: {fps:.1f}")

            proc_ms = latest.get("proc_ms", 0)
            self._fps_history.append(fps)
            self._proc_ms_history.append(proc_ms)
            self._veh_count_history.append(latest.get("n_vehicles", 0))

            if latest["vehicle_detected"] is not None:
                if latest["vehicle_detected"]:
                    self.lbl_detect.config(
                        text=f"VEH: {latest.get('n_vehicles',1)}", fg="#a6e3a1")
                else:
                    self.lbl_detect.config(text="VEH: No", fg="#6c7086")

            # PASO 3 — despachar placas a hilo de fondo (sin bloquear UI)
            orig_snap = latest.get("orig")
            for pdata in latest.get("confirmed_plates", []):
                resp_ms = pdata.get("response_ms", 0)
                self._total_detections += 1
                self._response_times.append(resp_ms)
                # deque con maxlen maneja automáticamente el límite
                threading.Thread(
                    target=self._save_plate_async,
                    args=(pdata["text"], pdata["crop"],
                          pdata["conf"], resp_ms, orig_snap),
                    daemon=True,
                ).start()

        self.root.after(16, self.update_ui)
        # Revisar si el modelo CNN fue actualizado por retrain_loop.py (cada 30s)
        # FIX: usar time.time() en lugar de contador de frames — el contador
        # es inexacto si la UI va más lenta de lo normal (lag, ventana minimizada).
        now_ts = time.time()
        if now_ts - self._last_cnn_check >= 30.0:
            self._last_cnn_check = now_ts
            get_char_recognizer().reload_if_updated()

        # Notificar cuando TrOCR o PaddleOCR terminan de cargar
        if not self._trocr_was_ready and get_trocr_recognizer().available:
            self._trocr_was_ready = True
            self.lbl_status.config(text="TrOCR listo", fg="#a6e3a1")
            self.root.after(3000, lambda: self.lbl_status.config(text="Listo", fg="#a6e3a1"))
        if not getattr(self, "_paddle_was_ready", False) and get_paddle_recognizer().available:
            self._paddle_was_ready = True
            self.lbl_status.config(text="PaddleOCR listo", fg="#a6e3a1")
            self.root.after(3000, lambda: self.lbl_status.config(text="Listo", fg="#a6e3a1"))

    # ------------------------------------------------------------------
    # Worker de guardado — hilo daemon, nunca en el hilo UI
    # ------------------------------------------------------------------
    def _save_plate_async(self, text: str, crop, conf: float,
                          resp_ms: float, orig):
        """Todo el I/O pesado en hilo daemon. El render ya ocurrió."""
        if self.validate_mx_format:
            valid, reason = validate_mx_plate(text)
            if not valid:
                ih = image_hash(crop)
                ip = ""
                if self.auto_save_images and orig is not None:
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ip = save_image_encrypted(
                        orig, os.path.join(IMAGE_FOLDER, f"inv_{ts_str}_{text}"))
                save_invalid_registration(text, reason, ih, ip)
                return

        state      = identify_mx_state(text)
        registered = is_plate_registered(text)

        def _upd(t=text, s=state):
            self.lbl_plate.config(text=f"PLACA: {format_mx_plate(t)}")
            self.lbl_state.config(text=f"ESTADO: {s}")
            self.last_plate_text = t
        self.root.after(0, _upd)

        if not should_register(text):
            return

        with _traffic_lock:
            global traffic_counter
            traffic_counter += 1
            cnt = traffic_counter

        img_path = ""
        if self.auto_save_images and orig is not None:
            ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = save_image_encrypted(
                orig, os.path.join(IMAGE_FOLDER, f"{ts_str}_{text}"))

        ih = (image_hash(orig)
              if self.auto_save_images and orig is not None
              else image_hash(crop))

        save_to_db(text, img_path, ih, cnt, registered, state, conf)

        if img_path and getattr(self, "_video_training_mode", False):
            self._video_saved_count = getattr(self, "_video_saved_count", 0) + 1

        color = "verde" if registered else "nueva"
        print(f"[{color.upper()}] {text} | {state} | "
              f"#{cnt} | {resp_ms:.0f}ms | conf={conf:.2f}")
        self.root.after(0, self.refresh_detections_list)

    # ------------------------------------------------------------------
    # Lista de detecciones
    # ------------------------------------------------------------------
    def refresh_detections_list(self):
        self.listbox.delete(0, tk.END)
        try:
            with sqlite3.connect(DB_NAME) as conn:
                rows = conn.execute(
                    "SELECT plate,timestamp,traffic_count,is_registered,state "
                    "FROM detections ORDER BY timestamp DESC LIMIT 30"
                ).fetchall()
        except Exception as e:
            print(f"Error lista: {e}"); rows = []
        for plate,ts,cnt,reg,state in rows:
            mark  = "OK" if reg else "--"
            ts_s  = str(ts)[11:19]
            st    = (state or "?")[:3].upper()
            self.listbox.insert(tk.END,
                f"{format_mx_plate(plate):<12} {ts_s}  {st:<3}  #{cnt:<4} {mark}")

    def registrar_ultima_placa(self):
        if self.last_plate_text:
            register_new_plate(self.last_plate_text)
            self.refresh_detections_list()
        else:
            messagebox.showinfo("Registrar","Aun no se ha detectado ninguna placa.")

    def buscar_placa(self):
        win = tk.Toplevel(self.root)
        win.title("Buscar placa"); win.geometry("360x200")
        tk.Label(win,text="Matricula:").pack(pady=6)
        ent = tk.Entry(win,font=("Consolas",14)); ent.pack(pady=4)
        res_lbl = tk.Label(win,text="",wraplength=320); res_lbl.pack(pady=4)
        def buscar():
            t = ent.get().strip().upper().replace("-","")
            if not t: return
            t0 = time.time()
            try:
                with sqlite3.connect(DB_NAME) as conn:
                    n = conn.execute(
                        "SELECT COUNT(*) FROM detections WHERE plate=?",(t,)
                    ).fetchone()[0]
                    reg = conn.execute(
                        "SELECT state,registration_date FROM registered_plates "
                        "WHERE plate=? AND status='active'",(t,)
                    ).fetchone()
                ms = (time.time()-t0)*1000
                if reg:
                    msg = (f"REGISTRADA\nEstado: {reg[0]}\n"
                           f"Desde: {str(reg[1])[:10]}\n"
                           f"Detecciones: {n}\nBusqueda: {ms:.1f}ms")
                    res_lbl.config(text=msg, fg="green")
                else:
                    msg = f"No registrada\nDetecciones: {n}\nBusqueda: {ms:.1f}ms"
                    res_lbl.config(text=msg, fg="gray")
            except Exception as e:
                messagebox.showerror("Error",str(e))
        tk.Button(win,text="Buscar",command=buscar).pack(pady=6)
        ent.bind("<Return>", lambda _: buscar())

    # ------------------------------------------------------------------
    # Cierre
    # ------------------------------------------------------------------
    def on_closing(self):
        self.running = False
        if self.proc_thread:
            self.proc_thread.stop()
            try: self.frame_queue.put_nowait(None)
            except queue.Full: pass
        
        # FIX: Liberar VideoCapture de forma segura para evitar crash de FFmpeg
        if self.cap:
            try:
                # Detener la captura antes de liberar
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.1)  # Dar tiempo a FFmpeg para limpiar
                self.cap.release()
                self.cap = None
            except Exception as e:
                print(f"[Cleanup] Error liberando VideoCapture: {e}")
                self.cap = None
        
        # FIX: Control de brillo eliminado
        self.root.destroy()

    def abrir_validacion_manual(self):
        """Abre la herramienta de validación manual de datos de entrenamiento."""
        if self.model_plate is None or self.reader is None:
            messagebox.showwarning(
                "Modelos no listos",
                "Espera a que los modelos terminen de cargar antes de validar.")
            return
        tool = ValidationTool(self.root, self.reader, self.model_plate)
        tool.open()


# ============================================================
# EJECUCION
# ============================================================
if __name__ == "__main__":
    init_db()
    load_registered_plates()
    root = tk.Tk()
    app = LPRApp(root)
    # update_ui se inicia automáticamente después de cargar modelos y abrir cámara
    root.mainloop()
    