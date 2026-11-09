# ==============================================================================
# MAIN.PY - SERVER FASTAPI & INTERFASE STREAMLIT CHATBOT RAG KAMPUS
# ==============================================================================
# File ini menggabungkan backend FastAPI dan frontend Streamlit untuk aplikasi
# Chatbot RAG (Retrieval-Augmented Generation) Universitas Trunojoyo Madura.
# ==============================================================================

# --- KONFIGURASI VARIABEL LINGKUNGAN (ENVIRONMENT VARIABLES) ---
import os
# Matikan TensorFlow agar aplikasi lebih ringan.
os.environ["USE_TF"] = "0"                 # Menonaktifkan penggunaan TensorFlow untuk efisiensi RAM/GPU
# Matikan telemetri anonim dari ChromaDB.
os.environ["ANONYMIZED_TELEMETRY"] = "False" # Menonaktifkan pengiriman data telemetri anonim di ChromaDB

# --- IMPORT LIBRARY / DEPENDENCIES ---
# Mengimpor modul FastAPI untuk layanan REST API backend
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
# Response HTML dipakai untuk halaman, JSON untuk API, dan redirect untuk login admin.
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import bcrypt
from dotenv import load_dotenv, set_key
# Uvicorn menjalankan aplikasi FastAPI sebagai web server.
import uvicorn

# Mengimpor utilitas Python bawaan
import hashlib
import hmac
import json
import re
import secrets
import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Any

# Mengimpor modul LangChain untuk RAG, Embeddings, Vector Store, LLM, dan Memori
from langchain_huggingface import HuggingFaceEmbeddings  # Mengubah teks menjadi vektor (embeddings)
from langchain_chroma import Chroma                        # Database Vektor (Vector Store)
from langchain_groq import ChatGroq                        # Model Bahasa (LLM) dari Groq
from langchain.memory import ConversationBufferMemory      # Menyimpan riwayat percakapan dalam sesi
from langchain.chains import ConversationalRetrievalChain  # Chain RAG untuk pencarian & percakapan
from langchain.prompts import PromptTemplate               # Template instruksi prompt untuk LLM
from langchain_core.retrievers import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# Modul lokal untuk membaca PDF dan menyimpan statistik pertanyaan.
from pdf_parser import extract_pdf
from popular_questions import record_user_question, get_popular_questions
from unanswered_questions import (
    is_unanswered_response,
    record_question,
    record_unanswered_question,
    get_all_unanswered,
    get_all_question_history,
    get_unanswered_stats,
    get_history_stats,
    update_status,
    delete_unanswered,
    CATEGORY_ANSWERED,
    CATEGORY_UNANSWERED,
    STATUS_AUTO_ANSWERED,
    STATUS_DONE,
    STATUS_PROCESSING,
    STATUS_UNPROCESSED,
)

# --- FUNGSI UTILITY (PEMBERSIHAN EMOJI UNTUK EXPORT PDF) ---
def remove_emojis(text):
    """
    Fungsi untuk menghapus karakter emoji dan simbol unicode khusus dari teks.
    Diperlukan karena perpustakaan FPDF standar tidak mendukung karakter Unicode multi-byte (emoji).
    """
    # Pola ini mencakup emoji, simbol, dan karakter Unicode tertentu.
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # Emotikon / ekspresi wajah
        u"\U0001F300-\U0001F5FF"  # Simbol & gambar (pictographs)
        u"\U0001F680-\U0001F6FF"  # Simbol transportasi & peta
        u"\U0001F1E0-\U0001F1FF"  # Simbol bendera negara (iOS)
        u"\U00002500-\U00002BEF"  # Karakter khusus Mandarin/Asia
        u"\U00002702-\U000027B0"  # Simbol dekoratif (Dingbats)
        u"\U000024C2-\U0001F251"  # Simbol lingkaran dan karakter khusus lainnya
        u"\U0001f926-\U0001f937"  # Karakter gesture / tubuh manusia
        u"\U00002600-\U00002B55"  # Simbol cuaca & astronomi
        u"\u200d\u23cf\u23e9\u231a\ufe0f\u3030"  # Variasi modifier unicode
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# --- MEMUAT KONFIGURASI APLIKASI (CONFIG.JSON) ---
# Menentukan jalur direktori tempat file main.py berada
working_dir = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(working_dir, ".env")
load_dotenv(env_path, override=False)
# Semua file relatif aplikasi dicari dari folder tempat main.py berada.
config_path = os.path.join(working_dir, "config.json")

# Membaca file config.json jika tersedia untuk mengambil konfigurasi API Key dan Nama Kampus
if os.path.exists(config_path):
    # Ambil pengaturan yang sudah disimpan admin.
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
else:
    # Gunakan konfigurasi kosong jika file belum tersedia.
    config_data = {}

SECRET_CONFIG_KEYS = {
    "GROQ_API_KEY",
    "ADMIN_USERNAME",
    "ADMIN_PASSWORD",
    "ADMIN_PASSWORD_HASH",
    "ADMIN_EMAIL",
    "ADMIN_SESSION_SECRET",
}


def _write_env_value(key, value):
    """Store a secret in .env without printing it."""
    set_key(env_path, key, str(value), quote_mode="auto")


def _migrate_legacy_secrets():
    """Move legacy config secrets to .env while preserving existing accounts."""
    legacy_values = {
        "GROQ_API_KEY": config_data.get("GROQ_API_KEY", ""),
        "ADMIN_USERNAME": config_data.get("ADMIN_USERNAME", ""),
        "ADMIN_EMAIL": config_data.get("ADMIN_EMAIL", ""),
        "ADMIN_SESSION_SECRET": config_data.get("ADMIN_SESSION_SECRET", ""),
    }
    legacy_password = config_data.get("ADMIN_PASSWORD", "")
    if not os.getenv("ADMIN_PASSWORD_HASH") and legacy_password:
        if legacy_password.startswith(("$2a$", "$2b$", "$2y$")):
            legacy_hash = legacy_password
        else:
            legacy_hash = bcrypt.hashpw(
                legacy_password.encode("utf-8"), bcrypt.gensalt()
            ).decode("utf-8")
        legacy_values["ADMIN_PASSWORD_HASH"] = legacy_hash

    migrated = False
    for key, value in legacy_values.items():
        if value and not os.getenv(key):
            _write_env_value(key, value)
            os.environ[key] = str(value)
            migrated = True

    if migrated:
        for key in SECRET_CONFIG_KEYS:
            config_data.pop(key, None)
        with open(config_path, "w", encoding="utf-8") as config_file:
            json.dump(config_data, config_file, indent=2, ensure_ascii=False)


_migrate_legacy_secrets()

# Rahasia dibaca dari environment, bukan disimpan kembali ke config.json.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Mengambil Nama Kampus dari konfigurasi (default: Universitas Trunojoyo Madura)
COLLEGE_NAME = config_data.get("COLLEGE_NAME", "Universitas Trunojoyo Madura")
if COLLEGE_NAME == "Nama Kampus Anda" or not COLLEGE_NAME:
    COLLEGE_NAME = "Universitas Trunojoyo Madura"

ADMIN_NAME = config_data.get("ADMIN_NAME", "Administrator IF UTM")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin.if@trunojoyo.ac.id")


def save_config(updates: dict):
    """Simpan perubahan konfigurasi ke config.json dan perbarui variabel global runtime."""
    # Variabel global diperbarui agar perubahan langsung berlaku tanpa restart.
    global config_data, ADMIN_USERNAME, ADMIN_PASSWORD_HASH, ADMIN_NAME, ADMIN_EMAIL, COLLEGE_NAME, GROQ_API_KEY
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception:
            pass
    for k, v in updates.items():
        # Simpan setiap nilai baru ke konfigurasi dan runtime.
        if k in SECRET_CONFIG_KEYS:
            _write_env_value(k, v)
            os.environ[k] = str(v)
        else:
            config_data[k] = v
        if k == "ADMIN_USERNAME":
            ADMIN_USERNAME = v
        elif k == "ADMIN_PASSWORD_HASH":
            ADMIN_PASSWORD_HASH = v
        elif k == "ADMIN_NAME":
            ADMIN_NAME = v
        elif k == "ADMIN_EMAIL":
            ADMIN_EMAIL = v
        elif k == "COLLEGE_NAME":
            COLLEGE_NAME = v
        elif k == "GROQ_API_KEY":
            GROQ_API_KEY = v
            os.environ["GROQ_API_KEY"] = v
            try:
                chat_chain.cache_clear()
            except Exception:
                pass
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    except Exception as err:
        print(f"[WARN] Gagal menyimpan konfigurasi ke config.json: {err}")
# Gunakan secret persisten dari config.json agar cookie tetap valid meski server restart.
# Jika belum ada, generate sekali lalu simpan ke config.json.
_stored_secret = os.getenv("ADMIN_SESSION_SECRET", "")
if not _stored_secret:
    # Secret acak dibuat sekali untuk menandatangani cookie admin.
    _stored_secret = secrets.token_urlsafe(32)
    _write_env_value("ADMIN_SESSION_SECRET", _stored_secret)
    os.environ["ADMIN_SESSION_SECRET"] = _stored_secret
ADMIN_SESSION_SECRET = _stored_secret
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "true").lower() in {"1", "true", "yes", "on"}
LOGIN_RATE_WINDOW = 15 * 60
LOGIN_RATE_LIMIT = 5
_login_attempts = defaultdict(deque)
_csrf_cookie_name = "admin_csrf"
MAX_UPLOAD_SIZE = 25 * 1024 * 1024
# PDF yang diunggah dibatasi maksimal 25 MB.
data_dir = Path(working_dir) / "data"


INDONESIAN_STOPWORDS = {
    "ada", "adalah", "adanya", "agar", "akan", "akankah", "antara", "apa", "apabila", 
    "apakah", "ataupun", "atau", "bagai", "bagaimana", "bagaimanakah", "bagi", "bahkan", "bahwa", 
    "banyak", "beberapa", "begitu", "belum", "berada", "berapa", "berapakah", "berikut", "bisa", 
    "boleh", "cara", "cukup", "dalam", "dan", "dapat", "dari", "daripada", "dekat", "demikian", 
    "dengan", "depan", "di", "dia", "dimana", "dimanakah", "diri", "disini", "saat", "saatnya", 
    "hal", "hanya", "harus", "ia", "ialah", "ini", "itu", "itukah", "jadi", "jangan", "jika", 
    "jikalau", "juga", "kalau", "kami", "kamu", "kapan", "kapankah", "karena", "ke", "kecil", 
    "kenapa", "kepada", "kita", "lagi", "lain", "lalu", "lama", "mana", "manakah", "masih", 
    "mau", "maupun", "melalui", "memang", "mengapa", "merupakan", "mereka", "nah", "namun", 
    "nanti", "oleh", "pada", "padahal", "paling", "para", "pasti", "pernah", "pula", "pun", 
    "saja", "saling", "sama", "sampai", "sana", "sangat", "saya", "sebab", "sebagai", "sebuah", 
    "secara", "sedang", "sedangkan", "sedikit", "sejak", "sekali", "sekarang", "sekitar", 
    "selain", "selalu", "selama", "seluruh", "semakin", "sementara", "semua", "semula", "sendiri", 
    "seperti", "sering", "serta", "sesuatu", "sesudah", "setelah", "setiap", "siapa", "siapakah", 
    "sudah", "sungguh", "supaya", "tahu", "tak", "telah", "tentang", "tentu", "terhadap", "tersebut", 
    "tiap", "tidak", "toh", "waduh", "wah", "wahai", "yaitu", "yakni", "yang", "tentang", "untuk"
}


class KeywordAwareRetriever(BaseRetriever):
    """Hybrid Retriever: Menggabungkan semantic search (Chroma) dan keyword search terfilter stopwords."""

    vectorstore: Any
    k: int = 6

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query, *, run_manager=None):
        # Retriever menggabungkan pencarian berdasarkan makna dan kata kunci.
        # 1. Ekstrak token kata penting (abaikan stopwords dan kata < 3 huruf)
        raw_tokens = re.findall(r"[\w@.-]{3,}", query.lower())
        meaningful_tokens = set(t for t in raw_tokens if t not in INDONESIAN_STOPWORDS)

        # 2. Ambil dokumen dari Semantic Search (vektor)
        semantic_documents = self.vectorstore.similarity_search(query, k=self.k)

        # 3. Pencarian Kata Kunci Tertarget dari koleksi Chroma
        keyword_scored_documents = []
        if meaningful_tokens:
            # Ambil seluruh chunk untuk mencari kecocokan istilah secara langsung.
            collection_documents = self.vectorstore._collection.get(
                include=["documents", "metadatas"]
            )
            for text, metadata in zip(
                collection_documents.get("documents", []),
                collection_documents.get("metadatas", []),
            ):
                normalized_text = (text or "").lower()
                
                # Lewati chunk yang hanya berupa daftar isi berulang (titik-titik ...)
                if normalized_text.count("....") > 3:
                    continue

                chunk_words = set(re.findall(r"[\w@.-]{3,}", normalized_text))
                matched_tokens = meaningful_tokens & chunk_words
                match_count = len(matched_tokens)

                if match_count > 0:
                    # Berikan bobot lebih tinggi jika cocok banyak kata kunci unik
                    keyword_scored_documents.append((
                        match_count,
                        Document(page_content=text, metadata=metadata or {})
                    ))

            keyword_scored_documents.sort(key=lambda x: x[0], reverse=True)

        # 4. Hybrid Merging: Berikan prioritas pada dokumen yang muncul di semantic + keyword
        merged_docs = []
        seen = set()

        # Prioritas A: Dokumen keyword dengan skor kecocokan tertinggi (>= 2 keyword relevan)
        for score, doc in keyword_scored_documents:
            if score >= 2:
                key = (doc.metadata.get("source"), doc.page_content[:150])
                if key not in seen:
                    seen.add(key)
                    merged_docs.append(doc)
            if len(merged_docs) >= 3:
                break

        # Prioritas B: Dokumen dari Semantic Search
        for doc in semantic_documents:
            key = (doc.metadata.get("source"), doc.page_content[:150])
            if key not in seen:
                seen.add(key)
                merged_docs.append(doc)

        # Prioritas C: Sisa dokumen keyword jika masih ada slot
        for score, doc in keyword_scored_documents:
            key = (doc.metadata.get("source"), doc.page_content[:150])
            if key not in seen:
                seen.add(key)
                merged_docs.append(doc)

        return merged_docs[: self.k]


# --- INISIALISASI BACKEND FASTAPI ---
# Menginisialisasi aplikasi FastAPI untuk mendukung integrasi REST API eksternal
app = FastAPI(title=f"Chatbot API {COLLEGE_NAME}")
# Hanya aset publik yang boleh disajikan oleh endpoint static.
public_dir = os.path.join(working_dir, "public")
app.mount("/static", StaticFiles(directory=public_dir), name="static")


def make_admin_session(username):
    """Create a short signed token stored in the admin browser cookie."""
    # Token berisi username, waktu pembuatan, dan tanda tangan HMAC.
    payload = f"{username}:{int(time.time())}"
    signature = hmac.new(
        ADMIN_SESSION_SECRET.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    return f"{payload}:{signature}"


def is_admin_authenticated(request):
    # Validasi cookie: format, username, masa berlaku, dan signature.
    token = request.cookies.get("admin_session", "")
    # maxsplit=2: username:timestamp:hexsignature — signature tidak pernah dipecah
    parts = token.split(":", 2)
    if len(parts) != 3:
        return False
    username, issued_at, signature = parts
    if username != ADMIN_USERNAME or not issued_at.isdigit():
        return False
    if time.time() - int(issued_at) > 8 * 60 * 60:
        return False
    payload = f"{username}:{issued_at}"
    expected = hmac.new(
        ADMIN_SESSION_SECRET.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)


def verify_admin_password(password):
    """Verify a candidate password against the bcrypt hash from .env."""
    if not ADMIN_PASSWORD_HASH:
        return False
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"), ADMIN_PASSWORD_HASH.encode("utf-8")
        )
    except ValueError:
        return False


def _client_key(request):
    return request.client.host if request.client else "unknown"


def _login_rate_limited(request):
    now = time.time()
    attempts = _login_attempts[_client_key(request)]
    while attempts and now - attempts[0] >= LOGIN_RATE_WINDOW:
        attempts.popleft()
    if len(attempts) >= LOGIN_RATE_LIMIT:
        return True
    attempts.append(now)
    return False


def _clear_login_attempts(request):
    _login_attempts.pop(_client_key(request), None)


def _valid_csrf(request):
    cookie_token = request.cookies.get(_csrf_cookie_name, "")
    request_token = request.headers.get("X-CSRF-Token", "")
    return bool(cookie_token) and hmac.compare_digest(cookie_token, request_token)


def read_html(filename):
    # Baca file HTML dan ubah menjadi response FastAPI.
    html_path = os.path.join(working_dir, filename)
    try:
        with open(html_path, "r", encoding="utf-8") as html_file:
            return HTMLResponse(content=html_file.read())
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"Frontend tidak dapat dimuat: {error}")


@app.get("/", response_class=HTMLResponse)
async def frontend():
    # Halaman utama untuk pengguna chatbot.
    """Menyajikan frontend chatbot yang dipakai pengguna."""
    return read_html("codeUser.html")


@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    # Halaman login juga menyiapkan token CSRF untuk request POST.
    csrf_token = request.cookies.get(_csrf_cookie_name) or secrets.token_urlsafe(32)
    response = read_html("loginAdmin.html")
    response.set_cookie(
        _csrf_cookie_name,
        csrf_token,
        httponly=False,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=8 * 60 * 60,
        path="/",
    )
    return response


@app.post("/admin/login")
async def admin_login(request: Request):
    # Ambil kredensial dari form login.
    if _login_rate_limited(request):
        return JSONResponse(
            {"detail": "Terlalu banyak percobaan login. Coba lagi dalam 15 menit."},
            status_code=429,
        )
    if not _valid_csrf(request):
        return JSONResponse({"detail": "Token CSRF tidak valid."}, status_code=403)
    form = await request.form()
    username = str(form.get("username", "")).strip()
    password = str(form.get("password", ""))
    
    is_user_match = hmac.compare_digest(username.lower(), ADMIN_USERNAME.lower()) or (
        bool(ADMIN_EMAIL) and hmac.compare_digest(username.lower(), ADMIN_EMAIL.lower())
    )
    is_pw_match = verify_admin_password(password)

    if not is_user_match or not is_pw_match:
        # Kredensial salah dikembalikan sebagai HTTP 401.
        return JSONResponse({"detail": "Username atau password salah."}, status_code=401)
    response = JSONResponse({"message": "Login berhasil.", "redirect": "/admin"})
    response.set_cookie(
        "admin_session",
        make_admin_session(ADMIN_USERNAME),
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=8 * 60 * 60,
        path="/",
    )
    _clear_login_attempts(request)
    return response


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    # Dashboard hanya boleh dibuka oleh admin yang sudah login.
    if not is_admin_authenticated(request):
        return RedirectResponse("/admin/login", status_code=303)
    return read_html("codeAdmin.html")


@app.post("/admin/logout")
async def admin_logout():
    # Hapus cookie session agar admin keluar dari sistem.
    response = JSONResponse({"redirect": "/admin/login"})
    response.delete_cookie("admin_session", path="/")
    return response


@app.get("/admin/profile")
async def admin_get_profile(request: Request):
    if not is_admin_authenticated(request):
        raise HTTPException(status_code=401, detail="Sesi admin tidak valid.")

    masked_key = ""
    if GROQ_API_KEY:
        if len(GROQ_API_KEY) > 10:
            masked_key = f"{GROQ_API_KEY[:6]}...{GROQ_API_KEY[-4:]}"
        else:
            masked_key = "********"

    return {
        "name": ADMIN_NAME,
        "username": ADMIN_USERNAME,
        "email": ADMIN_EMAIL,
        "college_name": COLLEGE_NAME,
        "groq_api_key_masked": masked_key,
        "has_groq_api_key": bool(GROQ_API_KEY)
    }


@app.post("/admin/profile")
async def admin_update_profile(request: Request):
    # Perbarui data profil setelah session admin diverifikasi.
    if not is_admin_authenticated(request):
        raise HTTPException(status_code=401, detail="Sesi admin tidak valid.")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Data permintaan JSON tidak valid.")

    name = str(data.get("name", "")).strip()
    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip()

    if not name or len(name) < 2:
        raise HTTPException(status_code=400, detail="Nama profil minimal 2 karakter.")
    if not username or len(username) < 3:
        raise HTTPException(status_code=400, detail="Username minimal 3 karakter.")
    if any(c in username for c in " :/\t\n"):
        raise HTTPException(status_code=400, detail="Username tidak boleh memuat spasi atau karakter khusus (: /).")

    updates = {
        "ADMIN_NAME": name,
        "ADMIN_USERNAME": username,
        "ADMIN_EMAIL": email
    }
    save_config(updates)

    response = JSONResponse({
        "message": "Profil berhasil diperbarui.",
        "name": ADMIN_NAME,
        "username": ADMIN_USERNAME,
        "email": ADMIN_EMAIL
    })
    # Perbarui sesi cookie dengan username baru agar admin tetap login lancar
    response.set_cookie(
        "admin_session",
        make_admin_session(ADMIN_USERNAME),
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=8 * 60 * 60,
        path="/",
    )
    return response


@app.post("/admin/change-password")
async def admin_change_password(request: Request):
    # Password lama dicek sebelum password baru disimpan.
    if not is_admin_authenticated(request):
        raise HTTPException(status_code=401, detail="Sesi admin tidak valid.")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Data permintaan JSON tidak valid.")

    current_password = str(data.get("current_password", ""))
    new_password = str(data.get("new_password", ""))
    confirm_password = str(data.get("confirm_password", ""))

    if not verify_admin_password(current_password):
        raise HTTPException(status_code=400, detail="Password saat ini (lama) tidak sesuai.")
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="Password baru minimal terdiri dari 6 karakter.")
    if new_password != confirm_password:
        raise HTTPException(status_code=400, detail="Konfirmasi password baru tidak cocok.")

    new_password_hash = bcrypt.hashpw(
        new_password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")
    save_config({"ADMIN_PASSWORD_HASH": new_password_hash})

    response = JSONResponse({"message": "Password berhasil diperbarui."})
    response.set_cookie(
        "admin_session",
        make_admin_session(ADMIN_USERNAME),
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=8 * 60 * 60,
        path="/",
    )
    return response


@app.post("/admin/settings/system")
async def admin_update_system_settings(request: Request):
    # Admin dapat mengubah nama kampus dan API key Groq.
    if not is_admin_authenticated(request):
        raise HTTPException(status_code=401, detail="Sesi admin tidak valid.")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Data permintaan JSON tidak valid.")

    college_name = str(data.get("college_name", "")).strip()
    groq_api_key = str(data.get("groq_api_key", "")).strip()

    updates = {}
    if college_name:
        updates["COLLEGE_NAME"] = college_name
    if groq_api_key and not groq_api_key.startswith("...") and "..." not in groq_api_key:
        if len(groq_api_key) < 15:
            raise HTTPException(status_code=400, detail="Format GROQ API Key tidak valid (minimal 15 karakter).")
        updates["GROQ_API_KEY"] = groq_api_key

    if updates:
        save_config(updates)

    masked_key = ""
    if GROQ_API_KEY:
        if len(GROQ_API_KEY) > 10:
            masked_key = f"{GROQ_API_KEY[:6]}...{GROQ_API_KEY[-4:]}"
        else:
            masked_key = "********"

    return {
        "message": "Pengaturan sistem berhasil diperbarui.",
        "college_name": COLLEGE_NAME,
        "groq_api_key_masked": masked_key,
        "has_groq_api_key": bool(GROQ_API_KEY)
    }


def index_pdf(file_path, category, note):
    # Ekstrak isi PDF, pecah menjadi chunk, lalu masukkan ke ChromaDB.
    extraction = extract_pdf(file_path)
    document = Document(
        page_content=extraction["text"],
        metadata={
            "source": file_path.name,
            "category": category,
            "note": note,
            "pages": extraction["page_count"],
            "extraction_method": extraction["method"],
            "character_count": extraction["character_count"],
        },
    )
    splitter = RecursiveCharacterTextSplitter(
        # Overlap menjaga konteks di batas antar-chunk tetap tersambung.
        chunk_size=2000, chunk_overlap=500, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents([document])
    # Hapus cache vectorstore agar instance baru dibuat setelah perubahan data
    setup_vectorstore.cache_clear()
    chat_chain.cache_clear()
    vectorstore = setup_vectorstore()
    vectorstore._collection.delete(where={"source": file_path.name})
    vectorstore.add_documents(chunks)
    return len(chunks), extraction


@app.post("/admin/upload")
async def admin_upload(
    request: Request,
    file: UploadFile = File(...),
    category: str = Form("aturan"),
    note: str = Form(""),
):
    # Endpoint upload knowledge base utama.
    if not is_admin_authenticated(request):
        raise HTTPException(status_code=401, detail="Sesi admin tidak valid.")
    filename = Path(file.filename or "").name
    # Path(...).name mencegah nama file membawa path dari komputer user.
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File harus berformat PDF.")
    content = await file.read(MAX_UPLOAD_SIZE + 1)
    # Baca satu byte ekstra agar file yang melebihi batas dapat terdeteksi.
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="Ukuran PDF maksimal 25 MB.")
    data_dir.mkdir(exist_ok=True)
    target = data_dir / filename
    target.write_bytes(content)
    try:
        chunks, extraction = index_pdf(target, category, note.strip())
    except Exception as error:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"PDF gagal diproses: {error}")
    return {
        "message": f"{filename} berhasil ditambahkan ke knowledge base.",
        "chunks": chunks,
        "pages": extraction["page_count"],
        "method": extraction["method"],
        "characters": extraction["character_count"],
    }


@app.get("/admin/documents")
async def admin_documents(request: Request):
    if not is_admin_authenticated(request):
        raise HTTPException(status_code=401, detail="Sesi admin tidak valid.")
    documents = []
    for file_path in sorted(data_dir.glob("*.pdf")):
        documents.append({
            "name": file_path.name,
            "size": file_path.stat().st_size,
            "updated": file_path.stat().st_mtime,
        })
    return {"documents": documents}


@app.delete("/admin/documents/{filename}")
async def admin_delete_document(filename: str, request: Request):
    # Hapus PDF dari disk dan chunk terkait dari vector store.
    if not is_admin_authenticated(request):
        raise HTTPException(status_code=401, detail="Sesi admin tidak valid.")
    # Keamanan: hanya izinkan nama file tanpa path traversal
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file PDF yang dapat dihapus.")
    file_path = data_dir / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File tidak ditemukan.")
    # Hapus dari vector store
    try:
        setup_vectorstore.cache_clear()
        chat_chain.cache_clear()
        vectorstore = setup_vectorstore()
        vectorstore._collection.delete(where={"source": safe_name})
    except Exception as error:
        print(f"Peringatan: gagal menghapus dari vector store: {error}")
    # Hapus file dari disk
    file_path.unlink(missing_ok=True)
    # Reset ulang cache agar vectorstore terbaru dipakai
    setup_vectorstore.cache_clear()
    chat_chain.cache_clear()
    return {"message": f"{safe_name} berhasil dihapus dari knowledge base."}

# Model skema data permintaan (request body) menggunakan Pydantic
class MessageRequest(BaseModel):
    # Bentuk JSON yang diterima endpoint /chat.
    message: str  # Teks pesan pertanyaan dari pengguna


# Rute/Endpoint GET HTTP untuk mengambil daftar pertanyaan populer mahasiswa
@app.get("/api/popular-questions")
async def api_popular_questions():
    """
    Endpoint untuk mengambil 6 pertanyaan terpopuler berdasarkan data riil pertanyaan mahasiswa.
    Jika data belum mencukupi, otomatis dilengkapi dengan pertanyaan default (fallback).
    """
    try:
        questions = get_popular_questions(limit=6)
        return {"questions": questions}
    except Exception as error:
        print(f"Gagal mengambil pertanyaan populer: {error}")
        return {"questions": get_popular_questions(limit=6)}


# Rute/Endpoint POST HTTP untuk layanan percakapan chatbot via REST API
@app.post("/chat")
async def chatbot(request: MessageRequest):
    """
    Endpoint FastAPI untuk memproses pertanyaan pengguna dan mengembalikan jawaban dari RAG Chain.
    Pertanyaan user dicatat untuk Pertanyaan Populer.
    Jika jawaban tidak relevan, pertanyaan otomatis disimpan ke Pertanyaan Belum Terjawab.
    """
    try:
        message = request.message
        # Pertanyaan dicatat lebih dulu untuk fitur pertanyaan populer.

        # Catat dan kelompokkan pertanyaan mahasiswa untuk pembaruan daftar Pertanyaan Populer
        try:
            vectorstore = setup_vectorstore()
            emb_model = getattr(vectorstore, "_embedding_function", None)
            record_user_question(message, embedding_model=emb_model)
        except Exception as record_err:
            print(f"Peringatan pencatatan pertanyaan: {record_err}")

        vectorstore = setup_vectorstore()
        # Ambil chain RAG yang menghubungkan retriever, prompt, memory, dan LLM.
        conversational_chain = chat_chain(vectorstore)

        # Memeriksa apakah pertanyaan mengandung kata/topik sensitif yang dibatasi
        if contains_sensitive_topics(message):
            # Pertanyaan di luar cakupan dapat ditolak sebelum memanggil LLM.
            response = f"Pertanyaan Anda berada di luar konteks informasi resmi {COLLEGE_NAME}."
        else:
            # Bersihkan riwayat lama agar pencarian pertanyaan berdiri sendiri dan tidak terkontaminasi topik sebelumnya
            if hasattr(conversational_chain, "memory") and conversational_chain.memory:
                conversational_chain.memory.clear()
            response = conversational_chain.invoke({"question": message})["answer"]

        # --- CATAT RIWAYAT PERTANYAAN (KATEGORI TERJAWAB VS BELUM TERJAWAB) ---
        try:
            # Jawaban fallback ditandai agar masuk daftar belum terjawab.
            is_unans = is_unanswered_response(response)
            emb_model = getattr(setup_vectorstore(), "_embedding_function", None)
            record_question(
                question=message,
                is_unanswered=is_unans,
                bot_response=response,
                embedding_model=emb_model,
            )
        except Exception as q_err:
            print(f"Peringatan pencatatan riwayat pertanyaan: {q_err}")

        return {"response": response}
    except Exception as error:
        print(f"Chat gagal diproses: {type(error).__name__}: {error}")
        return JSONResponse(
            {"detail": "Chatbot gagal memproses pertanyaan. Periksa terminal server untuk detail error."},
            status_code=500,
        )


# ============================================================
# ENDPOINT PERTANYAAN BELUM TERJAWAB (UNANSWERED QUESTIONS)
# ============================================================

@app.get("/admin/unanswered")
async def admin_unanswered_list(request: Request, status: str = None):
    # Tampilkan pertanyaan yang belum terjawab, dengan filter status opsional.
    """Ambil semua pertanyaan yang belum terjawab, opsional filter berdasarkan status."""
    if not is_admin_authenticated(request):
        raise HTTPException(status_code=401, detail="Sesi admin tidak valid.")
    try:
        questions = get_all_unanswered(status_filter=status if status else None)
        stats = get_unanswered_stats()
        return {"questions": questions, "stats": stats}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Gagal memuat data: {error}")


@app.patch("/admin/unanswered/{question_id}/status")
async def admin_update_unanswered_status(question_id: str, request: Request):
    # Admin mengubah progres penanganan pertanyaan.
    """Perbarui status pertanyaan belum terjawab (Belum Diproses / Diproses / Selesai)."""
    if not is_admin_authenticated(request):
        raise HTTPException(status_code=401, detail="Sesi admin tidak valid.")
    try:
        body = await request.json()
        new_status = body.get("status", "")
        if new_status not in [STATUS_UNPROCESSED, STATUS_PROCESSING, STATUS_DONE]:
            raise HTTPException(status_code=400, detail="Status tidak valid.")
        ok = update_status(question_id, new_status)
        if not ok:
            raise HTTPException(status_code=404, detail="Pertanyaan tidak ditemukan.")
        return {"message": f"Status diperbarui menjadi '{new_status}'."}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Gagal memperbarui status: {error}")


@app.delete("/admin/unanswered/{question_id}")
async def admin_delete_unanswered(question_id: str, request: Request):
    # Hapus satu pertanyaan dari daftar tindak lanjut.
    """Hapus pertanyaan belum terjawab berdasarkan ID."""
    if not is_admin_authenticated(request):
        raise HTTPException(status_code=401, detail="Sesi admin tidak valid.")
    ok = delete_unanswered(question_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Pertanyaan tidak ditemukan.")
    return {"message": "Pertanyaan berhasil dihapus."}


@app.post("/admin/unanswered/{question_id}/upload")
async def admin_unanswered_upload(
    question_id: str,
    request: Request,
    file: UploadFile = File(...),
    category: str = Form("aturan"),
    note: str = Form(""),
):
    """
    Unggah PDF untuk menjawab pertanyaan yang belum terjawab.
    Setelah indexing berhasil, status pertanyaan otomatis diperbarui menjadi 'Selesai'.
    """
    # Upload ini menghubungkan dokumen baru dengan pertanyaan tertentu.
    if not is_admin_authenticated(request):
        raise HTTPException(status_code=401, detail="Sesi admin tidak valid.")

    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File harus berformat PDF.")

    content = await file.read(MAX_UPLOAD_SIZE + 1)
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="Ukuran PDF maksimal 25 MB.")

    data_dir.mkdir(exist_ok=True)
    target = data_dir / filename
    target.write_bytes(content)

    try:
        chunks, extraction = index_pdf(target, category, note.strip())
    except Exception as error:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"PDF gagal diproses: {error}")

    # Tandai pertanyaan sebagai Selesai setelah indexing berhasil
    update_status(question_id, STATUS_DONE, resolved_pdf=filename)

    return {
        "message": f"{filename} berhasil diindeks. Status pertanyaan diperbarui menjadi 'Selesai'.",
        "chunks": chunks,
        "pages": extraction["page_count"],
        "method": extraction["method"],
        "characters": extraction["character_count"],
    }


# ============================================================
# ENDPOINT RIWAYAT SEMUA PERTANYAAN (QUESTION HISTORY)
# ============================================================

@app.get("/admin/questions/history")
async def admin_question_history_list(
    request: Request,
    category: str = None,
    status: str = None,
    search: str = None
):
    # Ambil riwayat pertanyaan dengan filter kategori, status, dan teks pencarian.
    """Ambil riwayat seluruh pertanyaan mahasiswa (filter kategori, status, pencarian)."""
    if not is_admin_authenticated(request):
        raise HTTPException(status_code=401, detail="Sesi admin tidak valid.")
    try:
        history = get_all_question_history(
            category_filter=category if category else None,
            status_filter=status if status else None,
            search_query=search if search else None,
        )
        stats = get_history_stats()
        return {"questions": history, "stats": stats}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Gagal memuat riwayat: {error}")


@app.get("/admin/questions/stats")
async def admin_question_stats(request: Request):
    # Kirim statistik ringkas untuk kartu informasi di dashboard.
    """Ambil statistik ringkas seluruh riwayat pertanyaan."""
    if not is_admin_authenticated(request):
        raise HTTPException(status_code=401, detail="Sesi admin tidak valid.")
    try:
        return get_history_stats()
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Gagal memuat statistik: {error}")


# --- TEMPLATE PROMPT UNTUK MODEL AI (LLM) ---
# Prompt Sistem Utama: Menentukan identitas, peran, gaya bahasa, dan aturan jawaban AI
DEFAULT_SYSTEM_PROMPT = f"""Anda adalah **Asisten AI Resmi** khusus untuk **{COLLEGE_NAME}**.
Tugas utama Anda adalah memberikan jawaban yang **akurat, tepat, dan ramah** berdasarkan dokumen resmi {COLLEGE_NAME} yang tercantum dalam Konteks.

Pedoman Menjawab:
1. Jawab pertanyaan pengguna **HANYA berdasarkan fakta yang tertulis dalam Konteks Dokumen Resmi di bawah**.
2. DILARANG KERAS mengarang, berspekulasi, atau menggunakan asumsi umum di luar isi dokumen resmi (seperti jumlah SKS kelulusan, syarat skripsi, aturan perwalian, dll).
3. Jika informasi yang ditanyakan tercantum dalam dokumen konteks, sebutkan secara jelas, spesifik, dan tepat sesuai bunyi dokumen (contoh: jika dokumen menyatakan lulus minimal 145 SKS, maka sampaikan 145 SKS).
4. Gunakan poin-poin (bullet points) jika menjelaskan persyaratan, daftar mata kuliah, atau langkah prosedur agar mudah dibaca mahasiswa.
5. Jika informasi yang ditanyakan sama sekali TIDAK DITEMUKAN dalam Konteks Dokumen Resmi yang disediakan, sampaikan secara jujur dan sopan:
   "Maaf, informasi spesifik mengenai hal tersebut belum tercantum dalam dokumen panduan resmi {COLLEGE_NAME} yang tersedia saat ini."
"""

# Prompt Negatif: Menjelaskan larangan/pantangan yang harus dihindari oleh AI
DEFAULT_NEGATIVE_PROMPT = f"""
- Jangan mengarang data atau angka apapun yang tidak tertulis pada Konteks Dokumen.
- Jangan bersikap ragu-ragu apabila data secara jelas tertulis dalam Konteks Dokumen.
- Jangan menanyakan pertanyaan kuesioner balik kepada pengguna; langsung berikan jawaban informatif.
- Tetap jaga kesopanan dan keramahan profesional setiap saat.
"""


# --- FUNGSI FILTER TOPIK SENSITIF ---
def contains_sensitive_topics(question):
    """
    Fungsi penapis untuk mendeteksi apakah pertanyaan memuat kata kunci sensitif.
    """
    # Daftar ini dapat diisi jika kampus ingin membatasi topik tertentu.
    sensitive_keywords = []
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in sensitive_keywords)


# --- FUNGSI INSIALISASI VECTOR STORE (CHROMADB) ---
@lru_cache(maxsize=1)
def setup_vectorstore():
    """
    Fungsi untuk memuat database vektor ChromaDB dari direktori 'vector_db_dir'.
    """
    # Cache mencegah model embedding dan koneksi Chroma dibuat berulang kali.
    persist_directory = os.path.join(working_dir, "vector_db_dir")
    embeddings = HuggingFaceEmbeddings()  # Model embedding HuggingFace untuk pencarian kesamaan makna
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embeddings)
    return vectorstore


# --- FUNGSI SETUP RAG CHAIN (LANGCHAIN + GROQ) ---
@lru_cache(maxsize=1)
def chat_chain(vectorstore, system_prompt=DEFAULT_SYSTEM_PROMPT, negative_prompt=DEFAULT_NEGATIVE_PROMPT):
    """
    Fungsi untuk menyusun dan mengembalikan ConversationalRetrievalChain RAG.
    Menghubungkan LLM Groq dengan dukungan fallback model otomatis, Retriever Chroma, Prompt Template, dan Memori Percakapan.
    """
    # Daftar model Groq dicoba berurutan jika model sebelumnya gagal.
    model_candidates = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "qwen/qwen3.8-27b",
        "qwen/qwen3.6-27b",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "openai/gpt-oss-20b"
    ]

    llm = None
    for model_name in model_candidates:
        try:
            # Uji model dengan request sederhana sebelum digunakan.
            candidate_llm = ChatGroq(
                model=model_name,
                temperature=0.1,
                max_tokens=512
            )
            # Uji validitas ketersediaan model dengan prompt singkat
            candidate_llm.invoke("test")
            llm = candidate_llm
            break
        except Exception:
            # Lanjut ke kandidat berikutnya jika model tidak tersedia.
            continue

    if llm is None:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512)
    
    # Batas karakter context agar total token tidak melebihi limit 7000 input token Groq.
    # Estimasi: 1 token ≈ 4 karakter; system prompt ~600 token + context ~2000 token + history + query ≈ aman di bawah 7000.
    MAX_CONTEXT_CHARS = 8000  # ~2000 token untuk context dokumen

    # Penggabungan struktur prompt template dengan variabel dinamis (context, chat_history, question)
    prompt_template = f"""{system_prompt}

{negative_prompt}

Konteks Dokumen Resmi {COLLEGE_NAME}:
{{context}}

Riwayat Percakapan:
{{chat_history}}

Pertanyaan: {{question}}

Jawaban:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

    def truncate_docs(docs):
        """Pangkas daftar dokumen agar total karakter context tidak melebihi MAX_CONTEXT_CHARS."""
        # Context dipotong agar request tidak melewati batas token model.
        truncated = []
        total_chars = 0
        for doc in docs:
            content = doc.page_content
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining <= 0:
                break
            if len(content) > remaining:
                # Potong chunk terakhir agar tidak melebihi batas
                from langchain.schema import Document as LCDocument
                truncated.append(LCDocument(page_content=content[:remaining], metadata=doc.metadata))
                break
            truncated.append(doc)
            total_chars += len(content)
        return truncated

    # Bungkus retriever dengan pemangkasan context agar request aman di bawah 7000 token
    from langchain_core.retrievers import BaseRetriever as _BR
    from langchain.schema import Document as _Doc
    from typing import List as _List

    class TruncatingRetriever(_BR):
        """Retriever wrapper yang memotong hasil retrieval agar context tidak terlalu panjang."""
        inner: object

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(self, query, *, run_manager=None):
            # Ambil hasil retriever lalu batasi panjang context-nya.
            raw_docs = self.inner._get_relevant_documents(query, run_manager=run_manager)
            return truncate_docs(raw_docs)
    
    # Menyiapkan retriever untuk mengambil 6 potongan dokumen paling relevan dari database vektor
    # Dibungkus TruncatingRetriever untuk memastikan context tidak melampaui MAX_CONTEXT_CHARS
    base_retriever = KeywordAwareRetriever(vectorstore=vectorstore, k=6)
    retriever = TruncatingRetriever(inner=base_retriever)
    
    # Memory menyimpan pasangan pertanyaan dan jawaban selama chain aktif.
    memory = ConversationBufferMemory(
        llm = llm,
        output_key = "answer",
        memory_key = "chat_history",
        return_messages = True
    )
    
    # Gabungkan LLM, retriever, prompt, dan memory menjadi chain RAG.
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        chain_type = "stuff",
        memory = memory,
        verbose = False,
        return_source_documents = True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain


if __name__ == "__main__":
    # Jalankan server hanya ketika file ini dieksekusi langsung.
    port = int(os.getenv("PORT", "8501"))
    uvicorn.run(app, host="127.0.0.1", port=port)
