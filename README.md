# Chatbot Akademik Teknik Informatika UTM

Chatbot akademik berbasis Retrieval-Augmented Generation (RAG) untuk membantu mahasiswa Teknik Informatika Universitas Trunojoyo Madura memperoleh informasi dari dokumen akademik resmi.

Project ini menjalankan backend FastAPI dengan antarmuka HTML/JavaScript untuk pengguna dan dashboard administrator. Jawaban chatbot dibuat menggunakan dokumen yang diindeks ke ChromaDB dan model bahasa Groq.

## Tujuan Sistem

- Menjawab pertanyaan akademik berdasarkan knowledge base resmi.
- Membantu mahasiswa mencari informasi kurikulum, skripsi, kerja praktik, perwalian, dan layanan akademik.
- Membantu admin mengelola PDF knowledge base.
- Mencatat pertanyaan populer, riwayat pertanyaan, dan pertanyaan yang belum terjawab.

## Fitur

### Chatbot User

- Chat RAG berbasis dokumen akademik.
- Pencarian semantic dan keyword melalui ChromaDB.
- Pertanyaan populer.
- Pencatatan riwayat pertanyaan.
- Jawaban fallback jika informasi tidak ditemukan.

### Dashboard Admin

- Login admin dengan password bcrypt.
- Session cookie bertanda tangan HMAC.
- CSRF protection dan rate limiting login.
- Upload, indexing, dan penghapusan PDF.
- Daftar pertanyaan belum terjawab.
- Riwayat dan statistik pertanyaan.
- Pengaturan profil admin dan konfigurasi sistem.

## Teknologi

- Python 3.10 atau lebih baru direkomendasikan.
- FastAPI dan Uvicorn.
- LangChain.
- ChromaDB untuk vector database.
- HuggingFace Embeddings.
- Groq untuk LLM.
- pypdf dan PyMuPDF untuk PDF.
- Tesseract OCR untuk PDF hasil scan.
- HTML, JavaScript, dan Tailwind CSS CDN untuk frontend.

## Struktur Project

```text
.
├── main.py                 # Aplikasi FastAPI dan endpoint chatbot/admin
├── pdf_parser.py           # Ekstraksi teks PDF dan OCR
├── vectorize_documents.py  # Pembuatan atau pembaruan vector database
├── popular_questions.py    # Penyimpanan pertanyaan populer
├── unanswered_questions.py  # Riwayat dan pertanyaan belum terjawab
├── cli_chat.py             # Client chat melalui terminal
├── codeUser.html           # Antarmuka chatbot user
├── loginAdmin.html         # Halaman login admin
├── codeAdmin.html          # Dashboard admin
├── public/                 # Hanya aset yang boleh diakses browser
├── .env.example            # Template secret environment
├── config.example.json     # Template konfigurasi non-rahasia
├── requirements.txt        # Dependency Python
├── data/                   # Lokal: PDF dan JSON runtime, tidak di Git
├── vector_db_dir/          # Lokal: ChromaDB, tidak di Git
└── tessdata/               # Lokal: bahasa Tesseract, tidak di Git
```

`code.html` dipertahankan sebagai file frontend lama, tetapi route `/` saat ini menyajikan `codeUser.html`.

## Instalasi Lokal

### Prasyarat

- Python dan pip.
- Tesseract OCR jika akan memproses PDF scan.
- API key Groq.
- Akses internet saat model embedding HuggingFace pertama kali diunduh.

Pada Linux/Ubuntu:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip tesseract-ocr tesseract-ocr-ind
```

Pada Windows, install Tesseract dan pastikan executable tersedia di `PATH`, atau set `TESSERACT_CMD`.

### Virtual Environment dan Dependency

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Konfigurasi

### File `.env`

Salin `.env.example` menjadi `.env`, lalu isi dengan nilai milik deployment. Jangan commit atau upload `.env`.

```env
GROQ_API_KEY=<groq-api-key>
ADMIN_USERNAME=<admin-username>
ADMIN_EMAIL=<admin-email>
ADMIN_PASSWORD_HASH=<bcrypt-password-hash>
ADMIN_SESSION_SECRET=<random-session-secret>
COOKIE_SECURE=true
```

Password admin harus berupa bcrypt hash. Jangan mengisi `ADMIN_PASSWORD_HASH` dengan password plaintext.

Untuk membuat hash bcrypt melalui environment project:

```bash
python -c "import bcrypt; print(bcrypt.hashpw(input('Password: ').encode(), bcrypt.gensalt()).decode())"
```

Pada testing lokal HTTP, `COOKIE_SECURE=false` dapat digunakan. Pada HTTPS production, gunakan `true`.

### File `config.json`

`config.json` bersifat opsional untuk konfigurasi non-rahasia yang dapat diubah dari dashboard, misalnya:

```json
{
  "COLLEGE_NAME": "Universitas Trunojoyo Madura",
  "ADMIN_NAME": "Administrator IF UTM"
}
```

Gunakan `config.example.json` sebagai template. API key, password, username, email, dan session secret tidak boleh disimpan di file ini.

Jika `config.json` belum ada, aplikasi menggunakan nilai default non-rahasia dan dapat membuat file tersebut saat admin menyimpan pengaturan. Secret tetap disimpan di `.env`.

## Menyiapkan Knowledge Base

Folder `data/` dan `vector_db_dir/` sengaja tidak ada di repository GitHub karena dapat berisi dokumen akademik dan data lokal.

1. Buat folder `data/`.
2. Salin PDF akademik ke `data/`.
3. Jika membuat vector database baru, jalankan:

```bash
python vectorize_documents.py
```

4. Pastikan `vector_db_dir/` terbentuk.

Saat deployment, vector database yang sudah ada dapat digunakan kembali dengan menyalin seluruh folder `vector_db_dir/` ke folder project server. ChromaDB akan membuka persistence directory tersebut; tidak perlu membuat embedding ulang selama struktur dan model embedding tetap kompatibel.

PDF tetap perlu disalin ke `data/` karena dashboard membaca daftar dokumen dari folder tersebut. Riwayat pertanyaan disimpan di:

- `data/popular_questions.json`
- `data/unanswered_questions.json`

Jangan menghapus file-file tersebut saat memperbarui aplikasi.

## Menjalankan Aplikasi

Aplikasi aktif menggunakan FastAPI, bukan Streamlit.

```bash
uvicorn main:app --host 127.0.0.1 --port 8501
```

Buka:

- User: `http://127.0.0.1:8501/`
- Login admin: `http://127.0.0.1:8501/admin/login`

Untuk akses dari jaringan lokal saat testing saja:

```bash
uvicorn main:app --host 0.0.0.0 --port 8501
```

Pada production, bind ke `127.0.0.1` dan gunakan Nginx sebagai reverse proxy HTTPS.

## Pengujian

Pemeriksaan sintaks dan import:

```bash
python -m py_compile main.py pdf_parser.py popular_questions.py unanswered_questions.py vectorize_documents.py cli_chat.py
python -c "import main, pdf_parser, popular_questions, unanswered_questions, vectorize_documents, cli_chat; print('imports ok')"
python -m pip check
```

Pemeriksaan endpoint dasar:

```bash
curl http://127.0.0.1:8501/
curl http://127.0.0.1:8501/admin/login
```

Checklist manual:

- Halaman user terbuka.
- Login admin berhasil dengan credential dari `.env`.
- Dashboard admin mengarahkan user tanpa session kembali ke login.
- Logout menghapus session.
- Upload PDF berhasil dan jumlah chunk dikembalikan.
- PDF muncul pada daftar dokumen.
- Pertanyaan tampil pada riwayat atau daftar belum terjawab.
- File `config.json` tidak dapat diakses melalui `/static/config.json`.
- `data/` dan `vector_db_dir/` tetap ada setelah restart aplikasi.

## Backup dan Restore Data

Backup minimal harus mencakup:

- `data/`
- `vector_db_dir/`
- `config.json` jika berisi konfigurasi non-rahasia yang ingin dipertahankan
- `.env` melalui media rahasia terpisah, bukan Git
- `tessdata/` jika memakai OCR dengan data bahasa lokal

Linux:

```bash
tar -czf chatbot-data-backup.tar.gz data vector_db_dir tessdata config.json
```

Backup secret secara terpisah dan aman:

```bash
cp .env /secure/backup/location/chatbot.env
chmod 600 /secure/backup/location/chatbot.env
```

Restore dilakukan dengan mengekstrak kembali folder ke root project yang sama, lalu memastikan owner dan permission service benar.

Windows PowerShell:

```powershell
Compress-Archive -Path data,vector_db_dir,tessdata,config.json -DestinationPath chatbot-data-backup.zip
```

Jangan menyimpan `.env` di archive yang akan di-upload ke GitHub.

## Persiapan Deployment Linux

1. Clone repository.
2. Buat virtual environment dan install `requirements.txt`.
3. Install Tesseract jika OCR diperlukan.
4. Buat `.env` di server dengan permission terbatas.
5. Salin `data/`, `vector_db_dir/`, dan `tessdata/` dari backup.
6. Salin `config.json` hanya jika diperlukan.
7. Jalankan uji lokal dengan Uvicorn.
8. Buat service systemd agar aplikasi tetap aktif setelah SSH ditutup.
9. Gunakan Nginx untuk domain dan HTTPS.
10. Pastikan port aplikasi hanya terbuka untuk localhost dan firewall hanya membuka SSH/HTTP/HTTPS sesuai kebutuhan.

Contoh command service sementara:

```bash
uvicorn main:app --host 127.0.0.1 --port 8501
```

Jangan menggunakan `--reload` pada production.

## Keamanan Repository

File berikut sengaja tidak disimpan di Git:

- `.env`
- `config.json`
- `data/`
- `vector_db_dir/`
- `tessdata/`
- PDF, JSON runtime, SQLite, log, cache, dan file sementara

Folder `public/` hanya berisi aset yang boleh dilihat browser. Jangan menaruh credential atau dokumen privat di folder tersebut.

## Status Repository

Repository GitHub utama:

```text
https://github.com/elpitaa/chatbot
```

Branch utama: `main`.

Perubahan lokal yang sudah diperiksa dapat diunggah dengan:

```bash
git status
git add README.md config.example.json requirements.txt .gitignore
git commit -m "Improve deployment documentation and configuration"
git push origin main
```

Perintah push harus dijalankan secara sadar setelah memeriksa `git diff --cached` dan memastikan tidak ada secret.
