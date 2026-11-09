# ==============================================================================
# UNANSWERED_QUESTIONS.PY - MODUL MANAJEMEN RIWAYAT PERTANYAAN & BELUM TERJAWAB
# ==============================================================================
# Mengelola penyimpanan, pengelompokan (clustering) semantik, riwayat semua
# pertanyaan mahasiswa, dan tindak lanjut pertanyaan belum terjawab untuk Admin.
# ==============================================================================
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
# Data riwayat pertanyaan disimpan sebagai JSON di folder data.
DATA_DIR = BASE_DIR / "data"
QUESTIONS_FILE = DATA_DIR / "unanswered_questions.json"

# Mencegah dua request mengubah file riwayat secara bersamaan.
_lock = threading.Lock()

# Kategori utama untuk membedakan hasil chatbot.
CATEGORY_UNANSWERED = "Belum Terjawab"
CATEGORY_ANSWERED   = "Terjawab"

# Status yang ditampilkan dan diubah admin pada dashboard.
STATUS_UNPROCESSED    = "Belum Diproses"
STATUS_PROCESSING     = "Diproses"
STATUS_DONE           = "Selesai"
STATUS_AUTO_ANSWERED  = "Terjawab Otomatis"

# Kata umum diabaikan saat membandingkan isi pertanyaan.
STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "adalah", "untuk", "ini", "itu", "pada", "dengan",
    "sebagai", "oleh", "dalam", "atau", "apa", "apakah", "bagaimana", "gimana", "kapan",
    "dimana", "kenapa", "mengapa", "berapa", "siapa", "tolong", "bisa", "mohon", "halo",
    "hai", "permisi", "admin", "kak", "min", "bot", "tanya", "mau", "ingin", "info",
    "tentang", "soal", "dong", "sih", "ya", "kah", "deh", "ada", "tidak", "belum",
    "saya", "kami", "kita", "kamu", "anda", "saja", "secara", "seperti", "yaitu",
    "antara", "lain", "banyak", "setiap", "menurut", "berdasarkan", "tersebut", "sebut",
    "suatu", "agar", "supaya", "jika", "bila", "kalau", "apabila", "sudah", "akan",
    "telah", "sedang", "harus", "wajib", "boleh", "dapat", "sama", "bagi", "hal",
    "maaf", "informasi", "spesifik", "mengenai", "tersedia", "saat", "panduan",
    "resmi", "tercantum", "dokumen"
}

# Jika response mengandung salah satu frasa ini, pertanyaan dianggap belum terjawab.
UNANSWERED_INDICATORS = [
    "maaf, informasi spesifik mengenai hal tersebut belum tercantum",
    "belum tercantum dalam dokumen panduan resmi",
    "tidak ditemukan dalam konteks",
    "tidak ada informasi yang relevan",
    "tidak dapat menemukan jawaban",
    "informasi yang anda tanyakan tidak tersedia",
    "di luar konteks informasi resmi",
]


def _clean_text(text: str) -> str:
    """Pembersihan teks untuk normalisasi perbandingan string."""
    # Samakan huruf, hilangkan tanda baca, dan rapikan spasi.
    text = re.sub(r"[^\w\s-]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _extract_keywords(text: str) -> set:
    """Mengambil token kata kunci penting dari teks tanpa stopwords."""
    # Ambil kata penting minimal tiga karakter untuk pencocokan keyword.
    cleaned = _clean_text(text)
    tokens = set(re.findall(r"\b[a-zA-Z0-9-]{3,}\b", cleaned))
    return {t for t in tokens if t not in STOPWORDS}


def _cosine_similarity(vec_a, vec_b) -> float:
    """Menghitung cosine similarity antara dua vektor."""
    # Nilai semakin mendekati 1 berarti arah/kemiripan embedding semakin dekat.
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_embedding(text: str, embedding_model=None):
    """Menghasilkan embedding teks dari model jika tersedia."""
    # Mendukung model yang menyediakan embed_query, embed_documents, atau callable.
    if not embedding_model:
        return None
    try:
        if hasattr(embedding_model, "embed_query"):
            return embedding_model.embed_query(text)
        elif hasattr(embedding_model, "embed_documents"):
            res = embedding_model.embed_documents([text])
            return res[0] if res else None
        elif callable(embedding_model):
            res = embedding_model([text])
            return res[0] if isinstance(res, list) else res
    except Exception:
        # Jika embedding gagal, sistem tetap dapat memakai keyword matching.
        return None
    return None


def _load_data() -> dict:
    """Membaca data file json dengan aman."""
    # Pastikan folder data tersedia sebelum membaca file.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not QUESTIONS_FILE.exists():
        # Struktur awal ketika belum ada riwayat.
        return {"clusters": []}
    try:
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # File rusak/tidak valid diperlakukan sebagai data kosong.
        return {"clusters": []}


def _save_data(data: dict):
    """Menyimpan data file json dengan penulisan atomik."""
    # File sementara mengurangi risiko file utama rusak saat proses penulisan.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = QUESTIONS_FILE.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    temp_file.replace(QUESTIONS_FILE)


def is_unanswered_response(response_text: str) -> bool:
    """Mengecek apakah respon AI merupakan jawaban fallback / tidak ditemukan di dokumen."""
    # Pemeriksaan dilakukan berdasarkan frasa indikator yang sudah ditentukan.
    response_lower = response_text.lower()
    return any(indicator in response_lower for indicator in UNANSWERED_INDICATORS)


def record_question(
    question: str,
    is_unanswered: bool = False,
    bot_response: str = "",
    embedding_model=None
):
    """
    Mencatat pertanyaan ke riwayat pertanyaan terpusat.
    Mengelompokkan (clustering) pertanyaan yang serupa agar tidak terjadi duplikasi.
    Membedakan kategori 'Belum Terjawab' dan 'Terjawab'.
    """
    # Fungsi ini dipanggil setiap kali chatbot memproses pertanyaan.
    clean_q = question.strip()
    if not clean_q or len(clean_q) < 4:
        return

    category = CATEGORY_UNANSWERED if is_unanswered else CATEGORY_ANSWERED
    initial_status = STATUS_UNPROCESSED if is_unanswered else STATUS_AUTO_ANSWERED

    with _lock:
        # Baca cluster lama sebelum mencari pertanyaan yang memiliki maksud sama.
        data = _load_data()
        clusters = data.get("clusters", [])

        now = time.time()
        query_cleaned = _clean_text(clean_q)
        query_keywords = _extract_keywords(clean_q)
        query_vector = _get_embedding(clean_q, embedding_model)

        matched_cluster = None
        best_similarity = 0.0

        for cluster in clusters:
            # Cari kecocokan dengan exact match, embedding, dan keyword.
            sim = 0.0
            is_match = False

            canonical_clean = _clean_text(cluster.get("canonical_question", ""))
            cluster_examples_clean = [_clean_text(ex) for ex in cluster.get("examples", [])]

            # 1. Exact match menjadi kecocokan paling kuat.
            if query_cleaned == canonical_clean or query_cleaned in cluster_examples_clean:
                is_match = True
                sim = 1.0

            # 2. Semantic similarity membandingkan makna pertanyaan.
            cluster_vector = cluster.get("embedding")
            cos_sim = 0.0
            if not is_match and query_vector and cluster_vector:
                cos_sim = _cosine_similarity(query_vector, cluster_vector)
                if cos_sim >= 0.82:
                    is_match = True
                    sim = cos_sim

            # 3. Jaccard similarity membantu menguatkan atau menggantikan embedding.
            cluster_keywords = set(cluster.get("keywords", []))
            if query_keywords and cluster_keywords:
                intersect = len(query_keywords & cluster_keywords)
                union = len(query_keywords | cluster_keywords)
                keyword_score = intersect / union if union > 0 else 0.0

                if not is_match and query_vector and cluster_vector:
                    if cos_sim >= 0.72 and keyword_score >= 0.35 and intersect >= 2:
                        is_match = True
                        sim = cos_sim

                if not is_match and (not query_vector or not cluster_vector):
                    if (keyword_score >= 0.50 and intersect >= 2) or (intersect >= 3 and keyword_score >= 0.40):
                        is_match = True
                        sim = keyword_score

            if is_match and sim > best_similarity:
                best_similarity = sim
                matched_cluster = cluster

        response_snippet = bot_response.strip()
        # Simpan cuplikan jawaban agar admin dapat melihat konteks singkat.
        if len(response_snippet) > 400:
            response_snippet = response_snippet[:397] + "..."

        if matched_cluster:
            # Tambahkan pertanyaan baru ke cluster yang sudah dianggap sama.
            matched_cluster["count"] = matched_cluster.get("count", 0) + 1
            matched_cluster["last_asked"] = now
            if response_snippet:
                matched_cluster["last_response"] = response_snippet

            # Simpan maksimal 15 variasi pertanyaan sebagai contoh cluster.
            examples = matched_cluster.get("examples", [])
            if clean_q not in examples:
                examples.append(clean_q)
                if len(examples) > 15:
                    examples = examples[-15:]
            matched_cluster["examples"] = examples

            # Jadikan variasi paling sering ditanyakan sebagai judul cluster.
            q_counts = matched_cluster.setdefault("question_counts", {})
            q_counts[clean_q] = q_counts.get(clean_q, 0) + 1
            best_variant = max(q_counts.keys(), key=lambda k: (q_counts[k], len(k)))
            matched_cluster["canonical_question"] = best_variant[0].upper() + best_variant[1:]
            matched_cluster["keywords"] = list(_extract_keywords(matched_cluster["canonical_question"]))[:15]

            # Dokumen baru dapat membuat cluster lama berubah menjadi terjawab.
            if not is_unanswered and matched_cluster.get("category") == CATEGORY_UNANSWERED:
                if matched_cluster.get("status") != STATUS_DONE:
                    matched_cluster["category"] = CATEGORY_ANSWERED
                    matched_cluster["status"] = STATUS_AUTO_ANSWERED

            # Perbarui representasi makna cluster secara bertahap.
            if query_vector and cluster_vector and len(query_vector) == len(cluster_vector):
                updated_vec = [0.90 * a + 0.10 * b for a, b in zip(cluster_vector, query_vector)]
                norm = math.sqrt(sum(v * v for v in updated_vec))
                if norm > 0:
                    matched_cluster["embedding"] = [v / norm for v in updated_vec]
            elif query_vector and not cluster_vector:
                matched_cluster["embedding"] = query_vector
        else:
            # Jika tidak ada cluster yang cocok, buat kelompok baru.
            new_id = f"q_{int(now)}_{len(clusters)+1}"
            title = clean_q[0].upper() + clean_q[1:]
            new_cluster = {
                "id": new_id,
                "canonical_question": title,
                "count": 1,
                "category": category,
                "status": initial_status,
                "question_counts": {clean_q: 1},
                "keywords": list(query_keywords)[:15],
                "examples": [clean_q],
                "last_response": response_snippet,
                "created_at": now,
                "last_asked": now,
                "resolved_pdf": None,
                "resolved_at": None,
            }
            if query_vector:
                new_cluster["embedding"] = query_vector
            clusters.append(new_cluster)

        data["clusters"] = clusters
        # Simpan hasil perubahan riwayat ke JSON.
        _save_data(data)


# Alias backward-compatible untuk pertanyaan belum terjawab
def record_unanswered_question(question: str, embedding_model=None):
    # Alias lama agar pemanggil yang hanya mencatat pertanyaan belum terjawab tetap kompatibel.
    record_question(question, is_unanswered=True, embedding_model=embedding_model)


def get_all_unanswered(status_filter: str = None) -> list:
    """Mengambil pertanyaan dengan kategori 'Belum Terjawab'."""
    # Ambil data yang perlu ditindaklanjuti admin.
    with _lock:
        data = _load_data()
        clusters = data.get("clusters", [])

    # Sertakan cluster belum terjawab dan cluster yang masih memiliki status tindak lanjut.
    filtered = [
        c for c in clusters
        if c.get("category") == CATEGORY_UNANSWERED or c.get("status") in [STATUS_UNPROCESSED, STATUS_PROCESSING, STATUS_DONE]
    ]

    status_order = {STATUS_UNPROCESSED: 0, STATUS_PROCESSING: 1, STATUS_DONE: 2}

    if status_filter:
        # Terapkan filter status jika dipilih dari dashboard.
        filtered = [c for c in filtered if c.get("status") == status_filter]

    sorted_clusters = sorted(
        filtered,
        key=lambda c: (
            status_order.get(c.get("status", STATUS_UNPROCESSED), 0),
            -c.get("count", 0),
            -c.get("last_asked", 0),
        )
    )

    result = []
    for c in sorted_clusters:
        # Kembalikan hanya field yang diperlukan oleh frontend admin.
        result.append({
            "id": c.get("id"),
            "canonical_question": c.get("canonical_question", ""),
            "count": c.get("count", 1),
            "category": c.get("category", CATEGORY_UNANSWERED),
            "status": c.get("status", STATUS_UNPROCESSED),
            "created_at": c.get("created_at", 0),
            "last_asked": c.get("last_asked", 0),
            "examples": c.get("examples", []),
            "last_response": c.get("last_response", ""),
            "resolved_pdf": c.get("resolved_pdf"),
            "resolved_at": c.get("resolved_at"),
        })
    return result


def get_all_question_history(
    category_filter: str = None,
    status_filter: str = None,
    search_query: str = None
) -> list:
    """
    Mengambil SEMUA riwayat pertanyaan mahasiswa dengan dukungan filter:
    - category_filter: 'Terjawab' atau 'Belum Terjawab'
    - status_filter: 'Terjawab Otomatis', 'Belum Diproses', 'Diproses', 'Selesai'
    - search_query: pencarian kata pada teks pertanyaan
    """
    # Fungsi ini menyediakan data riwayat untuk tab history admin.
    with _lock:
        data = _load_data()
        clusters = data.get("clusters", [])

    results = []
    q_search = _clean_text(search_query) if search_query else None

    for c in clusters:
        category = c.get("category", CATEGORY_UNANSWERED)
        status   = c.get("status", STATUS_UNPROCESSED)

        if category_filter and category != category_filter:
            continue
        if status_filter and status != status_filter:
            continue

        if q_search:
            # Pencarian dilakukan pada judul canonical dan seluruh contoh pertanyaan.
            title = _clean_text(c.get("canonical_question", ""))
            examples = " ".join(_clean_text(ex) for ex in c.get("examples", []))
            if q_search not in title and q_search not in examples:
                continue

        results.append({
            # Format record disesuaikan dengan kebutuhan tabel/detail frontend.
            "id": c.get("id"),
            "canonical_question": c.get("canonical_question", ""),
            "count": c.get("count", 1),
            "category": category,
            "status": status,
            "created_at": c.get("created_at", 0),
            "last_asked": c.get("last_asked", 0),
            "examples": c.get("examples", []),
            "last_response": c.get("last_response", ""),
            "resolved_pdf": c.get("resolved_pdf"),
            "resolved_at": c.get("resolved_at"),
        })

    # Tampilkan pertanyaan yang paling baru ditanyakan terlebih dahulu.
    results.sort(key=lambda x: -x.get("last_asked", 0))
    return results


def get_unanswered_stats() -> dict:
    """Ringkasan statistik pertanyaan belum terjawab."""
    # Statistik ini digunakan oleh kartu ringkasan tab pertanyaan belum terjawab.
    with _lock:
        data = _load_data()
        clusters = data.get("clusters", [])

    uq_clusters = [
        c for c in clusters
        if c.get("category") == CATEGORY_UNANSWERED or c.get("status") in [STATUS_UNPROCESSED, STATUS_PROCESSING, STATUS_DONE]
    ]

    total = len(uq_clusters)
    unprocessed = sum(1 for c in uq_clusters if c.get("status") == STATUS_UNPROCESSED)
    processing  = sum(1 for c in uq_clusters if c.get("status") == STATUS_PROCESSING)
    done        = sum(1 for c in uq_clusters if c.get("status") == STATUS_DONE)
    total_asks  = sum(c.get("count", 1) for c in uq_clusters)

    # Kembalikan jumlah cluster dan jumlah total kemunculan pertanyaan.
    return {
        "total": total,
        "unprocessed": unprocessed,
        "processing": processing,
        "done": done,
        "total_asks": total_asks,
    }


def get_history_stats() -> dict:
    """Ringkasan statistik seluruh riwayat pertanyaan (Terjawab vs Belum Terjawab)."""
    # Hitung statistik seluruh kategori dan status untuk dashboard history.
    with _lock:
        data = _load_data()
        clusters = data.get("clusters", [])

    total_clusters = len(clusters)
    total_asks     = sum(c.get("count", 1) for c in clusters)
    
    total_answered   = sum(1 for c in clusters if c.get("category") == CATEGORY_ANSWERED)
    total_unanswered = sum(1 for c in clusters if c.get("category") == CATEGORY_UNANSWERED)

    unprocessed = sum(1 for c in clusters if c.get("status") == STATUS_UNPROCESSED)
    processing  = sum(1 for c in clusters if c.get("status") == STATUS_PROCESSING)
    done        = sum(1 for c in clusters if c.get("status") == STATUS_DONE)
    auto_ans    = sum(1 for c in clusters if c.get("status") == STATUS_AUTO_ANSWERED)

    return {
        "total_clusters": total_clusters,
        "total_asks": total_asks,
        "total_answered": total_answered,
        "total_unanswered": total_unanswered,
        "unprocessed": unprocessed,
        "processing": processing,
        "done": done,
        "auto_answered": auto_ans,
    }


def update_status(question_id: str, new_status: str, resolved_pdf: str = None) -> bool:
    """Memperbarui status pertanyaan."""
    # Hanya status yang ada di daftar allowed yang boleh disimpan.
    allowed = [STATUS_UNPROCESSED, STATUS_PROCESSING, STATUS_DONE, STATUS_AUTO_ANSWERED]
    if new_status not in allowed:
        return False

    with _lock:
        data = _load_data()
        clusters = data.get("clusters", [])

        for cluster in clusters:
            if cluster.get("id") == question_id:
                # Update status berdasarkan ID cluster.
                cluster["status"] = new_status
                if new_status == STATUS_DONE:
                    # Catat waktu selesai dan PDF penyelesaian jika tersedia.
                    cluster["resolved_at"] = time.time()
                    if resolved_pdf:
                        cluster["resolved_pdf"] = resolved_pdf
                _save_data(data)
                return True
    return False


def delete_unanswered(question_id: str) -> bool:
    """Menghapus data pertanyaan berdasarkan ID."""
    # Hapus satu cluster pertanyaan dari riwayat berdasarkan ID unik.
    with _lock:
        data = _load_data()
        clusters = data.get("clusters", [])
        original_len = len(clusters)
        clusters = [c for c in clusters if c.get("id") != question_id]
        if len(clusters) == original_len:
            # False berarti ID tidak ditemukan.
            return False
        data["clusters"] = clusters
        # Simpan daftar cluster setelah penghapusan.
        _save_data(data)
    return True
