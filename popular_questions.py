import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Lokasi folder proyek dan file JSON untuk menyimpan data pertanyaan.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "popular_questions.json"

# Lock mencegah dua request menulis file JSON pada waktu bersamaan.
_lock = threading.Lock()

# Pertanyaan awal yang ditampilkan jika data riil belum mencukupi.
DEFAULT_POPULAR_QUESTIONS = [
    {
        "id": "def_1",
        "title": "Syarat Pengajuan Judul & Sidang Skripsi",
        "prompt": "Syarat Pengajuan Judul & Sidang Skripsi",
        "subtitle": "Tahapan berkas, komisi etik & jadwal pendaftaran",
        "count": 0,
        "is_fallback": True,
    },
    {
        "id": "def_2",
        "title": "Alur KRS & Prasyarat Mata Kuliah Wajib",
        "prompt": "Alur KRS & Prasyarat Mata Kuliah Wajib",
        "subtitle": "Prasyarat SKS, validasi DPA, dan pembagian kelas",
        "count": 0,
        "is_fallback": True,
    },
    {
        "id": "def_3",
        "title": "Batas Maksimal Skor Similarity Turnitin",
        "prompt": "Batas Maksimal Skor Similarity Turnitin",
        "subtitle": "Toleransi persentase & verifikasi bebas plagiasi",
        "count": 0,
        "is_fallback": True,
    },
    {
        "id": "def_4",
        "title": "Jadwal & Prosedur Peminjaman Lab Riset",
        "prompt": "Jadwal & Prosedur Peminjaman Lab Riset",
        "subtitle": "Form izin lab cerdas, rekayasa web, & jaringan",
        "count": 0,
        "is_fallback": True,
    },
    {
        "id": "def_5",
        "title": "Jadwal Layanan Konsultasi Koordinator TA",
        "prompt": "Jadwal Layanan Konsultasi Koordinator TA",
        "subtitle": "Jam kerja luring, link booking, dan ruang prodi",
        "count": 0,
        "is_fallback": True,
    },
    {
        "id": "def_6",
        "title": "Panduan Kerja Praktik (KP) & Magang",
        "prompt": "Panduan Kerja Praktik (KP) & Magang",
        "subtitle": "Surat pengantar, konversi SKS MSIB & format laporan",
        "count": 0,
        "is_fallback": True,
    },
]

# Kata umum diabaikan agar pencocokan fokus pada istilah penting.
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
    "diterbitkan", "keluar", "ikut", "mengikuti"
}


def _clean_text(text: str) -> str:
    """Membersihkan teks dan normalisasi spasi."""
    # Samakan huruf, tanda baca, dan spasi agar pertanyaan mudah dibandingkan.
    text = re.sub(r"[^\w\s-]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _simple_stem(word: str) -> str:
    """Stemming sederhana untuk kata bahasa Indonesia (pengajuan -> aju, mengajukan -> aju)."""
    # Kurangi imbuhan umum agar variasi kata dapat dianggap mirip.
    w = word.lower()
    for prefix in ["meng", "meny", "men", "mem", "me", "peng", "peny", "pen", "pem", "pe", "di", "ter", "ber", "per", "se"]:
        if w.startswith(prefix) and len(w) - len(prefix) >= 3:
            w = w[len(prefix):]
            break
    for suffix in ["kan", "an", "i", "nya", "lah", "kah"]:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[:-len(suffix)]
            break
    return w


def _extract_keywords(text: str) -> set:
    """Ekstraksi kata kunci penting dan bentuk stem dasarnya dari pertanyaan."""
    # Ambil kata minimal tiga karakter, buang stopwords, lalu tambahkan bentuk stem.
    cleaned = _clean_text(text)
    tokens = set(re.findall(r"\b[a-zA-Z0-9-]{3,}\b", cleaned))
    keywords = set()
    for t in tokens:
        if t not in STOPWORDS:
            keywords.add(t)
            stem = _simple_stem(t)
            if len(stem) >= 3 and stem not in STOPWORDS:
                keywords.add(stem)
    return keywords


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Menghitung cosine similarity antara dua vektor embedding."""
    # Cosine similarity mengukur kemiripan arah dua vektor embedding.
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_embedding(text: str, embedding_model=None) -> Optional[List[float]]:
    """Mendapatkan vektor embedding dari model embedding secara aman dan fleksibel."""
    # Dukung beberapa bentuk API model embedding yang mungkin digunakan.
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
        # Jika embedding gagal, proses tetap dapat memakai pencocokan keyword.
        return None
    return None


def _load_data() -> Dict[str, Any]:
    """Membaca file data pertanyaan populer."""
    # Buat folder data jika belum ada.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        # Struktur awal berisi log mentah dan cluster kosong.
        return {"raw_logs": [], "clusters": []}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Data rusak dianggap kosong agar aplikasi tetap dapat berjalan.
        return {"raw_logs": [], "clusters": []}


def _save_data(data: Dict[str, Any]):
    """Menyimpan file data pertanyaan populer."""
    # Tulis ke file sementara terlebih dahulu, lalu ganti file utama secara atomik.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = DATA_FILE.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    temp_file.replace(DATA_FILE)


def _format_canonical_title(text: str) -> str:
    """Format teks pertanyaan agar rapi menjadi judul kartu."""
    # Rapikan tanda tanya dan kapitalisasi untuk judul kartu frontend.
    text = text.strip()
    text = re.sub(r"\?+$", "", text).strip()
    if not text:
        return "Pertanyaan Mahasiswa"
    return text[0].upper() + text[1:]


def record_user_question(question: str, embedding_model=None):
    """
    Menyimpan dan mengelompokkan pertanyaan user berdasarkan kesamaan maksud.
    Menggunakan Cosine Similarity embedding yang ketat atau Jaccard overlap berbobot
    untuk mencegah penggabungan topik yang berbeda (mencegah snowball effect).
    """
    # Abaikan pertanyaan kosong atau terlalu pendek.
    clean_q = question.strip()
    if not clean_q or len(clean_q) < 4:
        return

    with _lock:
        # Seluruh proses baca, clustering, dan simpan dilakukan secara aman.
        data = _load_data()
        raw_logs = data.get("raw_logs", [])
        clusters = data.get("clusters", [])

        # Simpan maksimal 1000 pertanyaan asli sebagai bahan re-clustering.
        now = time.time()
        raw_logs.append({
            "timestamp": now,
            "question": clean_q,
        })
        if len(raw_logs) > 1000:
            raw_logs = raw_logs[-1000:]

        query_cleaned = _clean_text(clean_q)
        query_keywords = _extract_keywords(clean_q)
        query_vector = _get_embedding(clean_q, embedding_model)

        matched_cluster = None
        best_similarity = 0.0

        for cluster in clusters:
            # Cari cluster terbaik menggunakan exact match, embedding, dan keyword.
            sim = 0.0
            is_match = False

            canonical_clean = _clean_text(cluster.get("canonical_question", ""))
            cluster_examples_clean = [_clean_text(ex) for ex in cluster.get("examples", [])]

            # 1. Exact match adalah kecocokan paling kuat.
            if query_cleaned == canonical_clean or query_cleaned in cluster_examples_clean:
                is_match = True
                sim = 1.0

            # 2. Semantic similarity membandingkan makna pertanyaan.
            cluster_vector = cluster.get("embedding")
            if not is_match and query_vector and cluster_vector:
                cos_sim = _cosine_similarity(query_vector, cluster_vector)
                # Ambang batas semantic: >= 0.83 sangat mirip
                if cos_sim >= 0.83:
                    is_match = True
                    sim = cos_sim

            # 3. Jaccard keyword menjadi penguat atau fallback tanpa embedding.
            cluster_keywords = set(cluster.get("keywords", []))
            if query_keywords and cluster_keywords:
                matched_set = query_keywords & cluster_keywords
                intersect_count = len(matched_set)
                union_count = len(query_keywords | cluster_keywords)
                keyword_score = intersect_count / union_count if union_count > 0 else 0.0

                # Jika ada embedding tetapi cosine sim sedang (0.74 - 0.83), konfirmasi dengan overlap keyword
                if not is_match and query_vector and cluster_vector:
                    if cos_sim >= 0.74 and keyword_score >= 0.35 and intersect_count >= 2:
                        is_match = True
                        sim = cos_sim

                # Fallback jika tanpa embedding: butuh kecocokan ketat (Jaccard >= 0.50 dan min 2 kata, atau min 3 kata)
                if not is_match and (not query_vector or not cluster_vector):
                    if (keyword_score >= 0.50 and intersect_count >= 2) or (intersect_count >= 3 and keyword_score >= 0.40):
                        is_match = True
                        sim = keyword_score

            if is_match and sim > best_similarity:
                best_similarity = sim
                matched_cluster = cluster

        if matched_cluster:
            # Tambahkan pertanyaan ke cluster yang dianggap memiliki maksud sama.
            matched_cluster["count"] = matched_cluster.get("count", 0) + 1
            matched_cluster["last_updated"] = now

            # Simpan frekuensi setiap variasi kalimat.
            q_counts = matched_cluster.setdefault("question_counts", {})
            q_counts[clean_q] = q_counts.get(clean_q, 0) + 1

            # Simpan maksimal 10 contoh untuk representasi cluster.
            examples = matched_cluster.get("examples", [])
            if clean_q not in examples:
                examples.append(clean_q)
                if len(examples) > 10:
                    examples = examples[-10:]
            matched_cluster["examples"] = examples

            # Gunakan variasi yang paling sering ditanyakan sebagai judul utama.
            best_variant = max(q_counts.keys(), key=lambda k: (q_counts[k], len(k)))
            matched_cluster["canonical_question"] = _format_canonical_title(best_variant)

            # Hitung ulang keyword dari judul agar cluster tidak dipenuhi kata acak.
            canonical_kw = _extract_keywords(matched_cluster["canonical_question"])
            matched_cluster["keywords"] = list(canonical_kw)[:15]

            # Perbarui embedding cluster dengan moving average agar tetap stabil.
            if query_vector and cluster_vector and len(query_vector) == len(cluster_vector):
                updated_vec = [0.90 * a + 0.10 * b for a, b in zip(cluster_vector, query_vector)]
                norm = math.sqrt(sum(v * v for v in updated_vec))
                if norm > 0:
                    matched_cluster["embedding"] = [v / norm for v in updated_vec]
            elif query_vector and not cluster_vector:
                matched_cluster["embedding"] = query_vector

        else:
            # Jika tidak ada kecocokan, buat cluster pertanyaan baru.
            new_id = f"cl_{int(now)}_{len(clusters)+1}"
            new_cluster = {
                "id": new_id,
                "canonical_question": _format_canonical_title(clean_q),
                "count": 1,
                "question_counts": {clean_q: 1},
                "keywords": list(query_keywords)[:15],
                "examples": [clean_q],
                "last_updated": now,
            }
            if query_vector:
                new_cluster["embedding"] = query_vector
            clusters.append(new_cluster)

        data["raw_logs"] = raw_logs
        data["clusters"] = clusters
        # Simpan hasil clustering ke file JSON.
        _save_data(data)


def rebuild_clusters_from_raw_logs(embedding_model=None):
    """
    Membangun ulang seluruh cluster dari data mentah `raw_logs` yang ada.
    Dapat digunakan untuk membersihkan cluster yang rusak atau menggembung akibat algoritma lama.
    """
    # Hapus cluster lama, tetapi pertahankan log mentah sebagai sumber pembangunan ulang.
    with _lock:
        data = _load_data()
        raw_logs = data.get("raw_logs", [])
        # Reset cluster agar seluruh pengelompokan dihitung dari awal.
        data["clusters"] = []
        _save_data(data)

    print(f"Memulai re-clustering {len(raw_logs)} pertanyaan mentah...")
    for entry in raw_logs:
        q = entry.get("question", "")
        if q:
            # Masukkan kembali setiap pertanyaan melalui algoritma terbaru.
            record_user_question(q, embedding_model=embedding_model)

    with _lock:
        refreshed_data = _load_data()
        total_clusters = len(refreshed_data.get("clusters", []))
    print(f"Re-clustering selesai! Dihasilkan {total_clusters} kelompok pertanyaan yang valid.")
    return total_clusters


def get_popular_questions(limit: int = 6) -> List[Dict[str, Any]]:
    """
    Mengambil daftar pertanyaan terpopuler berdasarkan frekuensi tertinggi.
    Jika data belum mencukupi (kurang dari limit), lengkapi dengan pertanyaan default (fallback).
    """
    # Ambil snapshot data agar proses pembacaan tidak bertabrakan dengan penulisan.
    with _lock:
        data = _load_data()
        clusters = data.get("clusters", [])

    # Prioritaskan cluster dengan jumlah pertanyaan terbanyak dan update terbaru.
    sorted_clusters = sorted(
        clusters,
        key=lambda c: (c.get("count", 0), c.get("last_updated", 0)),
        reverse=True
    )

    result = []
    seen_prompts = set()

    for c in sorted_clusters:
        # Hindari menampilkan judul pertanyaan yang sama dua kali.
        prompt_text = c.get("canonical_question") or (c.get("examples") or [""])[0]
        prompt_key = _clean_text(prompt_text)
        if prompt_key in seen_prompts:
            continue
        seen_prompts.add(prompt_key)

        count = c.get("count", 1)
        rank = len(result) + 1
        if count == 1:
            sub = "Ditanyakan 1x oleh mahasiswa"
        else:
            sub = f"Ditanyakan {count}x oleh mahasiswa"

        result.append({
            # Format ini dibaca frontend untuk membuat kartu pertanyaan populer.
            "id": c.get("id"),
            "title": prompt_text,
            "prompt": prompt_text,
            "subtitle": sub,
            "count": count,
            "rank": rank,
            "is_top": rank == 1,
            "is_fallback": False,
        })
        if len(result) >= limit:
            break

    # Lengkapi hasil dengan pertanyaan default jika data riil belum mencapai limit.
    if len(result) < limit:
        for fallback in DEFAULT_POPULAR_QUESTIONS:
            fallback_key = _clean_text(fallback["prompt"])
            if fallback_key not in seen_prompts:
                rank = len(result) + 1
                item = dict(fallback)
                item["rank"] = rank
                item["is_top"] = rank == 1
                result.append(item)
                seen_prompts.add(fallback_key)
            if len(result) >= limit:
                break

    return result[:limit]
