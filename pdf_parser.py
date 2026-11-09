import os
import shutil
from pathlib import Path

# Library untuk membuka PDF, menjalankan OCR, mengolah gambar, dan membaca teks PDF.
import pymupdf
import pytesseract
from PIL import Image
from pypdf import PdfReader

# Batas minimum untuk menentukan apakah ekstraksi teks PDF dianggap berhasil.
MIN_TOTAL_CHARACTERS = 100
MIN_AVERAGE_CHARACTERS_PER_PAGE = 40
# Lokasi data bahasa Tesseract lokal dan lokasi cadangan.
LOCAL_TESSDATA_DIR = Path(__file__).resolve().parent / "tessdata"
DEFAULT_TESSDATA_DIR = Path(r"D:\College_RAG_Chatbot\tessdata")

# Prioritaskan folder Tesseract dari environment variable jika tersedia.
if "TESSDATA_DIR" in os.environ and Path(os.environ["TESSDATA_DIR"]).is_dir():
    TESSDATA_DIR = Path(os.environ["TESSDATA_DIR"]).resolve()
# Jika tidak ada, gunakan folder tessdata yang berada di proyek.
elif LOCAL_TESSDATA_DIR.is_dir():
    TESSDATA_DIR = LOCAL_TESSDATA_DIR.resolve()
# Gunakan lokasi lama sebagai fallback terakhir.
elif DEFAULT_TESSDATA_DIR.is_dir():
    TESSDATA_DIR = DEFAULT_TESSDATA_DIR.resolve()
else:
    TESSDATA_DIR = LOCAL_TESSDATA_DIR.resolve()

if TESSDATA_DIR.is_dir():
    # Beri tahu Tesseract lokasi file bahasa seperti ind.traineddata.
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)


def configure_tesseract():
    # Cari executable Tesseract dari konfigurasi, lokasi Windows umum, atau PATH.
    configured_path = os.getenv("TESSERACT_CMD")
    candidates = [
        configured_path,
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"D:\Program Files\Tesseract-OCR\tesseract.exe",
        r"D:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"D:\Tesseract-OCR\tesseract.exe",
        r"D:\tesseract\tesseract.exe",
    ]
    for candidate in candidates:
        # Gunakan kandidat pertama yang benar-benar merupakan file executable.
        if candidate and Path(candidate).is_file():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate
    path_command = shutil.which("tesseract")
    if path_command:
        # shutil.which menemukan Tesseract jika sudah terdaftar di PATH.
        pytesseract.pytesseract.tesseract_cmd = path_command
        return path_command
    return pytesseract.pytesseract.tesseract_cmd


def _extract_text_with_pypdf(pdf_path):
    # Ekstraksi cepat untuk PDF yang memiliki layer teks.
    reader = PdfReader(str(pdf_path))
    # Ambil teks setiap halaman dan gabungkan dengan pemisah baris.
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(page for page in pages if page)
    return len(reader.pages), text


def _extract_text_with_ocr(pdf_path):
    # OCR digunakan untuk PDF scan atau PDF yang teksnya tidak terbaca pypdf.
    tesseract_path = configure_tesseract()
    if not Path(tesseract_path).is_file() and not shutil.which(tesseract_path):
        # Gagal secara jelas jika Tesseract belum terpasang.
        raise RuntimeError(
            "Tesseract OCR tidak ditemukan. Install Tesseract lalu set TESSERACT_CMD."
        )

    if TESSDATA_DIR.is_dir():
        os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)

    page_texts = []
    with pymupdf.open(str(pdf_path)) as document:
        # Render setiap halaman PDF menjadi gambar beresolusi lebih tinggi.
        page_count = len(document)
        for page_number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            try:
                # Coba OCR dengan bahasa Indonesia dan Inggris.
                text = pytesseract.image_to_string(image, lang="ind+eng")
            except Exception:
                try:
                    # Fallback ke bahasa Indonesia jika kombinasi bahasa gagal.
                    text = pytesseract.image_to_string(image, lang="ind")
                except Exception:
                    # Fallback terakhir hanya menggunakan bahasa Inggris.
                    text = pytesseract.image_to_string(image, lang="eng")
            page_texts.append(text.strip())
            print(f"  OCR page {page_number}/{page_count}")
    return page_count, "\n\n".join(page for page in page_texts if page)


def extract_pdf(pdf_path):
    """Extract one PDF with pypdf first, then OCR scan/image PDFs when needed."""
    # Fungsi utama: pilih metode TEXT atau OCR dan kembalikan metadata hasilnya.
    pdf_path = Path(pdf_path)
    page_count = 0
    text = ""
    method = "TEXT"
    character_count = 0
    try:
        # Tahap pertama selalu mencoba membaca layer teks menggunakan pypdf.
        page_count, text = _extract_text_with_pypdf(pdf_path)
        character_count = len(text)
        average_characters = character_count / page_count if page_count else 0

        if character_count >= MIN_TOTAL_CHARACTERS and average_characters >= MIN_AVERAGE_CHARACTERS_PER_PAGE:
            # Teks cukup banyak sehingga hasil pypdf dianggap valid.
            method = "TEXT"
        else:
            # Teks terlalu sedikit; anggap PDF berupa scan dan jalankan OCR.
            method = "OCR"
            page_count, text = _extract_text_with_ocr(pdf_path)
            character_count = len(text)

        if not text.strip():
            # PDF tanpa hasil teks tidak dapat dimasukkan ke knowledge base.
            raise ValueError("Tidak ada teks yang berhasil diekstrak.")
    except Exception:
        # Cetak informasi diagnostik lalu teruskan exception kepada pemanggil.
        print(
            f"PDF: {pdf_path.name} | halaman: {page_count} | metode: {method} | "
            f"karakter: {character_count} | status: GAGAL"
        )
        raise

    print(
        # Log ringkas membantu admin mengetahui metode dan hasil ekstraksi.
        f"PDF: {pdf_path.name} | halaman: {page_count} | metode: {method} | "
        f"karakter: {character_count} | status: BERHASIL"
    )
    return {
        # Data ini dipakai main.py untuk chunking dan metadata vector store.
        "text": text,
        "page_count": page_count,
        "method": method,
        "character_count": character_count,
    }


def load_pdf_documents(directory):
    """Process every PDF in a directory and continue after individual failures."""
    # Muat semua PDF dalam folder menjadi objek Document LangChain.
    from langchain.schema import Document

    directory = Path(directory)
    documents = []
    # Urutkan file agar proses dan hasilnya konsisten.
    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        # Kembalikan list kosong jika folder tidak berisi PDF.
        print(f"Tidak ada file PDF di {directory}")
        return documents

    for pdf_path in pdf_files:
        # Satu file gagal tidak menghentikan pemrosesan file berikutnya.
        print(f"Memproses: {pdf_path.name}")
        try:
            result = extract_pdf(pdf_path)
            documents.append(
                Document(
                    # page_content menjadi isi yang nantinya dipecah dan di-embed.
                    page_content=result["text"],
                    metadata={
                        # Metadata membantu melacak sumber setiap chunk dokumen.
                        "source": pdf_path.name,
                        "pages": result["page_count"],
                        "extraction_method": result["method"],
                        "character_count": result["character_count"],
                    },
                )
            )
        except Exception as error:
            # Catat error file tersebut lalu lanjutkan ke PDF berikutnya.
            print(
                f"PDF: {pdf_path.name} | status: GAGAL | error: {error} | "
                "lanjut ke file berikutnya"
            )

    # Laporkan jumlah PDF yang berhasil dibandingkan total file.
    print(f"Selesai: {len(documents)}/{len(pdf_files)} PDF berhasil diproses.")
    return documents
