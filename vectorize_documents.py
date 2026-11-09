import os
os.environ["USE_TF"] = "0"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pdf_parser import load_pdf_documents

def main():
    # Memastikan direktori data sudah ada
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory. Please add your PDF files here.")
        return

    # Memastikan direktori database vector sudah ada
    if not os.path.exists("vector_db_dir"):
        os.makedirs("vector_db_dir")
        print("Created 'vector_db_dir' directory for storing vectorized documents.")

    try:
        # Memuat model embedding
        print("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings()
        
        # Memuat dan memproses dokumen-dokumen PDF
        print("Loading and processing PDF documents...")
        documents = load_pdf_documents("data")
        
        if not documents:
            print("No documents were successfully processed. Please check your PDF files.")
            return
            
        print(f"Successfully loaded {len(documents)} documents")

        # Memotong dokumen menjadi bagian-bagian kecil (chunks)
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=500,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        text_chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(text_chunks)} chunks")

        # Membuat dan menyimpan database vector
        print("Creating vector database...")
        vectordb = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            persist_directory="vector_db_dir"
        )
        
        print("Successfully vectorized and stored documents in 'vector_db_dir'")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please ensure all required dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
