# Script Interaktif Chatbot via Terminal (Tanpa Streamlit)
import os
os.environ["USE_TF"] = "0"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from main import setup_vectorstore, chat_chain, COLLEGE_NAME

def main():
    print("=" * 65)
    print(f"🎓 Chatbot Informasi {COLLEGE_NAME} (Terminal CLI)")
    print("=" * 65)
    print("Ketik 'exit' atau 'keluar' untuk mengakhiri percakapan.\n")

    print("Memuat database vektor...")
    vectorstore = setup_vectorstore()
    chain = chat_chain(vectorstore)
    print("Siap! Silakan ajukan pertanyaan Anda.\n")

    while True:
        try:
            user_input = input("Anda: ")
            if user_input.strip().lower() in ["exit", "keluar", "q"]:
                print(f"\nTerima kasih telah menggunakan Chatbot {COLLEGE_NAME}!")
                break
            if not user_input.strip():
                continue

            response = chain({"question": user_input})
            print(f"\nBot: {response['answer']}\n")
            print("-" * 65)
        except KeyboardInterrupt:
            print("\n\nPercakapan dihentikan.")
            break
        except Exception as e:
            print(f"\nTerjadi kesalahan: {str(e)}\n")

if __name__ == "__main__":
    main()
