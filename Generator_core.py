
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Modern imports ───────────────────────────────────────────────────────────────
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# ── 1. Load .env ─────────────────────────────────────────────────────────────────
env_path = Path(__file__).parent / ".env"

if not env_path.is_file():
    print("Error: .env file not found")
    print("Create .env in the same folder with:")
    print("GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    exit(1)

load_dotenv(env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY missing in .env")
    exit(1)

# ── 2. Configuration ─────────────────────────────────────────────────────────────
DB_FOLDER = r"E:\cursor projects\RAG FAQ modiseh\local_chroma_db-2026\local_chroma_db"


# retrieve COLLECTION_NAME from the actual database to avoid mismatch issues
'''
db_path = r"E:\cursor projects\RAG FAQ modiseh\local_chroma_db-2026\local_chroma_db"
client = chromadb.PersistentClient(path=db_path)
print("اسم تمام مجموعه‌های موجود:")
for coll in client.list_collections():
    print(f" • {coll.name}   →   تعداد سند: {coll.count()}")
'''
COLLECTION_NAME = "langchain"

GEMINI_MODEL = "gemini-2.0-flash" 

# ── 3. Dummy Embeddings ──────────────────────────────────────────────────────────
class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.0] * 1024 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 1024

# ── 4. Load Chroma ───────────────────────────────────────────────────────────────
print("Loading Chroma database...")

vector_db = Chroma(
    persist_directory=DB_FOLDER,
    embedding_function=DummyEmbeddings(),
    collection_name=COLLECTION_NAME
)

doc_count = vector_db._collection.count()
print(f"→ Loaded {doc_count:,} documents from '{COLLECTION_NAME}'")

if doc_count == 0:
    print("Warning: Database empty!")
    print("Check:")
    print("  • COLLECTION_NAME correct?")
    print("  • Folder contains chroma.sqlite3 + index files?")
    print("  • Database created with this collection name?")

# ── 5. Gemini LLM ────────────────────────────────────────────────────────────────
print("Initializing Gemini...")

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0.15,
    max_output_tokens=1500,
    safety_settings={
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
)

# ── 6. Prompt ────────────────────────────────────────────────────────────────────
prompt_template = """تو یک دستیار هوشمند FAQ سایت مدیسه هستی.

دستورالعمل‌های مهم:
• فقط و فقط از اطلاعات موجود در Context زیر استفاده کن.
اطلاعاتی که به عنوان فایل ورودی بهت دادم سعی کن دقیقا از اون اطلاعات موجود در اون فایل استفاده کنید
• پاسخ را کوتاه، دقیق، حرفه‌ای و حداکثر در ۲–۴ جمله بده.
• هیچ اطلاعاتی از دانش قبلی خودت یا فرضیات اضافه نکن.
• اعداد، مهلت‌ها، هزینه‌ها و شرایط دقیق را عیناً و بدون تغییر ذکر کن.
• پاسخ را فقط به زبان فارسی بده.

Context:
{context}

سوال کاربر:
{input}

پاسخ (کوتاه و مفید به فارسی):"""

PROMPT = PromptTemplate.from_template(prompt_template)

# rag_generator_local.py
# ... تمام importها و کدهای قبلی بدون تغییر تا اینجا ...

# ── 7. ساخت زنجیره RAG (این بخش همیشه اجرا می‌شود) ────────────────────────────
print("Creating RAG chain...")

combine_docs_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=PROMPT
)

rag_chain = create_retrieval_chain(
    retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
    combine_docs_chain=combine_docs_chain
)

# ── run ───────────────
if __name__ == "__main__":
    print("\nRAG generator آماده است.")
    print("سوال خود را به فارسی وارد کنید (خروج: exit یا quit)\n")

    while True:
        question = input("سوال: ").strip()

        if question.lower() in ["exit", "quit", "خروج", "q"]:
            print("خداحافظ!")
            break

        if not question:
            continue

        print("\nدر حال جستجو و تولید پاسخ...\n")

        try:
            result = rag_chain.invoke({"input": question})

            print("پاسخ:")
            print(result["answer"].strip())
            print("\n" + "─" * 80)

            print("منابع استفاده شده:")
            for i, doc in enumerate(result["context"], 1):
                meta = doc.metadata
                page = meta.get("page", "نامشخص")
                typ  = meta.get("type", "نامشخص")
                topic = meta.get("topic")

                line = f"[{i}] صفحه {page}  |  نوع: {typ}"
                if topic:
                    line += f"  |  موضوع: {topic}"
                print(line)
                print("─" * 60)

            print()

        except Exception as e:
            print("خطا:")
            print(str(e))
            print("\nممکن است مشکل از این موارد باشد:")
            print("  • نام COLLECTION_NAME اشتباه است")
            print("  • دیتابیس خالی یا خراب است")
            print("  • مسیر DB_FOLDER اشتباه است")
            print("  • کلید GEMINI_API_KEY نامعتبر است\n")