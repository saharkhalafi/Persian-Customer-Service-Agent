# generate_answers_from_excel.py
import pandas as pd
import json
import sys
from pathlib import Path

# مسیر پروژه
sys.path.append(str(Path(__file__).parent))

from Generator_core import rag_chain

file_path = r"E:\cursor projects\RAG FAQ modiseh\Test generator.xlsx"

try:
    df = pd.read_excel(file_path, engine="openpyxl")
    print("دیتاست خوانده شد | شکل:", df.shape)
    print("ستون‌ها:", list(df.columns))

    q_col = None
    for col in df.columns:
        col_str = str(col).strip().lower()
        if 'سوال' in col_str or 'question' in col_str or 'پرسش' in col_str:
            q_col = col
            break

    if q_col is None:
        print("ستون سوال پیدا نشد. ستون‌های موجود:", df.columns.tolist())
        sys.exit(1)

    print(f"→ استفاده از ستون سوال: '{q_col}'")

    dataset = df.to_dict(orient="records")

    def get_rag_answer_and_chunks(question):
        if pd.isna(question) or not str(question).strip():
            return "سوال خالی یا نامعتبر", []

        try:
            result = rag_chain.invoke({"input": str(question).strip()})
            answer = result.get("answer", "پاسخ تولید نشد").strip()

            # Extract chunk_id
            retrieved_chunks = []
            for doc in result.get("context", []):
                cid = doc.metadata.get("chunk_id")
                if cid is not None:
                    retrieved_chunks.append(int(cid))  
                else:
                    print("هشدار: سند بدون chunk_id پیدا شد!")

            return answer, sorted(set(retrieved_chunks)) 

        except Exception as e:
            return f"خطا در تولید پاسخ: {str(e)}", []

    # process each question and update dataset
    for i, row in enumerate(dataset, 1):
        q = row.get(q_col, "")
        print(f"[{i}/{len(dataset)}] {str(q)[:70]}...")

        answer, chunk_ids = get_rag_answer_and_chunks(q)
        row["answer"] = answer
        row["retrieved_chunks"] = chunk_ids   


    output_file = "updated_dataset_with_rag_answers_and_chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nفایل به‌روزرسانی‌شده ذخیره شد: {output_file}")
    print(f"تعداد سوالات پردازش‌شده: {len(dataset)}")

except FileNotFoundError:
    print(f"فایل اکسل پیدا نشد:\n{file_path}")
except Exception as e:
    print(f"خطای کلی: {str(e)}")