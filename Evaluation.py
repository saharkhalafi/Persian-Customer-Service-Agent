import pandas as pd
import json
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
#from bert_score import score

with open('updated_dataset_with_rag_answers_and_chunks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

def normalize(s):
    return str(s or '').strip().lower().replace('\n', ' ').replace('  ', ' ')

df['ans_n'] = df['answer'].apply(normalize)
df['gt_n'] = df['ground_truth'].apply(normalize)

# Exact match
exact = (df['ans_n'] == df['gt_n']).mean() * 100

# BLEU
smooth = SmoothingFunction().method1
bleu = []
for a, g in zip(df['ans_n'], df['gt_n']):
    if a and g:
        bleu.append(sentence_bleu([g.split()], a.split(), smoothing_function=smooth))
    else:
        bleu.append(0)
avg_bleu = np.mean(bleu) * 100

# ROUGE-L
sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge = [sc.score(g, a)['rougeL'].fmeasure if a and g else 0 for a, g in zip(df['ans_n'], df['gt_n'])]
avg_rouge = np.mean(rouge) * 100


# BERTScore 
'''P, R, F1 = score(
    df['answer'].tolist(), 
    df['ground_truth'].tolist(), 
    lang="fa",                     # یا "multi" اگر مدل multilingual می‌خوای
    model_type="bert-base-multilingual-cased",  # یا بهتر: "intfloat/multilingual-e5-large-instruct"
    rescale_with_baseline=True,    # برای نمره بهتر و قابل مقایسه
    verbose=True
)
avg_bert_precision = P.mean().item() * 100
avg_bert_recall    = R.mean().item() * 100
avg_bert_f1        = F1.mean().item() * 100
'''
print(f"Exact Match: {exact:.1f}%")
print(f"BLEU: {avg_bleu:.1f}%")
print(f"ROUGE-L F1: {avg_rouge:.1f}%")
#print(f"BERTScore Precision: {avg_bert_precision:.1f}%")
#print(f"BERTScore Recall: {avg_bert_recall:.1f}%")
#print(f"BERTScore F1: {avg_bert_f1:.1f}%")
print(f"تعداد نمونه: {len(df)}")


# define Recall and Precision
df['retrieved_set'] = df['retrieved_chunks'].apply(lambda x: set(x) if isinstance(x, list) else set())
df['relevant_set'] = df['retrieved_chunks'].apply(lambda x: set(x) if isinstance(x, list) else set())

# Hit Rate 
df['hit'] = df.apply(lambda row: 1 if row['retrieved_set'] & row['relevant_set'] else 0, axis=1)
hit_rate = df['hit'].mean() * 100

# Recall 
df['recall'] = df.apply(lambda row: len(row['retrieved_set'] & row['relevant_set']) / len(row['relevant_set']) if len(row['relevant_set']) > 0 else 0, axis=1)
avg_recall = df['recall'].mean() * 100

# Precision
df['precision'] = df.apply(lambda row: len(row['retrieved_set'] & row['relevant_set']) / len(row['retrieved_set']) if len(row['retrieved_set']) > 0 else 0, axis=1)
avg_precision = df['precision'].mean() * 100

print(f"Hit Rate@5: {hit_rate:.1f}%")
print(f"Recall@5:   {avg_recall:.1f}%")
print(f"Precision@5: {avg_precision:.1f}%")