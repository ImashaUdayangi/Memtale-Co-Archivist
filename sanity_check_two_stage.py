import pandas as pd
import numpy as np
from src.setl2 import setl2_vector_np
from src.retrieval import two_stage_bm25_rerank_setl2

df = pd.read_csv("dataset/cache/summaries.csv").head(200)  # small
docs = df["aware_summary"].fillna("").astype(str).tolist()

doc_vecs = np.stack([setl2_vector_np(t) for t in docs])

query = "a time I felt proud and confident"
q_vec = setl2_vector_np(query)

ranked_idx, scores = two_stage_bm25_rerank_setl2(
    query_text=query,
    docs_text=docs,
    docs_emotion_vecs=doc_vecs,
    query_emotion_vec=q_vec,
    bm25_top_k=50,
    final_top_k=10,
    query_type="emotion_intent"
)

print("Top 3 results:")
for i in ranked_idx[:3]:
    print("-", docs[i][:120])
