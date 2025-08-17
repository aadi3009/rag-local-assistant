import json
import textwrap
import numpy as np
import pandas as pd
import torch
from sentence_transformers import util, SentenceTransformer

# === Paths ===
embeddings_df_save_path = r"D:\AI projects\newglados\GlaDOS\RAG_v1\Files_embeddings\text_chunks_and_embeddings_df.csv"

# === Device & model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)

# === Utils ===
def print_wrapped(text, width=100):
    print(textwrap.fill(text, width=width))

def parse_embedding_cell(x) -> np.ndarray:
    """
    Parse an embedding cell that may be stored as:
    - JSON list string: "[0.12, -0.34, ...]"
    - Python list string: "[0.12, -0.34, ...]"
    - Already a list/array
    Fallbacks to comma then space separation if json fails.
    """
    if isinstance(x, (list, np.ndarray)):
        return np.asarray(x, dtype=np.float32)

    s = str(x).strip()
    # Fast path: JSON
    try:
        arr = np.asarray(json.loads(s), dtype=np.float32)
        if arr.size > 0:
            return arr
    except Exception:
        pass

    # Fallbacks: comma-separated, then space-separated
    arr = np.fromstring(s.strip("[]"), sep=",", dtype=np.float32)
    if arr.size == 0:
        arr = np.fromstring(s.strip("[]"), sep=" ", dtype=np.float32)

    if arr.size == 0:
        raise ValueError(f"Could not parse embedding cell: {s[:120]}...")
    return arr

def retrieve_relevant_resources(query: str,
                                embeddings: torch.Tensor,
                                model: SentenceTransformer,
                                n_resources_to_return: int = 5):
    """
    Embed query and return top-k cosine similarities & indices.
    Note: embeddings are assumed to be already normalized (as saved),
    so we normalize the query too.
    """
    q = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(q, embeddings)[0]
    topk = torch.topk(scores, k=min(n_resources_to_return, embeddings.shape[0]))
    return topk.values, topk.indices

def prompt_formatter(query: str, context_items: list[dict]) -> str:
    context = "- " + "\n- ".join([item.get("sentence_chunk", "") for item in context_items])
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
keep answers as concise as possible. avoid contradictions, and make sure to be correct.
state only one opinion, and state any alternate opinions as alternatives.

Now use the following context items to answer the user query:
{context}

Relevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""
    return base_prompt.format(context=context, query=query)

def rag_add_context(query: str,
                    embeddings: torch.Tensor,
                    embedding_model: SentenceTransformer,
                    pages_and_chunks: list,
                    top_k: int = 5) -> str:
    scores, indices = retrieve_relevant_resources(query, embeddings, embedding_model, n_resources_to_return=top_k)
    idx_list = indices.detach().cpu().tolist()
    context_items = [pages_and_chunks[i] for i in idx_list]
    prompt = prompt_formatter(query=query, context_items=context_items)

    # Optional: print top contexts with scores
    print("\n[top contexts]")
    sc = scores.detach().cpu().tolist()
    for rank, (i, s) in enumerate(zip(idx_list, sc), start=1):
        it = pages_and_chunks[i]
        title = it.get("document_name", "") or it.get("source_title", "") or ""
        print_wrapped(f"{rank:>2}. score={s:.3f} | {title} | page={it.get('page_number', '?')}")
    print()
    return prompt

# === Load CSV and build tensors ===
text_chunks_and_embedding_df = pd.read_csv(embeddings_df_save_path)

if "embedding" not in text_chunks_and_embedding_df.columns:
    raise KeyError("CSV is missing the 'embedding' column.")

# Robust parsing for each row
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(parse_embedding_cell)

# To dicts for context
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

# Stack to (N, D), cast to float32, move to device
emb_matrix = np.vstack(text_chunks_and_embedding_df["embedding"].values).astype(np.float32)
embeddings = torch.tensor(emb_matrix, dtype=torch.float32, device=device)

print(f"Embeddings shape: {embeddings.shape}")
print(text_chunks_and_embedding_df.head(2))

# === Example usage ===
if __name__ == "__main__":
    # Change this query or wire up input() if you prefer interactive use
    query = "what is the difference between adderall and dextroamphetamine?"
    prompt = rag_add_context(query, embeddings, embedding_model, pages_and_chunks, top_k=5)
    print(prompt)
