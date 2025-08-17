import os
import re
import pandas as pd
import pymupdf as fitz   # PyMuPDF
from tqdm.auto import tqdm
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
import torch

# -----------------
# Config
# -----------------
pdf_directory = r"D:\AI projects\newglados\GlaDOS\RAG_v1\File_database"
embeddings_df_save_path = r"D:\AI projects\newglados\GlaDOS\RAG_v1\Files_embeddings\text_chunks_and_embeddings_df.csv"

num_sentence_chunk_size = 10
min_token_length = 30

# -----------------
# PDF processing
# -----------------
def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()

def process_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    for page_number, page in enumerate(doc, start=1):
        text = text_formatter(page.get_text())
        pages_and_texts.append({
            "document_name": doc_name,
            "page_number": page_number,
            "page_char_count": len(text),
            "page_word_count": len(text.split()),
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": len(text) / 4,
            "text": text
        })
    return pages_and_texts

# -----------------
# Gather pages
# -----------------
all_pages_and_texts = []
pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]

if not pdf_files:
    raise RuntimeError("No PDF files found in the directory!")

for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
    pdf_path = os.path.join(pdf_directory, pdf_file)
    all_pages_and_texts.extend(process_pdf(pdf_path))

print(f"Total pages processed: {len(all_pages_and_texts)}")

# -----------------
# Sentence splitting
# -----------------
nlp = English()
nlp.add_pipe("sentencizer")

for item in tqdm(all_pages_and_texts, desc="Sentence splitting"):
    sents = list(nlp(item["text"]).sents)
    item["sentences"] = [str(s) for s in sents]
    item["page_sentence_count_spacy"] = len(item["sentences"])

df = pd.DataFrame(all_pages_and_texts)
print(df.describe().round(2))

# -----------------
# Chunking
# -----------------
def split_list(input_list, size):
    return [input_list[i:i + size] for i in range(0, len(input_list), size)]

for item in tqdm(all_pages_and_texts, desc="Chunking sentences"):
    item["sentence_chunks"] = split_list(item["sentences"], num_sentence_chunk_size)
    item["num_chunks"] = len(item["sentence_chunks"])

pages_and_chunks = []
for item in all_pages_and_texts:
    for chunk in item["sentence_chunks"]:
        text = " ".join(chunk).replace("  ", " ").strip()
        text = re.sub(r'\.([A-Z])', r'. \1', text)  # ".A" -> ". A"
        pages_and_chunks.append({
            "document_name": item["document_name"],
            "page_number": item["page_number"],
            "sentence_chunk": text,
            "chunk_char_count": len(text),
            "chunk_word_count": len(text.split()),
            "chunk_token_count": len(text) / 4
        })

df_chunks = pd.DataFrame(pages_and_chunks)
print(df_chunks.describe().round(2))
print(f"Total chunks: {len(df_chunks)}")

# -----------------
# Filter short chunks
# -----------------
short_chunks = df_chunks[df_chunks["chunk_token_count"] <= min_token_length]
if not short_chunks.empty:
    for _, r in short_chunks.sample(min(3, len(short_chunks))).iterrows():
        print(f"Chunk token count: {r['chunk_token_count']:.1f} | Text: {r['sentence_chunk'][:100]}")

pages_and_chunks_over_min_token_len = df_chunks[df_chunks["chunk_token_count"] > min_token_length].to_dict(orient="records")
print(f"Kept {len(pages_and_chunks_over_min_token_len)} chunks ≥ {min_token_length} tokens")

# -----------------
# Embeddings
# -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading embedding model on {device}")
embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)

texts = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]
embs = embedding_model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

for item, vec in zip(pages_and_chunks_over_min_token_len, embs):
    item["embedding"] = vec.tolist()

# -----------------
# Save
# -----------------
text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
os.makedirs(os.path.dirname(embeddings_df_save_path), exist_ok=True)
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)
print(f"Saved embeddings to {embeddings_df_save_path}")
