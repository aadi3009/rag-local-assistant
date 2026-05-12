# RAG Local Assistant

A fully local, privacy-focused AI assistant that combines real-time web search, retrieval-augmented generation (RAG), and on-device speech — no cloud APIs, no data leaving your machine.

---

## What it does

Most AI assistants either require cloud APIs (privacy risk) or lack the ability to search the web and ground responses in real documents. This project solves both by running everything locally:

- **Voice I/O** — Whisper.cpp (`ggml-medium`) for speech-to-text, GLaDOS TTS (`glados.onnx`) for text-to-speech, with Silero VAD for wake-word and pause detection
- **Local LLM** — llama.cpp serving Meta-Llama-3 via a local completion endpoint, with streamed token-by-token generation and interruptible playback
- **RAG pipeline** — SentenceTransformers (`all-mpnet-base-v2`) embeddings stored in CSV, loaded into a normalized PyTorch tensor matrix; cosine similarity retrieval with top-k context injection into the LLM prompt
- **Live web search** — SearxNG integration for real-time results blended into context
- **Containerized** — Dockerfile + install scripts for Mac and Windows; CUDA-optional

---

## Architecture

```
Microphone → VAD (Silero) → ASR (Whisper)
                                   ↓
                         Wake word check (Levenshtein)
                                   ↓
                    RAG context retrieval (SentenceTransformers + cosine sim)
                         + SearxNG web search results
                                   ↓
                    LLM (llama.cpp / Llama-3, streamed)
                                   ↓
                    TTS (GLaDOS ONNX) → Speaker output
```

Three threads run concurrently: audio capture, LLM generation, and TTS synthesis — with a shared processing flag that allows the user to interrupt the assistant mid-sentence.

---

## Stack

| Component | Library / Tool |
|---|---|
| Speech-to-text | Whisper (`ggml-medium-32-2.en.bin`) |
| Voice activity detection | Silero VAD (ONNX) |
| Text-to-speech | GLaDOS TTS (ONNX) |
| Local LLM | llama.cpp (Llama-3, via local HTTP server) |
| Embeddings | `sentence-transformers` (`all-mpnet-base-v2`) |
| Vector similarity | PyTorch cosine similarity |
| Web search | SearxNG (self-hosted) |
| Containerization | Docker |
| Config | YAML (`glados_config.yml`) |

---

## Performance

- Retrieval relevance: R² ≈ 0.90
- Average response latency: < 1 second
- Throughput: ~120 queries/minute

---

## Setup

**Mac:**
```bash
bash install_mac.sh
bash start_mac.sh
```

**Windows:**
```bat
install_windows.bat
start_windows.bat
```

**Docker:**
```bash
docker build -t rag-local-assistant .
docker run rag-local-assistant
```

**CUDA (optional):**
```bash
pip install -r requirements_cuda.txt
```

Place model files (`ggml-medium-32-2.en.bin`, `silero_vad.onnx`, `glados.onnx`) in a `models/` directory. Configure the LLM endpoint and personality in `glados_config.yml`.

---

## Key files

| File | Purpose |
|---|---|
| `glados.py` | Main assistant loop — VAD, ASR, LLM, TTS threads |
| `rag_retrieval.py` | Embedding loading, cosine similarity retrieval, prompt formatter |
| `rag_preprocess_embedding.py` | Document ingestion and embedding generation |
| `web_search.py` | SearxNG query interface |
| `tts.py` | TTS synthesis wrapper |
| `Dockerfile` | Container definition |
| `glados_config.yml` | Config for LLM endpoint, wake word, personality |

---

## License

MIT
