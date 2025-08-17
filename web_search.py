import requests
import textwrap
from bs4 import BeautifulSoup
from newspaper import Article
import re


def print_wrapped(text, width=100):
    print(textwrap.fill(text, width=width))


def search_searxng(query, endpoint="http://localhost:8080/search"):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GladosBot/1.0)",
        "Accept": "application/json",
        "Referer": "http://localhost:8080/",
        # "X-Requested-With": "XMLHttpRequest",  # uncomment if your setup requires it
    }
    params = {"q": query, "format": "json"}
    r = requests.get(endpoint, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json().get("results", [])


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fetch_article_text(url: str, char_limit: int = 3000) -> str:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return clean_text(article.text[:char_limit])
    except Exception:
        try:
            # Fallback to simple scraping
            html = requests.get(url, timeout=5).text
            soup = BeautifulSoup(html, "html.parser")
            paragraphs = soup.find_all("p")
            content = " ".join(p.get_text() for p in paragraphs)
            return clean_text(content[:char_limit])
        except Exception as e:
            print(f"[WebSearch] Failed to fetch {url}: {e}")
            return ""


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks


def prompt_formatter(query: str, context_chunks: list[str]) -> str:
    context = "\n- " + "\n- ".join(context_chunks)
    base_prompt = """You are a helpful assistant with access to the internet. Using the following web-based context chunks, answer the query.
Do not make up information. Be factual and concise.
Avoid repeating context. If multiple perspectives exist, mention them as alternatives.

Web context:
{context}

User query: {query}
Answer:"""
    return base_prompt.format(context=context, query=query)


def websearch_add_context(query: str, top_k: int = 5, chunk_size: int = 600, overlap: int = 100) -> str:
    results = search_searxng(query)
    if not results:
        print("[WebSearch] No results.")
        return query

    all_chunks = []
    print("\n[Top Web Sources]")
    for idx, result in enumerate(results[:top_k]):
        url = result.get("url", "")
        title = result.get("title", "Untitled")
        print_wrapped(f"{idx+1}. {title} | {url}")

        if not url:
            continue

        text = fetch_article_text(url)
        if text:
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(chunks[:2])  # Use only top 2 chunks per source

    if not all_chunks:
        return query

    return prompt_formatter(query, all_chunks[:top_k])

if __name__ == "__main__":
    query = "latest news about artificial intelligence"
    prompt = websearch_add_context(query)
    print("\n[Generated Prompt]\n")
    print(prompt)
