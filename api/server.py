import os, io, re, uuid, requests, time, random
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pypdf import PdfReader

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

try:
    import trafilatura  
except Exception:
    trafilatura = None
from bs4 import BeautifulSoup

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "kb_chunks")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
OPENWEBUI_MODEL_ID = os.getenv("OPENWEBUI_MODEL_ID", "llama3_8b") 

DOCS_DIR = "/app/data/docs"

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def gauge(pct: float, width: int = 12) -> str:
    filled = int(round(width * clamp01(pct)))
    return "▓" * filled + "░" * (width - filled)

def tok_len(s: str) -> int:
    return len((s or "").split())

app = FastAPI(title="RAG API (Qdrant + Ollama)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
qdrant = QdrantClient(url=QDRANT_URL)
_EMBED_DIM: Optional[int] = None

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, start, step = [], 0, max(1, size - overlap)
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start += step
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    out = []
    for t in texts:
        r = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": EMBED_MODEL, "prompt": t}, timeout=120)
        r.raise_for_status()
        out.append(r.json()["embedding"])
    return out

def ensure_collection():
    global _EMBED_DIM
    if _EMBED_DIM is None:
        _EMBED_DIM = len(embed_texts(["probe"])[0])
    names = [c.name for c in qdrant.get_collections().collections]
    if QDRANT_COLLECTION not in names:
        qdrant.create_collection(
            QDRANT_COLLECTION,
            vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
        )

def upsert_chunks(chunks: List[str], source_name: str, namespace: str) -> int:
    if not chunks:
        return 0
    vectors = embed_texts(chunks)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload={"source": source_name, "chunk_index": i, "namespace": namespace, "text": c},
        )
        for i, c in enumerate(chunks)
    ]
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)

# ------------ Scrape utils ------------
def _slug_from_url(url: str) -> str:
    p = urlparse(url)
    slug = (p.netloc + p.path).strip("/").replace("/", "_")
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", slug) or "page"

def extract_text_from_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ")
    return re.sub(r"\s+", " ", text).strip()

def fetch_and_extract(url: str, timeout: int = 20) -> str:
    time.sleep(random.uniform(0.4, 0.9))
    if trafilatura is not None:
        try:
            downloaded = trafilatura.fetch_url(url, timeout=timeout)
            if downloaded:
                extracted = trafilatura.extract(downloaded, include_links=False, include_comments=False)
                if extracted and extracted.strip():
                    return extracted.strip()
        except Exception:
            pass
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (RAG/1.0)"})
    r.raise_for_status()
    return extract_text_from_html(r.text)

# ------------ Ingestion fichiers ------------
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...), namespace: Optional[str] = Form("default")):
    ensure_collection()
    ns = namespace or "default"
    total = 0
    for f in files:
        raw = await f.read()
        if f.filename.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(raw))
            text = "\n\n".join([(p.extract_text() or "") for p in reader.pages])
        else:
            text = raw.decode("utf-8", errors="ignore")
        total += upsert_chunks(chunk_text(text), source_name=f.filename or "upload", namespace=ns)
    return {"ingested": total, "namespace": ns}


# ------------ Search (utilisé par le chat) ------------
def search_similar(query: str, top_k: int = 5, namespace: Optional[str] = None):
    ensure_collection()
    qvec = embed_texts([query])[0]
    qfilter = Filter(must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))]) if namespace else None
    return qdrant.search(collection_name=QDRANT_COLLECTION, query_vector=qvec, limit=top_k, query_filter=qfilter)

# ------------ OpenAI: models ------------
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": OPENWEBUI_MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "owner",
                "root": OPENWEBUI_MODEL_ID,
                "parent": None,
                "name": os.getenv("CHAT_MODEL", CHAT_MODEL),
            }
        ],
    }

# ------------ OpenAI: chat/completions ------------
class Message(BaseModel):
    role: str
    content: str

class ChatBody(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None

@app.post("/v1/chat/completions")
def chat_completions(body: ChatBody):
    requested = (body.model or OPENWEBUI_MODEL_ID)
    ollama_model = CHAT_MODEL if requested == OPENWEBUI_MODEL_ID else requested

    user_msg = next((m.content for m in reversed(body.messages) if m.role == "user"), "") or ""
    namespace = body.metadata.get("namespace") if isinstance(body.metadata, dict) else None
    m = re.search(r"\[ns:(.*?)\]", user_msg)
    if m:
        namespace = m.group(1)
        user_msg = re.sub(r"\[ns:.*?\]", "", user_msg).strip()

    hits = search_similar(user_msg, top_k=5, namespace=namespace) or []
    context = "\n\n".join([h.payload.get("text", "") for h in hits])

    avg_score = (sum(float(h.score or 0.0) for h in hits) / len(hits)) if hits else 0.0
    confidence = clamp01(avg_score) 
    header = f"Confiance RAG: {int(confidence*100)}%  [{gauge(confidence)}] — {len(hits)} extrait(s)"

    prompt_tokens = tok_len(user_msg)
    context_tokens = tok_len(context)
    coverage = context_tokens / max(1, (prompt_tokens + context_tokens))
    header = header + f" · Couverture contexte: {int(coverage*100)}%"

    system = body.system_prompt or (
        "Tu es un assistant RAG. Réponds uniquement à partir du CONTEXTE fourni. "
        "Si l'information n'est pas présente, dis-le."
    )
    prompt = f"SYSTEM:\n{system}\n\nCONTEXTE:\n{context}\n\nDERNIER MESSAGE UTILISATEUR:\n{user_msg}\nRéponse:"

    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": body.temperature, "top_p": body.top_p},
        },
        timeout=120,
    )
    r.raise_for_status()
    completion_text = r.json().get("response", "")

    if "Références:" not in completion_text and hits:
        refs = []
        for h in hits:
            src, idx = h.payload.get("source"), h.payload.get("chunk_index")
            snippet = (h.payload.get("text", "")[:160]).replace("\n", " ")
            sc = int(clamp01(h.score) * 100)
            refs.append(f"- ({sc}%) source: {src}, chunk: {idx} — \"{snippet}…\"")
        completion_text = completion_text.strip() + "\n\nRéférences:\n" + "\n".join(refs)

    completion_text = header + "\n\n" + completion_text

    return JSONResponse(
        content={
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(uuid.uuid1().time // 1e7),
            "model": body.model or CHAT_MODEL,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": completion_text}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        }
    )

@app.get("/health")
def health():
    return {"status": "ok"}
