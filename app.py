import logging
import re
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings

import chromadb
from rank_bm25 import BM25Okapi
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# ==============================
# CONFIG
# ==============================

DATA_DIR = Path(r"D:\Download1\kerala_ayurveda_content_pack_v1")
CSV_FILENAME = "products_catalog.csv"

GROQ_API_KEY = "enter your groq api key"
GROQ_MODEL = "llama-3.3-70b-versatile"

MAX_TOKENS_CHUNK = 400
CHUNK_OVERLAP = 50
MAX_HISTORY_TURNS = 4


# ==============================
# LIGHTWEIGHT EMBEDDING (TF-IDF Based)
# ==============================


class TfidfEmbedding:
    """Lightweight TF-IDF based embeddings - no PyTorch needed!"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=384, stop_words="english")
        self.is_fitted = False
        print("📊 Using TF-IDF embeddings (lightweight, no GPU needed)")

    def fit(self, texts: List[str]):
        """Fit the vectorizer on the corpus."""
        self.vectorizer.fit(texts)
        self.is_fitted = True

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings."""
        if not self.is_fitted:
            raise RuntimeError("Must fit vectorizer first!")

        vectors = self.vectorizer.transform(texts).toarray()
        return vectors.tolist()


# ==============================
# GLOBALS
# ==============================

embed_model: Optional[TfidfEmbedding] = None
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="kerala_ayurveda_corpus",
    metadata={"hnsw:space": "cosine"},
)

corpus_texts: List[str] = []
corpus_metadatas: List[Dict[str, Any]] = []
bm25: Optional[BM25Okapi] = None

groq_client = Groq(api_key=GROQ_API_KEY)
conversation_history: List[Dict[str, str]] = []


# ==============================
# UTILS
# ==============================


def normalize_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|\S", text.lower())


def guess_corpus_type(name: str) -> str:
    name = name.lower()
    if name.startswith("product_"):
        return "product"
    if name.startswith("faq_"):
        return "faq"
    if "tone" in name or "style" in name:
        return "style_guide"
    return "article"


def make_doc_id(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("product_"):
        stem = stem[len("product_") :]
    return stem.replace("_internal", "") + "_dossier"


# ==============================
# CHUNKING
# ==============================


def chunk_markdown(
    text: str,
    doc_id: str,
    corpus_type: str,
    max_tokens: int = MAX_TOKENS_CHUNK,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    sections = []
    current_title = "intro"
    current_buf: List[str] = []

    for line in lines:
        if re.match(r"^#{1,6}\s+", line):
            if current_buf:
                sections.append((current_title, "\n".join(current_buf)))
                current_buf = []
            current_title = re.sub(r"^#{1,6}\s+", "", line).strip()
        else:
            current_buf.append(line)

    if current_buf:
        sections.append((current_title, "\n".join(current_buf)))

    chunks: List[Dict[str, Any]] = []

    for idx, (title, body) in enumerate(sections, 1):
        tokens = simple_tokenize(body)

        if len(tokens) <= max_tokens:
            if body.strip():
                chunks.append(
                    {
                        "section_id": f"{idx:02d}_{title[:40].replace(' ', '_').lower()}",
                        "title": title,
                        "content": body,
                        "corpus_type": corpus_type,
                        "doc_id": doc_id,
                    }
                )
        else:
            start = 0
            while start < len(tokens):
                end = min(start + max_tokens, len(tokens))
                sub_tokens = tokens[start:end]
                sub_text = " ".join(sub_tokens)

                if sub_text.strip():
                    chunks.append(
                        {
                            "section_id": f"{idx:02d}_{title[:40].replace(' ', '_').lower()}_{start}",
                            "title": title,
                            "content": sub_text,
                            "corpus_type": corpus_type,
                            "doc_id": doc_id,
                        }
                    )

                if end == len(tokens):
                    break
                start = end - overlap

    return chunks


def csv_row_to_chunk(row: Dict[str, str], idx: int) -> Dict[str, Any]:
    product_id = row.get("product_id") or str(idx)
    name = row.get("name") or f"product_{product_id}"

    field_lines = [
        f"{k.replace('_', ' ').title()}: {v.strip()}"
        for k, v in row.items()
        if v and v.strip()
    ]
    content = "\n".join(field_lines)

    return {
        "section_id": f"csv_{idx}",
        "title": name,
        "content": content,
        "corpus_type": "product",
        "doc_id": f"product_catalog_{product_id}",
        "product_id": product_id,
        "name": name,
    }


# ==============================
# INGEST CORPUS
# ==============================


def ingest_corpus():
    global corpus_texts, corpus_metadatas, bm25, embed_model

    # Initialize embedding model
    if embed_model is None:
        embed_model = TfidfEmbedding()

    corpus_texts = []
    corpus_metadatas = []
    row_id_counter = 0

    print(f"📁 Scanning directory: {DATA_DIR}")

    # ---- Markdown files ----
    md_files = list(DATA_DIR.glob("*.md"))
    if not md_files:
        print(f"⚠️  No .md files found in {DATA_DIR}")
    else:
        print(f"✓ Found {len(md_files)} markdown files")

    for md_path in md_files:
        try:
            raw = md_path.read_text(encoding="utf-8")
            normalized = normalize_text(raw)
            corpus_type = guess_corpus_type(md_path.name)
            doc_id = make_doc_id(md_path.name)

            md_chunks = chunk_markdown(normalized, doc_id, corpus_type)

            for ch in md_chunks:
                corpus_texts.append(ch["content"])
                ch["row_id"] = row_id_counter
                corpus_metadatas.append(ch)
                row_id_counter += 1

            print(f"  ✓ {md_path.name}: {len(md_chunks)} chunks")
        except Exception as e:
            print(f"  ✗ Error with {md_path.name}: {e}")

    # ---- CSV file ----
    csv_path = DATA_DIR / CSV_FILENAME
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                csv_rows = list(reader)

                for idx, row in enumerate(csv_rows):
                    chunk = csv_row_to_chunk(row, idx)
                    chunk["row_id"] = row_id_counter
                    corpus_texts.append(chunk["content"])
                    corpus_metadatas.append(chunk)
                    row_id_counter += 1

            print(f"  ✓ {CSV_FILENAME}: {len(csv_rows)} products")
        except Exception as e:
            print(f"  ✗ Error with CSV: {e}")
    else:
        print(f"⚠️  {CSV_FILENAME} not found")

    if not corpus_texts:
        raise RuntimeError(f"❌ No data found! Check: {DATA_DIR}")

    print(f"\n📚 Total chunks: {len(corpus_texts)}")

    # ---- BM25 index ----
    print("🔨 Building BM25 index...")
    tokenized_corpus = []
    for t in corpus_texts:
        toks = simple_tokenize(t)
        tokenized_corpus.append(toks if toks else ["_empty_"])
    bm25 = BM25Okapi(tokenized_corpus)
    print("  ✓ BM25 ready")

    # ---- Fit TF-IDF and create embeddings ----
    print("🔨 Building TF-IDF embeddings...")
    embed_model.fit(corpus_texts)
    embeddings = embed_model.encode(corpus_texts)
    print(f"  ✓ Created {len(embeddings)} embeddings")

    # ---- Clear existing ChromaDB collection ----
    try:
        existing = collection.get()
        if existing.get("ids"):
            collection.delete(ids=existing["ids"])
            print(f"  ✓ Cleared {len(existing['ids'])} old entries")
    except:
        pass

    # ---- Add to ChromaDB ----
    print("📥 Indexing into ChromaDB...")
    try:
        collection.add(
            documents=corpus_texts,
            metadatas=corpus_metadatas,
            embeddings=embeddings,
            ids=[str(m["row_id"]) for m in corpus_metadatas],
        )
        print("  ✓ Indexed successfully")
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        raise

    print("\n✅ System ready!\n")


# ==============================
# RETRIEVAL (HYBRID)
# ==============================


def hybrid_retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if bm25 is None:
        raise RuntimeError("BM25 not initialized!")
    if embed_model is None:
        raise RuntimeError("Embedding model not initialized!")

    # BM25 scoring
    tokens = simple_tokenize(query)
    scores = bm25.get_scores(tokens)

    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    candidate_indices = [idx for idx, score in indexed_scores if score > 0][:50]

    try:
        query_embedding = embed_model.encode([query])

        if not candidate_indices:
            # Pure semantic search
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, len(corpus_texts)),
            )
        else:
            # Hybrid search
            candidate_ids = [corpus_metadatas[i]["row_id"] for i in candidate_indices]

            results = collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, len(candidate_ids)),
                where={"row_id": {"$in": candidate_ids}},
            )
    except Exception as e:
        print(f"⚠️  Retrieval warning: {e}")
        # Fallback to pure semantic
        try:
            query_embedding = embed_model.encode([query])
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, len(corpus_texts)),
            )
        except Exception as e2:
            print(f"❌ Search failed: {e2}")
            return []

    chunks: List[Dict[str, Any]] = []
    if results.get("documents") and results["documents"]:
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append(
                {
                    "content": text,
                    "doc_id": meta.get("doc_id", "unknown"),
                    "section_id": meta.get("section_id", "unknown"),
                    "corpus_type": meta.get("corpus_type", ""),
                    "title": meta.get("title", ""),
                }
            )

    return chunks


# ==============================
# PROMPTING + LLM
# ==============================


def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    numbered_blocks = []
    for i, c in enumerate(chunks, 1):
        label = f"[{i}] {c['doc_id']} / {c['section_id']}"
        if c.get("title"):
            label += f" ({c['title']})"
        numbered_blocks.append(f"{label}\n{c['content']}\n")

    context = "\n\n".join(numbered_blocks)

    history_text = ""
    for turn in conversation_history[-MAX_HISTORY_TURNS:]:
        role = turn["role"].capitalize()
        history_text += f"{role}: {turn['content']}\n"

    system = """You are an internal Kerala Ayurveda assistant.

Rules:
- Use ONLY the provided context for factual statements.
- If something is not supported by the context, reply: "I don't know based on the internal corpus."
- Warm, grounded, reassuring tone.
- Do NOT make medical or cure claims.
- Encourage consulting a qualified physician for medical questions.
- When using a chunk, cite it with [1], [2], etc., matching the numbered context."""

    prompt = f"""SYSTEM:
{system}

CONVERSATION HISTORY:
{history_text}

CONTEXT:
{context}

USER QUESTION:
{query}

ASSISTANT:
Please answer using the context and include citations like [1], [2] where appropriate."""

    return prompt


def call_llama(prompt: str) -> str:
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️  Groq API error: {str(e)}"


def extract_citations(
    answer: str, chunks: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    nums = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    out: List[Dict[str, str]] = []
    for n in sorted(nums):
        if 1 <= n <= len(chunks):
            c = chunks[n - 1]
            out.append({"doc_id": c["doc_id"], "section_id": c["section_id"]})
    return out


# ==============================
# MAIN ANSWER FUNCTION
# ==============================


def answer_user_query(query: str) -> Dict[str, Any]:
    conversation_history.append({"role": "user", "content": query})

    chunks = hybrid_retrieve(query, top_k=5)

    if not chunks:
        answer = "I don't know based on the internal corpus."
        conversation_history.append({"role": "assistant", "content": answer})
        return {"answer": answer, "citations": []}

    prompt = build_prompt(query, chunks)
    answer = call_llama(prompt)
    citations = extract_citations(answer, chunks)

    conversation_history.append({"role": "assistant", "content": answer})

    return {"answer": answer, "citations": citations}


# ==============================
# CLI CHAT LOOP
# ==============================

if __name__ == "__main__":
    print("=" * 70)
    print("    KERALA AYURVEDA INTERNAL Q&A SYSTEM")
    print("=" * 70)
    print()

    try:
        ingest_corpus()
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    print("=" * 70)
    print("💬 CHAT READY - Ask me anything!")
    print("=" * 70)
    print("Commands: 'exit' to quit | 'clear' to reset history")
    print()

    while True:
        try:
            user_q = input("You: ").strip()

            if not user_q:
                continue

            if user_q.lower() in {"exit", "quit"}:
                print("\n👋 Goodbye!")
                break

            if user_q.lower() == "clear":
                conversation_history.clear()
                print("🔄 History cleared\n")
                continue

            result = answer_user_query(user_q)

            print(f"\n🤖 {result['answer']}")

            if result["citations"]:
                print(f"\n📚 Sources: {result['citations']}")

            print("\n" + "-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {e}\n")
            print("-" * 70 + "\n")
