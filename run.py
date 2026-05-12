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
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)

# ==============================
# CONFIG
# ==============================

DATA_DIR = Path(r"D:\Download1\kerala_ayurveda_content_pack_v1")
CSV_FILENAME = "products_catalog.csv"

GROQ_API_KEY = "enter your groq api key here"
GROQ_MODEL = "llama-3.3-70b-versatile"

MAX_TOKENS_CHUNK = 450
CHUNK_OVERLAP = 50
MAX_HISTORY_TURNS = 4
SIMILARITY_THRESHOLD = 0.3


# ==============================
# LIGHTWEIGHT EMBEDDING (TF-IDF Based)
# ==============================


class TfidfEmbedding:
    """Lightweight TF-IDF based embeddings - no PyTorch needed!

    NOTE: For production, consider upgrading to BGE-m3 or similar neural embeddings
    once PyTorch/CUDA environment is properly configured.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=384, stop_words="english")
        self.is_fitted = False
        print("📊 Using TF-IDF embeddings (lightweight, no GPU needed)")
        print("   ℹ️  For production: Consider BGE-m3 once PyTorch is available")

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


def hybrid_retrieve(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Hybrid retrieval with confidence scoring."""
    if bm25 is None:
        raise RuntimeError("BM25 not initialized!")
    if embed_model is None:
        raise RuntimeError("Embedding model not initialized!")

    # BM25 scoring - narrow to 50 candidates
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
            # Hybrid search - semantic re-ranking of BM25 candidates
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
            return {"chunks": [], "confidence": "none", "distances": []}

    chunks: List[Dict[str, Any]] = []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []

    if results.get("documents") and results["documents"]:
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            distances if distances else [0] * len(results["documents"][0]),
        ):
            chunks.append(
                {
                    "content": text,
                    "doc_id": meta.get("doc_id", "unknown"),
                    "section_id": meta.get("section_id", "unknown"),
                    "corpus_type": meta.get("corpus_type", ""),
                    "title": meta.get("title", ""),
                    "distance": dist,
                }
            )

    # Determine confidence based on distances
    confidence = "high"
    if distances:
        avg_distance = sum(distances[:3]) / min(3, len(distances))
        if avg_distance > 1.0:
            confidence = "low"
        elif avg_distance > 0.7:
            confidence = "medium"

    return {
        "chunks": chunks,
        "confidence": confidence,
        "distances": distances[:top_k] if distances else [],
    }


# ==============================
# PROMPTING + LLM
# ==============================


def build_prompt(
    query: str, chunks: List[Dict[str, Any]], confidence: str = "high"
) -> str:
    """Build the prompt with numbered citations and confidence hints."""
    numbered_blocks = []
    for i, c in enumerate(chunks, 1):
        label = f"[{i}] {c['doc_id']} – {c['section_id']}"
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
- Warm, grounded, reassuring tone following Kerala Ayurveda's style.
- Do NOT make medical or cure claims.
- Encourage consulting a qualified physician for medical questions.
- When using a chunk, cite it with [1], [2], etc., matching the numbered context.
- If the context seems unrelated or mixed, ask a clarifying question instead of guessing."""

    confidence_hint = ""
    if confidence == "low":
        confidence_hint = "\n\nNOTE: The retrieved context may not be closely related to the question. If the context doesn't support a clear answer, ask for clarification."

    prompt = f"""SYSTEM:
{system}

CONVERSATION HISTORY:
{history_text}

CONTEXT:
{context}{confidence_hint}

USER QUESTION:
{query}

A:
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
    """Main query answering with structured output."""
    try:
        conversation_history.append({"role": "user", "content": query})

        retrieval_result = hybrid_retrieve(query, top_k=5)
        chunks = retrieval_result["chunks"]
        confidence = retrieval_result["confidence"]

        if not chunks:
            answer = "I don't know based on the internal corpus."
            conversation_history.append({"role": "assistant", "content": answer})
            return {
                "answer": answer,
                "citations": [],
                "confidence": "none",
                "retrieved_chunks": 0,
            }

        prompt = build_prompt(query, chunks, confidence)
        answer = call_llama(prompt)
        citations = extract_citations(answer, chunks)

        conversation_history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "retrieved_chunks": len(chunks),
            "distances": retrieval_result.get("distances", []),
        }
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(f"⚠️  {error_msg}")
        import traceback

        traceback.print_exc()
        return {
            "answer": "Sorry, I encountered an error processing your question. Please try again.",
            "citations": [],
            "confidence": "error",
            "retrieved_chunks": 0,
        }


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
                print("\n📚 Citations:")
                for i, cite in enumerate(result["citations"], 1):
                    print(f"   {i}. {cite['doc_id']} → {cite['section_id']}")

            if result.get("confidence") in ["low", "medium"]:
                print(
                    f"\n⚠️  Confidence: {result['confidence']} (retrieved {result.get('retrieved_chunks', 0)} chunks)"
                )

            print("\n" + "-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {e}\n")
            print("-" * 70 + "\n")
# # # D:\Download1\kerala_ayurveda_content_pack_v1\agents.py
# # """
# # Multi-Agent System for Content Generation
# # Enhanced with revision capabilities for orchestrator retry loop
# # """

# # import json
# # import re
# # from typing import List, Dict
# # from datetime import datetime
# # import config


# # class OutlineAgent:
# #     """Agent 1: Converts brief → structured outline with RAG validation"""

# #     def __init__(self, rag_pipeline):
# #         self.rag = rag_pipeline

# #     def generate_outline(self, brief_data: Dict) -> Dict:
# #         """Generate article outline from brief"""
# #         prompt = f"""Create a structured article outline for the following brief:

# # BRIEF: {brief_data["brief"]}
# # PRODUCT: {brief_data["product"]}
# # AUDIENCE: {brief_data["audience"]}
# # TARGET LENGTH: {brief_data["word_count"]} words
# # TONE: {brief_data["tone"]}

# # First, search the knowledge base for relevant information about {brief_data["product"]}.
# # Then create an outline with 4-6 sections that:
# # 1. Follow {config.BRAND_NAME}'s warm, educational style
# # 2. Are grounded in available corpus information
# # 3. Include: introduction, ingredients/heritage, benefits, usage, safety
# # 4. Avoid unsupported medical claims

# # Return ONLY valid JSON in this format:
# # {{
# #     "sections": [
# #         {{
# #             "id": "sec_intro",
# #             "title": "Section Title",
# #             "goals": ["goal 1", "goal 2"]
# #         }}
# #     ]
# # }}"""

# #         # Retrieve relevant context
# #         retrieved = self.rag.hybrid_retrieve(
# #             f"{brief_data['product']} benefits ingredients usage safety", top_k=8
# #         )

# #         # Build prompt with context
# #         full_prompt = self.rag.build_prompt(
# #             query=prompt,
# #             retrieved_chunks=retrieved,
# #             system_prompt="You are an expert Ayurvedic content strategist.",
# #         )

# #         # Generate outline
# #         response = self.rag.call_llm(
# #             full_prompt, temperature=config.OUTLINE_TEMPERATURE
# #         )

# #         # Parse JSON
# #         outline_json = self._extract_json(response)

# #         # Add metadata and validate
# #         outline_id = f"outline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# #         validated_sections = []
# #         rag_sources = set()

# #         for section in outline_json.get("sections", []):
# #             # RAG validation for section
# #             section_query = f"{brief_data['product']} {section['title']}"
# #             section_chunks = self.rag.hybrid_retrieve(section_query, top_k=3)

# #             if (
# #                 section_chunks
# #                 and section_chunks[0]["similarity"] > config.SIMILARITY_THRESHOLD
# #             ):
# #                 section["rag_sources"] = [
# #                     {"doc_id": chunk["doc_id"], "page": chunk["page"]}
# #                     for chunk in section_chunks[:2]
# #                 ]
# #                 for chunk in section_chunks[:2]:
# #                     rag_sources.add(f"{chunk['doc_id']} (p.{chunk['page']})")
# #             else:
# #                 section["rag_sources"] = []
# #                 section["note"] = "Limited corpus support - keep general"

# #             validated_sections.append(section)

# #         return {
# #             "outline_id": outline_id,
# #             "product": brief_data["product"],
# #             "sections": validated_sections,
# #             "rag_sources_referenced": list(rag_sources),
# #         }

# #     def _extract_json(self, text: str) -> Dict:
# #         """Extract JSON from LLM response"""
# #         json_match = re.search(r"\{.*\}", text, re.DOTALL)
# #         if json_match:
# #             try:
# #                 return json.loads(json_match.group())
# #             except json.JSONDecodeError:
# #                 pass

# #         # Fallback outline
# #         return {
# #             "sections": [
# #                 {
# #                     "id": "sec_intro",
# #                     "title": "Introduction",
# #                     "goals": ["Introduce the product", "Set context"],
# #                 },
# #                 {
# #                     "id": "sec_benefits",
# #                     "title": "Benefits and Traditional Uses",
# #                     "goals": ["Describe key benefits", "Traditional positioning"],
# #                 },
# #                 {
# #                     "id": "sec_usage",
# #                     "title": "How to Use",
# #                     "goals": ["Application instructions", "Best practices"],
# #                 },
# #                 {
# #                     "id": "sec_safety",
# #                     "title": "Safety and Precautions",
# #                     "goals": ["Safety information", "Consult practitioner"],
# #                 },
# #             ]
# #         }


# # class WriterAgent:
# #     """Agent 2: Generates draft with RAG-backed citations"""

# #     def __init__(self, rag_pipeline):
# #         self.rag = rag_pipeline

# #     def generate_draft(self, outline: Dict) -> Dict:
# #         """Generate article draft from outline"""
# #         draft_id = f"draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# #         paragraphs = []

# #         for section in outline["sections"]:
# #             # Retrieve context for section
# #             section_query = (
# #                 f"{outline['product']} {section['title']} {' '.join(section['goals'])}"
# #             )
# #             retrieved = self.rag.hybrid_retrieve(section_query, top_k=5)

# #             # Generate section content
# #             prompt = f"""Write a {len(section["goals"]) * 50}-word paragraph for:

# # SECTION: {section["title"]}
# # GOALS: {", ".join(section["goals"])}
# # PRODUCT: {outline["product"]}

# # Requirements:
# # - Use ONLY information from the provided context
# # - Write in warm, educational tone
# # - Cite sources using [1], [2] notation with page numbers
# # - Use language like "traditionally used for" or "may support"
# # - Avoid disease treatment claims
# # - Keep it engaging and readable

# # Write the paragraph now:"""

# #             full_prompt = self.rag.build_prompt(
# #                 query=prompt,
# #                 retrieved_chunks=retrieved,
# #                 system_prompt=f"You are a {config.BRAND_NAME} content writer.",
# #             )

# #             paragraph_text = self.rag.call_llm(
# #                 full_prompt, temperature=config.WRITER_TEMPERATURE
# #             )

# #             # Extract citations
# #             citations = self.rag.extract_citations(paragraph_text, retrieved)

# #             # Determine paragraph type
# #             is_narrative = (
# #                 section["id"] == "sec_intro"
# #                 or "scene-setting" in " ".join(section["goals"]).lower()
# #             )

# #             paragraphs.append(
# #                 {
# #                     "section_id": section["id"],
# #                     "text": paragraph_text.strip(),
# #                     "citations": citations if not is_narrative else [],
# #                     "type": "narrative" if is_narrative else "factual",
# #                 }
# #             )

# #         return {
# #             "draft_id": draft_id,
# #             "outline_id": outline["outline_id"],
# #             "product": outline["product"],
# #             "paragraphs": paragraphs,
# #         }

# #     def revise_draft(
# #         self, draft: Dict, fact_check_issues: List[Dict], outline: Dict
# #     ) -> Dict:
# #         """
# #         Revise draft based on fact-checker feedback

# #         This is called by the orchestrator during retry loops
# #         """
# #         draft_id = f"draft_rev_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# #         revised_paragraphs = []

# #         # Group issues by paragraph
# #         issues_by_para = {}
# #         for issue in fact_check_issues:
# #             para_idx = issue["paragraph_index"]
# #             if para_idx not in issues_by_para:
# #                 issues_by_para[para_idx] = []
# #             issues_by_para[para_idx].append(issue)

# #         # Revise each paragraph
# #         for idx, para in enumerate(draft["paragraphs"]):
# #             if idx in issues_by_para:
# #                 # This paragraph has issues - revise it
# #                 issues = issues_by_para[idx]

# #                 # Get section info
# #                 section = next(
# #                     (s for s in outline["sections"] if s["id"] == para["section_id"]),
# #                     None,
# #                 )

# #                 if section:
# #                     # Retrieve fresh context
# #                     section_query = f"{outline['product']} {section['title']}"
# #                     retrieved = self.rag.hybrid_retrieve(section_query, top_k=5)

# #                     # Build revision prompt
# #                     issue_descriptions = "\n".join(
# #                         [f"- {issue['type']}: {issue['message']}" for issue in issues]
# #                     )

# #                     prompt = f"""Revise this paragraph to address the following issues:

# # ORIGINAL PARAGRAPH:
# # {para["text"]}

# # ISSUES TO FIX:
# # {issue_descriptions}

# # REVISION REQUIREMENTS:
# # - Fix all identified issues
# # - Use ONLY information from the provided context
# # - Maintain warm, educational tone
# # - Cite sources using [1], [2] notation
# # - Use gentle language ("may support", "traditionally used for")
# # - Avoid unsupported medical claims

# # Write the revised paragraph:"""

# #                     full_prompt = self.rag.build_prompt(
# #                         query=prompt,
# #                         retrieved_chunks=retrieved,
# #                         system_prompt=f"You are a {config.BRAND_NAME} content writer revising content.",
# #                     )

# #                     revised_text = self.rag.call_llm(
# #                         full_prompt, temperature=config.WRITER_TEMPERATURE
# #                     )
# #                     citations = self.rag.extract_citations(revised_text, retrieved)

# #                     revised_paragraphs.append(
# #                         {
# #                             "section_id": para["section_id"],
# #                             "text": revised_text.strip(),
# #                             "citations": citations
# #                             if para["type"] != "narrative"
# #                             else [],
# #                             "type": para["type"],
# #                         }
# #                     )
# #                 else:
# #                     # Keep original if section not found
# #                     revised_paragraphs.append(para)
# #             else:
# #                 # No issues - keep original
# #                 revised_paragraphs.append(para)

# #         return {
# #             "draft_id": draft_id,
# #             "outline_id": draft["outline_id"],
# #             "product": draft["product"],
# #             "paragraphs": revised_paragraphs,
# #             "revision_of": draft["draft_id"],
# #         }


# # class FactCheckerAgent:
# #     """Agent 3: Validates claims against corpus"""

# #     def __init__(self, rag_pipeline):
# #         self.rag = rag_pipeline
# #         self.medical_patterns = config.MEDICAL_CLAIM_PATTERNS

# #     def check_draft(self, draft: Dict) -> Dict:
# #         """Fact-check draft paragraphs"""
# #         checks = []
# #         issues = []

# #         for idx, para in enumerate(draft["paragraphs"]):
# #             # Skip narrative paragraphs
# #             if para.get("type") == "narrative":
# #                 checks.append(
# #                     {
# #                         "paragraph_index": idx,
# #                         "claims": [],
# #                         "note": "Narrative paragraph - no factual validation needed",
# #                     }
# #                 )
# #                 continue

# #             # Extract and validate claims
# #             claims = self._extract_claims(para["text"])
# #             validated_claims = []

# #             for claim_text in claims:
# #                 is_medical = self._is_medical_claim(claim_text)

# #                 # Retrieve supporting docs
# #                 retrieved = self.rag.hybrid_retrieve(claim_text, top_k=3)

# #                 # Determine support status
# #                 if retrieved and retrieved[0]["similarity"] > 0.4:
# #                     status = "supported"
# #                     supporting_docs = [
# #                         {"doc_id": chunk["doc_id"], "page": chunk["page"]}
# #                         for chunk in retrieved[:2]
# #                     ]
# #                 else:
# #                     status = "unsupported"
# #                     supporting_docs = []

# #                     # Flag as issue
# #                     severity = "high" if is_medical else "medium"
# #                     issues.append(
# #                         {
# #                             "severity": severity,
# #                             "type": "unsupported_medical_claim"
# #                             if is_medical
# #                             else "unsupported_claim",
# #                             "paragraph_index": idx,
# #                             "claim": claim_text,
# #                             "message": f"Claim not supported by corpus: '{claim_text[:100]}...'",
# #                         }
# #                     )

# #                 validated_claims.append(
# #                     {
# #                         "text": claim_text,
# #                         "status": status,
# #                         "is_medical": is_medical,
# #                         "supporting_docs": supporting_docs,
# #                     }
# #                 )

# #             checks.append({"paragraph_index": idx, "claims": validated_claims})

# #             # Check for missing citations
# #             if para.get("type") == "factual" and not para.get("citations"):
# #                 issues.append(
# #                     {
# #                         "severity": "medium",
# #                         "type": "missing_citations",
# #                         "paragraph_index": idx,
# #                         "message": "Factual paragraph lacks citations",
# #                     }
# #                 )

# #         return {
# #             "draft_id": draft["draft_id"],
# #             "checks": checks,
# #             "issues": issues,
# #             "status": "needs_revision"
# #             if any(i["severity"] == "high" for i in issues)
# #             else "approved",
# #         }

# #     def _extract_claims(self, text: str) -> List[str]:
# #         """Extract factual claims from text"""
# #         sentences = re.split(r"[.!?]+", text)
# #         claims = []

# #         for sent in sentences:
# #             sent = sent.strip()
# #             if not sent:
# #                 continue

# #             # Skip narrative sentences
# #             narrative_indicators = ["after", "when", "while", "imagine", "picture"]
# #             if any(ind in sent.lower() for ind in narrative_indicators):
# #                 continue

# #             # Check for factual language
# #             factual_indicators = [
# #                 "contains",
# #                 "includes",
# #                 "supports",
# #                 "helps",
# #                 "used for",
# #                 "traditionally",
# #                 "known for",
# #                 "rich in",
# #                 "blend of",
# #             ]

# #             if any(ind in sent.lower() for ind in factual_indicators):
# #                 claims.append(sent)

# #         return claims

# #     def _is_medical_claim(self, text: str) -> bool:
# #         """Check if text contains medical claim language"""
# #         text_lower = text.lower()
# #         return any(re.search(pattern, text_lower) for pattern in self.medical_patterns)


# # class FinalizationAgent:
# #     """Agent 4: Applies tone/style and prepares final output"""

# #     def __init__(self, rag_pipeline):
# #         self.rag = rag_pipeline

# #     def finalize(self, draft: Dict, fact_check: Dict) -> Dict:
# #         """Finalize draft with tone/style adjustments"""
# #         final_id = f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# #         # Compile full article
# #         article_parts = []
# #         all_citations = {}
# #         citation_counter = 1
# #         notes = []

# #         for idx, para in enumerate(draft["paragraphs"]):
# #             para_check = next(
# #                 (c for c in fact_check["checks"] if c["paragraph_index"] == idx), None
# #             )

# #             para_text = para["text"]

# #             # Renumber citations
# #             if para.get("citations"):
# #                 for old_cit in para["citations"]:
# #                     cit_key = f"{old_cit['doc_id']}_p{old_cit['page']}"
# #                     if cit_key not in all_citations:
# #                         all_citations[cit_key] = {
# #                             "label": citation_counter,
# #                             "doc_id": old_cit["doc_id"],
# #                             "page": old_cit["page"],
# #                         }
# #                         citation_counter += 1

# #             article_parts.append(para_text)

# #             # Add notes
# #             if para.get("type") == "narrative":
# #                 notes.append(f"Paragraph {idx + 1}: Scene-setting, no factual claims")
# #             elif para_check and para_check.get("claims"):
# #                 supported = sum(
# #                     1 for c in para_check["claims"] if c["status"] == "supported"
# #                 )
# #                 total = len(para_check["claims"])
# #                 notes.append(
# #                     f"Paragraph {idx + 1}: {supported}/{total} claims supported"
# #                 )

# #         # Compile full text
# #         full_text = "\n\n".join(article_parts)

# #         # Add disclaimer
# #         disclaimer = "\n\n*These statements have not been evaluated by medical authorities. This product is not intended to diagnose, treat, cure, or prevent any disease. Please consult with an Ayurvedic practitioner or healthcare provider before use.*"
# #         full_text += disclaimer

# #         # Determine status
# #         high_severity_issues = [
# #             i for i in fact_check["issues"] if i["severity"] == "high"
# #         ]

# #         if high_severity_issues:
# #             status = "needs_revision"
# #             notes.append(f"⚠️ {len(high_severity_issues)} high-severity issues found")
# #         else:
# #             status = "ready_for_editor"
# #             notes.append("✓ All checks passed")

# #         return {
# #             "final_draft_id": final_id,
# #             "draft_id": draft["draft_id"],
# #             "text": full_text,
# #             "citations": list(all_citations.values()),
# #             "status": status,
# #             "notes_for_editor": notes,
# #             "fact_check_summary": {
# #                 "total_checks": len(fact_check["checks"]),
# #                 "total_issues": len(fact_check["issues"]),
# #                 "high_severity": len(high_severity_issues),
# #             },
# #         }
# """
# Multi-Agent System for Content Generation
# Enhanced with balanced retrieval (product + PDF knowledge)
# """

# import json
# import re
# from typing import List, Dict
# from datetime import datetime
# import config


# class OutlineAgent:
#     """Agent 1: Converts brief → structured outline with RAG validation"""

#     def __init__(self, rag_pipeline):
#         self.rag = rag_pipeline

#     def generate_outline(self, brief_data: Dict) -> Dict:
#         """Generate article outline from brief"""
#         prompt = f"""Create a structured article outline for the following brief:

# BRIEF: {brief_data["brief"]}
# PRODUCT: {brief_data["product"]}
# AUDIENCE: {brief_data["audience"]}
# TARGET LENGTH: {brief_data["word_count"]} words
# TONE: {brief_data["tone"]}

# First, search the knowledge base for relevant information about {brief_data["product"]}.
# Then create an outline with 4-6 sections that:
# 1. Follow {config.BRAND_NAME}'s warm, educational style
# 2. Are grounded in available corpus information
# 3. Include: introduction, ingredients/heritage, benefits, usage, safety
# 4. Avoid unsupported medical claims

# Return ONLY valid JSON in this format:
# {{
#     "sections": [
#         {{
#             "id": "sec_intro",
#             "title": "Section Title",
#             "goals": ["goal 1", "goal 2"]
#         }}
#     ]
# }}"""

#         # Retrieve relevant context - prefer traditional knowledge for outline
#         retrieved = self.rag.hybrid_retrieve(
#             f"{brief_data['product']} ayurvedic benefits ingredients traditional usage",
#             top_k=10,
#             prefer_product_info="no",  # Don't boost product info for outline phase
#         )

#         # Build prompt with context
#         full_prompt = self.rag.build_prompt(
#             query=prompt,
#             retrieved_chunks=retrieved,
#             system_prompt="You are an expert Ayurvedic content strategist.",
#         )

#         # Generate outline
#         response = self.rag.call_llm(
#             full_prompt, temperature=config.OUTLINE_TEMPERATURE
#         )

#         # Parse JSON
#         outline_json = self._extract_json(response)

#         # Add metadata and validate
#         outline_id = f"outline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         validated_sections = []
#         rag_sources = set()

#         for section in outline_json.get("sections", []):
#             # RAG validation for section - use balanced retrieval
#             section_query = f"{brief_data['product']} {section['title']}"
#             section_chunks = self.rag.hybrid_retrieve(
#                 section_query,
#                 top_k=5,
#                 prefer_product_info="auto",  # Smart balancing
#             )

#             if (
#                 section_chunks
#                 and section_chunks[0]["similarity"] > config.SIMILARITY_THRESHOLD
#             ):
#                 section["rag_sources"] = [
#                     {"doc_id": chunk["doc_id"], "page": chunk["page"]}
#                     for chunk in section_chunks[:2]
#                 ]
#                 for chunk in section_chunks[:2]:
#                     rag_sources.add(f"{chunk['doc_id']} (p.{chunk['page']})")
#             else:
#                 section["rag_sources"] = []
#                 section["note"] = "Limited corpus support - keep general"

#             validated_sections.append(section)

#         return {
#             "outline_id": outline_id,
#             "product": brief_data["product"],
#             "sections": validated_sections,
#             "rag_sources_referenced": list(rag_sources),
#         }

#     def _extract_json(self, text: str) -> Dict:
#         """Extract JSON from LLM response"""
#         json_match = re.search(r"\{.*\}", text, re.DOTALL)
#         if json_match:
#             try:
#                 return json.loads(json_match.group())
#             except json.JSONDecodeError:
#                 pass

#         # Fallback outline
#         return {
#             "sections": [
#                 {
#                     "id": "sec_intro",
#                     "title": "Introduction",
#                     "goals": ["Introduce the product", "Set context"],
#                 },
#                 {
#                     "id": "sec_heritage",
#                     "title": "Traditional Heritage and Formulation",
#                     "goals": ["Traditional background", "Classical references"],
#                 },
#                 {
#                     "id": "sec_ingredients",
#                     "title": "Key Ingredients and Their Properties",
#                     "goals": ["Describe ingredients", "Traditional properties"],
#                 },
#                 {
#                     "id": "sec_benefits",
#                     "title": "Benefits and Traditional Uses",
#                     "goals": ["Describe key benefits", "Traditional positioning"],
#                 },
#                 {
#                     "id": "sec_usage",
#                     "title": "How to Use",
#                     "goals": ["Application instructions", "Best practices"],
#                 },
#                 {
#                     "id": "sec_safety",
#                     "title": "Safety and Precautions",
#                     "goals": ["Safety information", "Consult practitioner"],
#                 },
#             ]
#         }


# class WriterAgent:
#     """Agent 2: Generates draft with RAG-backed citations"""

#     def __init__(self, rag_pipeline):
#         self.rag = rag_pipeline

#     def generate_draft(self, outline: Dict) -> Dict:
#         """Generate article draft from outline"""
#         draft_id = f"draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         paragraphs = []

#         for section in outline["sections"]:
#             # Determine retrieval strategy based on section type
#             prefer_mode = self._determine_preference(section)

#             # Retrieve context for section with appropriate preference
#             section_query = (
#                 f"{outline['product']} {section['title']} {' '.join(section['goals'])}"
#             )
#             retrieved = self.rag.hybrid_retrieve(
#                 section_query,
#                 top_k=8,  # Increased from 5 to get more diverse sources
#                 prefer_product_info=prefer_mode,
#             )

#             # Generate section content
#             prompt = f"""Write a {len(section["goals"]) * 50}-word paragraph for:

# SECTION: {section["title"]}
# GOALS: {", ".join(section["goals"])}
# PRODUCT: {outline["product"]}

# Requirements:
# - Use ONLY information from the provided context
# - Balance traditional knowledge with practical product information
# - Write in warm, educational tone
# - Cite sources using [1], [2] notation with page numbers
# - Use language like "traditionally used for" or "may support"
# - Avoid disease treatment claims
# - Keep it engaging and readable
# - Prefer traditional/classical sources for ingredient properties
# - Use product info for practical usage details

# Write the paragraph now:"""

#             full_prompt = self.rag.build_prompt(
#                 query=prompt,
#                 retrieved_chunks=retrieved,
#                 system_prompt=f"You are a {config.BRAND_NAME} content writer who balances traditional Ayurvedic wisdom with practical product guidance.",
#             )

#             paragraph_text = self.rag.call_llm(
#                 full_prompt, temperature=config.WRITER_TEMPERATURE
#             )

#             # Extract citations
#             citations = self.rag.extract_citations(paragraph_text, retrieved)

#             # Determine paragraph type
#             is_narrative = (
#                 section["id"] == "sec_intro"
#                 or "scene-setting" in " ".join(section["goals"]).lower()
#             )

#             paragraphs.append(
#                 {
#                     "section_id": section["id"],
#                     "text": paragraph_text.strip(),
#                     "citations": citations if not is_narrative else [],
#                     "type": "narrative" if is_narrative else "factual",
#                 }
#             )

#         return {
#             "draft_id": draft_id,
#             "outline_id": outline["outline_id"],
#             "product": outline["product"],
#             "paragraphs": paragraphs,
#         }

#     def _determine_preference(self, section: Dict) -> str:
#         """
#         Determine whether to prefer product info based on section type

#         Args:
#             section: Section dictionary with id and title

#         Returns:
#             'yes', 'no', or 'auto'
#         """
#         section_id = section.get("id", "").lower()
#         section_title = section.get("title", "").lower()

#         # Sections that should prefer product info
#         product_sections = ["usage", "application", "how to use", "pricing"]

#         # Sections that should prefer traditional knowledge
#         traditional_sections = [
#             "heritage",
#             "history",
#             "traditional",
#             "ingredients",
#             "properties",
#             "ayurveda",
#         ]

#         # Check section ID and title
#         combined = f"{section_id} {section_title}"

#         if any(ps in combined for ps in product_sections):
#             return "yes"  # Prefer product info
#         elif any(ts in combined for ts in traditional_sections):
#             return "no"  # Prefer traditional/PDF knowledge
#         else:
#             return "auto"  # Smart balancing

#     def revise_draft(
#         self, draft: Dict, fact_check_issues: List[Dict], outline: Dict
#     ) -> Dict:
#         """
#         Revise draft based on fact-checker feedback

#         This is called by the orchestrator during retry loops
#         """
#         draft_id = f"draft_rev_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         revised_paragraphs = []

#         # Group issues by paragraph
#         issues_by_para = {}
#         for issue in fact_check_issues:
#             para_idx = issue["paragraph_index"]
#             if para_idx not in issues_by_para:
#                 issues_by_para[para_idx] = []
#             issues_by_para[para_idx].append(issue)

#         # Revise each paragraph
#         for idx, para in enumerate(draft["paragraphs"]):
#             if idx in issues_by_para:
#                 # This paragraph has issues - revise it
#                 issues = issues_by_para[idx]

#                 # Get section info
#                 section = next(
#                     (s for s in outline["sections"] if s["id"] == para["section_id"]),
#                     None,
#                 )

#                 if section:
#                     # Determine preference
#                     prefer_mode = self._determine_preference(section)

#                     # Retrieve fresh context
#                     section_query = f"{outline['product']} {section['title']}"
#                     retrieved = self.rag.hybrid_retrieve(
#                         section_query, top_k=8, prefer_product_info=prefer_mode
#                     )

#                     # Build revision prompt
#                     issue_descriptions = "\n".join(
#                         [f"- {issue['type']}: {issue['message']}" for issue in issues]
#                     )

#                     prompt = f"""Revise this paragraph to address the following issues:

# ORIGINAL PARAGRAPH:
# {para["text"]}

# ISSUES TO FIX:
# {issue_descriptions}

# REVISION REQUIREMENTS:
# - Fix all identified issues
# - Use ONLY information from the provided context
# - Balance traditional knowledge with product information
# - Maintain warm, educational tone
# - Cite sources using [1], [2] notation
# - Use gentle language ("may support", "traditionally used for")
# - Avoid unsupported medical claims

# Write the revised paragraph:"""

#                     full_prompt = self.rag.build_prompt(
#                         query=prompt,
#                         retrieved_chunks=retrieved,
#                         system_prompt=f"You are a {config.BRAND_NAME} content writer revising content.",
#                     )

#                     revised_text = self.rag.call_llm(
#                         full_prompt, temperature=config.WRITER_TEMPERATURE
#                     )
#                     citations = self.rag.extract_citations(revised_text, retrieved)

#                     revised_paragraphs.append(
#                         {
#                             "section_id": para["section_id"],
#                             "text": revised_text.strip(),
#                             "citations": citations
#                             if para["type"] != "narrative"
#                             else [],
#                             "type": para["type"],
#                         }
#                     )
#                 else:
#                     # Keep original if section not found
#                     revised_paragraphs.append(para)
#             else:
#                 # No issues - keep original
#                 revised_paragraphs.append(para)

#         return {
#             "draft_id": draft_id,
#             "outline_id": draft["outline_id"],
#             "product": draft["product"],
#             "paragraphs": revised_paragraphs,
#             "revision_of": draft["draft_id"],
#         }


# class FactCheckerAgent:
#     """Agent 3: Validates claims against corpus"""

#     def __init__(self, rag_pipeline):
#         self.rag = rag_pipeline
#         self.medical_patterns = config.MEDICAL_CLAIM_PATTERNS

#     def check_draft(self, draft: Dict) -> Dict:
#         """Fact-check draft paragraphs"""
#         checks = []
#         issues = []

#         for idx, para in enumerate(draft["paragraphs"]):
#             # Skip narrative paragraphs
#             if para.get("type") == "narrative":
#                 checks.append(
#                     {
#                         "paragraph_index": idx,
#                         "claims": [],
#                         "note": "Narrative paragraph - no factual validation needed",
#                     }
#                 )
#                 continue

#             # Extract and validate claims
#             claims = self._extract_claims(para["text"])
#             validated_claims = []

#             for claim_text in claims:
#                 is_medical = self._is_medical_claim(claim_text)

#                 # Retrieve supporting docs - use balanced retrieval
#                 retrieved = self.rag.hybrid_retrieve(
#                     claim_text,
#                     top_k=5,
#                     prefer_product_info="auto",  # Smart balancing for fact-checking
#                 )

#                 # Determine support status with more lenient threshold
#                 if retrieved and retrieved[0]["similarity"] > 0.35:  # Lowered from 0.4
#                     status = "supported"
#                     supporting_docs = [
#                         {"doc_id": chunk["doc_id"], "page": chunk["page"]}
#                         for chunk in retrieved[:2]
#                     ]
#                 else:
#                     status = "unsupported"
#                     supporting_docs = []

#                     # Flag as issue
#                     severity = "high" if is_medical else "medium"
#                     issues.append(
#                         {
#                             "severity": severity,
#                             "type": "unsupported_medical_claim"
#                             if is_medical
#                             else "unsupported_claim",
#                             "paragraph_index": idx,
#                             "claim": claim_text,
#                             "message": f"Claim not supported by corpus: '{claim_text[:100]}...'",
#                         }
#                     )

#                 validated_claims.append(
#                     {
#                         "text": claim_text,
#                         "status": status,
#                         "is_medical": is_medical,
#                         "supporting_docs": supporting_docs,
#                     }
#                 )

#             checks.append({"paragraph_index": idx, "claims": validated_claims})

#             # Check for missing citations
#             if para.get("type") == "factual" and not para.get("citations"):
#                 issues.append(
#                     {
#                         "severity": "medium",
#                         "type": "missing_citations",
#                         "paragraph_index": idx,
#                         "message": "Factual paragraph lacks citations",
#                     }
#                 )

#         return {
#             "draft_id": draft["draft_id"],
#             "checks": checks,
#             "issues": issues,
#             "status": "needs_revision"
#             if any(i["severity"] == "high" for i in issues)
#             else "approved",
#         }

#     def _extract_claims(self, text: str) -> List[str]:
#         """Extract factual claims from text"""
#         sentences = re.split(r"[.!?]+", text)
#         claims = []

#         for sent in sentences:
#             sent = sent.strip()
#             if not sent:
#                 continue

#             # Skip narrative sentences
#             narrative_indicators = ["after", "when", "while", "imagine", "picture"]
#             if any(ind in sent.lower() for ind in narrative_indicators):
#                 continue

#             # Check for factual language
#             factual_indicators = [
#                 "contains",
#                 "includes",
#                 "supports",
#                 "helps",
#                 "used for",
#                 "traditionally",
#                 "known for",
#                 "rich in",
#                 "blend of",
#                 "according to",
#                 "described in",
#             ]

#             if any(ind in sent.lower() for ind in factual_indicators):
#                 claims.append(sent)

#         return claims

#     def _is_medical_claim(self, text: str) -> bool:
#         """Check if text contains medical claim language"""
#         text_lower = text.lower()
#         return any(re.search(pattern, text_lower) for pattern in self.medical_patterns)


# class FinalizationAgent:
#     """Agent 4: Applies tone/style and prepares final output"""

#     def __init__(self, rag_pipeline):
#         self.rag = rag_pipeline

#     def finalize(self, draft: Dict, fact_check: Dict) -> Dict:
#         """Finalize draft with tone/style adjustments"""
#         final_id = f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

#         # Compile full article
#         article_parts = []
#         all_citations = {}
#         citation_counter = 1
#         notes = []

#         for idx, para in enumerate(draft["paragraphs"]):
#             para_check = next(
#                 (c for c in fact_check["checks"] if c["paragraph_index"] == idx), None
#             )

#             para_text = para["text"]

#             # Renumber citations
#             if para.get("citations"):
#                 for old_cit in para["citations"]:
#                     cit_key = f"{old_cit['doc_id']}_p{old_cit['page']}"
#                     if cit_key not in all_citations:
#                         all_citations[cit_key] = {
#                             "label": citation_counter,
#                             "doc_id": old_cit["doc_id"],
#                             "page": old_cit["page"],
#                         }
#                         citation_counter += 1

#             article_parts.append(para_text)

#             # Add notes
#             if para.get("type") == "narrative":
#                 notes.append(f"Paragraph {idx + 1}: Scene-setting, no factual claims")
#             elif para_check and para_check.get("claims"):
#                 supported = sum(
#                     1 for c in para_check["claims"] if c["status"] == "supported"
#                 )
#                 total = len(para_check["claims"])
#                 notes.append(
#                     f"Paragraph {idx + 1}: {supported}/{total} claims supported"
#                 )

#         # Compile full text
#         full_text = "\n\n".join(article_parts)

#         # Add disclaimer
#         disclaimer = "\n\n*These statements have not been evaluated by medical authorities. This product is not intended to diagnose, treat, cure, or prevent any disease. Please consult with an Ayurvedic practitioner or healthcare provider before use.*"
#         full_text += disclaimer

#         # Determine status
#         high_severity_issues = [
#             i for i in fact_check["issues"] if i["severity"] == "high"
#         ]

#         if high_severity_issues:
#             status = "needs_revision"
#             notes.append(f"⚠️ {len(high_severity_issues)} high-severity issues found")
#         else:
#             status = "ready_for_editor"
#             notes.append("✓ All checks passed")

#         return {
#             "final_draft_id": final_id,
#             "draft_id": draft["draft_id"],
#             "text": full_text,
#             "citations": list(all_citations.values()),
#             "status": status,
#             "notes_for_editor": notes,
#             "fact_check_summary": {
#                 "total_checks": len(fact_check["checks"]),
#                 "total_issues": len(fact_check["issues"]),
#                 "high_severity": len(high_severity_issues),
#             },
#         }
# D:\Download1\kerala_ayurveda_content_pack_v1\agents.py