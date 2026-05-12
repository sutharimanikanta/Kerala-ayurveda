# # # D:\Download1\kerala_ayurveda_content_pack_v1\rag_pipeline.py
"""
Enhanced RAG Pipeline with Persistent Corpus & Intent-Aware Retrieval
"""

import re
import json
import hashlib
import logging
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq

from pdf_processor import PDFProcessor
from product_knowledge import get_all_product_chunks
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Enhanced RAG pipeline with persistent corpus and intent-aware retrieval"""

    def __init__(self, pdf_paths: Optional[List[str]] = None, force_reingest: bool = False):
        logger.info("Initializing RAG Pipeline with persistent storage...")

        # Validate configuration
        config.validate_config()

        # Initialize components
        self.pdf_processor = PDFProcessor(
            ocr_enabled=config.OCR_ENABLED,
            min_text_length=config.OCR_MIN_TEXT_LENGTH
        )

        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

        self.groq_client = Groq(api_key=config.GROQ_API_KEY)

        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIRECTORY
        )

        # Try to get existing collection or create new one
        try:
            self.collection = self.chroma_client.get_collection(
                name=config.CHROMA_COLLECTION_NAME
            )
            logger.info("✓ Loaded existing persistent collection")
        except:
            self.collection = self.chroma_client.create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={"description": "Nalpamaradi Keram knowledge base"}
            )
            logger.info("✓ Created new persistent collection")

        # Load corpus metadata
        self.corpus_metadata = self._load_corpus_metadata()

        # In-memory chunks for BM25
        self.chunks = []

        # Check if we need to process corpus
        needs_processing = force_reingest or not self._is_corpus_ready()

        if needs_processing:
            logger.info("Processing corpus...")

            # Always add product knowledge first
            self._add_product_knowledge()

            # Process PDFs if provided
            if pdf_paths:
                self._process_pdfs(pdf_paths)

            # Save metadata
            self._save_corpus_metadata()
        else:
            logger.info("Loading existing corpus from persistent storage...")
            self._load_existing_corpus()

        # Build BM25 index
        self._build_bm25_index()

        logger.info(f"✓ RAG Pipeline ready with {len(self.chunks)} chunks")
        self._log_corpus_stats()

    def _load_corpus_metadata(self) -> Dict:
        """Load metadata about processed corpus"""
        if config.CORPUS_METADATA_FILE.exists():
            with open(config.CORPUS_METADATA_FILE, 'r') as f:
                return json.load(f)
        return {
            'processed_files': {},
            'product_knowledge_version': None,
            'last_updated': None
        }

    def _save_corpus_metadata(self):
        """Save metadata about processed corpus"""
        import datetime
        self.corpus_metadata['last_updated'] = datetime.datetime.now().isoformat()
        with open(config.CORPUS_METADATA_FILE, 'w') as f:
            json.dump(self.corpus_metadata, f, indent=2)

    def _get_file_hash(self, filepath: str) -> str:
        """Generate hash for file to detect changes"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _is_corpus_ready(self) -> bool:
        """Check if we have a usable corpus"""
        try:
            count = self.collection.count()
            return count > 0
        except:
            return False

    def _load_existing_corpus(self):
        """Load existing chunks from persistent ChromaDB"""
        try:
            # Get all documents from collection
            results = self.collection.get(
                include=['documents', 'metadatas', 'embeddings']
            )

            if not results['ids']:
                logger.warning("No documents found in persistent storage")
                return

            # Reconstruct chunks
            for i, chunk_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                chunk = {
                    'chunk_id': chunk_id,
                    'doc_id': metadata['doc_id'],
                    'page': metadata['page'],
                    'text': results['documents'][i],
                    'embedding': np.array(results['embeddings'][i]),
                    'corpus_type': metadata['corpus_type'],
                    'metadata': {
                        'language': metadata.get('language', 'unknown'),
                        'has_sanskrit': metadata.get('has_sanskrit', False),
                        'word_count': metadata.get('word_count', 0),
                        'section': metadata.get('section', 'general')
                    }
                }
                self.chunks.append(chunk)

            logger.info(f"✓ Loaded {len(self.chunks)} chunks from persistent storage")

        except Exception as e:
            logger.error(f"Failed to load existing corpus: {e}")
            # If loading fails, we'll need to reprocess
            self.chunks = []

    def _add_product_knowledge(self):
        """Add structured product knowledge to the corpus"""
        logger.info("Adding structured product knowledge...")

        # Check if product knowledge already exists
        existing_product = self.collection.get(
            where={"doc_id": "nalpamaradi_product_info"}
        )

        if existing_product['ids']:
            logger.info("Product knowledge already exists, skipping...")
            return

        product_chunks = get_all_product_chunks()

        for chunk in product_chunks:
            self._embed_and_store_chunk(chunk, doc_type="product")

        logger.info(f"✓ Added {len(product_chunks)} product knowledge chunks")

        # Update metadata
        self.corpus_metadata['product_knowledge_version'] = '1.0'

    def _process_pdfs(self, pdf_paths: List[str]):
        """Process PDFs with deduplication"""
        for pdf_path in pdf_paths:
            filename = Path(pdf_path).stem
            file_hash = self._get_file_hash(pdf_path)

            # Check if already processed
            if filename in self.corpus_metadata['processed_files']:
                existing_hash = self.corpus_metadata['processed_files'][filename].get('hash')
                if existing_hash == file_hash:
                    logger.info(f"Skipping {filename} (already processed)")
                    continue
                else:
                    logger.info(f"Re-processing {filename} (file changed)")
                    # TODO: Delete old chunks if needed

            logger.info(f"Processing PDF: {filename}")

            # Extract document metadata
            doc_type = self._infer_doc_type(filename)

            # Extract pages with isolation
            pages = self.pdf_processor.extract_pages(pdf_path)

            if not pages:
                logger.warning(f"⚠️ No pages extracted from {filename}")
                continue

            # Create chunks with page awareness
            chunks = self.pdf_processor.create_chunks(
                pages=pages,
                doc_id=filename,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                min_chunk_length=config.MIN_CHUNK_LENGTH
            )

            # Embed and store chunks
            for chunk in chunks:
                self._embed_and_store_chunk(chunk, doc_type)

            # Update metadata
            self.corpus_metadata['processed_files'][filename] = {
                'hash': file_hash,
                'chunks': len(chunks),
                'doc_type': doc_type
            }

    def _embed_and_store_chunk(self, chunk: dict, doc_type: str):
        """Embed chunk and store in persistent vector database"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(chunk['text'])

            # Prepare metadata
            metadata = {
                'doc_id': chunk['doc_id'],
                'page': chunk['page'],
                'corpus_type': chunk.get('metadata', {}).get('corpus_type', doc_type),
                'language': chunk.get('metadata', {}).get('language', 'unknown'),
                'has_sanskrit': chunk.get('metadata', {}).get('has_sanskrit', False),
                'word_count': chunk.get('metadata', {}).get('word_count', 0),
                'section': chunk.get('metadata', {}).get('section', 'general')
            }

            # Store in ChromaDB (persistent)
            self.collection.add(
                ids=[chunk['chunk_id']],
                embeddings=[embedding.tolist()],
                documents=[chunk['text']],
                metadatas=[metadata]
            )

            # Store in memory for BM25
            chunk['embedding'] = embedding
            chunk['corpus_type'] = metadata['corpus_type']
            self.chunks.append(chunk)

        except Exception as e:
            logger.error(f"Failed to embed chunk {chunk['chunk_id']}: {e}")

    def _infer_doc_type(self, filename: str) -> str:
        """Infer document type from filename"""
        filename_lower = filename.lower()

        if "charaka" in filename_lower or "samhita" in filename_lower:
            return "ayurvedic_text"
        elif "safety" in filename_lower or "contraindication" in filename_lower:
            return "safety"
        elif "product" in filename_lower or "nalpamaradi" in filename_lower:
            return "product"
        else:
            return "traditional"

    def _build_bm25_index(self):
        """Build BM25 index for keyword search"""
        if not self.chunks:
            logger.warning("No chunks available for BM25 indexing")
            self.bm25 = None
            return

        tokenized_chunks = [chunk['text'].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        logger.info("✓ BM25 index built")

    def _log_corpus_stats(self):
        """Log corpus statistics"""
        stats = self.get_knowledge_stats()
        logger.info(f"  - Product chunks: {stats['product_chunks']}")
        logger.info(f"  - PDF chunks: {stats['pdf_chunks']}")
        logger.info(f"  - Total documents: {len(self.corpus_metadata['processed_files'])}")

    def intent_aware_retrieve(
        self,
        query: str,
        use_product_direct: bool = False
    ) -> Dict:
        """
        Intent-aware retrieval that separates product info from PDF enrichment

        Args:
            query: Search query
            use_product_direct: If True, prioritize product chunks as primary answer
                               If False, blend product and PDF equally

        Returns:
            Dict with separate product and pdf chunks
        """
        if not self.chunks or not self.bm25:
            logger.warning("No chunks available for retrieval")
            return {'product_chunks': [], 'pdf_chunks': [], 'all_chunks': []}

        # Get query embedding
        query_embedding = self.embedding_model.encode(query)

        # BM25 filtering
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_indices = np.argsort(bm25_scores)[-config.BM25_CANDIDATES:][::-1]
        candidates = [self.chunks[i] for i in bm25_indices]

        # Separate by corpus type and calculate similarities
        product_candidates = []
        pdf_candidates = []

        for chunk in candidates:
            similarity = np.dot(query_embedding, chunk['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk['embedding'])
            )

            chunk_with_sim = {
                'chunk_id': chunk['chunk_id'],
                'doc_id': chunk['doc_id'],
                'page': chunk['page'],
                'text': chunk['text'],
                'corpus_type': chunk.get('corpus_type', 'unknown'),
                'language': chunk.get('metadata', {}).get('language', 'unknown'),
                'section': chunk.get('metadata', {}).get('section', 'general'),
                'similarity': float(similarity)
            }

            if chunk['doc_id'] == 'nalpamaradi_product_info':
                product_candidates.append(chunk_with_sim)
            else:
                pdf_candidates.append(chunk_with_sim)

        # Sort by similarity
        product_candidates.sort(key=lambda x: x['similarity'], reverse=True)
        pdf_candidates.sort(key=lambda x: x['similarity'], reverse=True)

        if use_product_direct:
            # Product-first mode: get top product chunks, then PDF for context
            product_chunks = product_candidates[:config.PRODUCT_FIRST_RETRIEVAL]
            pdf_chunks = pdf_candidates[:config.PDF_ENRICHMENT_RETRIEVAL]
        else:
            # Balanced mode: blend both equally
            top_k = config.TOP_K_RETRIEVAL
            product_chunks = product_candidates[:max(1, top_k // 2)]
            pdf_chunks = pdf_candidates[:max(1, top_k // 2)]

        # Combine for backward compatibility
        all_chunks = product_chunks + pdf_chunks

        return {
            'product_chunks': product_chunks,
            'pdf_chunks': pdf_chunks,
            'all_chunks': all_chunks
        }

    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = None,
        bm25_candidates: int = None,
        prefer_product_info: str = 'no'  # Changed default to 'no'
    ) -> List[Dict]:
        """
        Hybrid retrieval: BM25 + semantic similarity (backward compatible)

        For new code, use intent_aware_retrieve instead
        """
        result = self.intent_aware_retrieve(query, use_product_direct=False)
        return result['all_chunks'][:top_k or config.TOP_K_RETRIEVAL]

    def build_prompt(
        self,
        query: str,
        product_chunks: List[Dict],
        pdf_chunks: List[Dict],
        system_prompt: str
    ) -> str:
        """Build LLM prompt with separated product and PDF context"""

        # Product information section
        product_context = ""
        if product_chunks:
            product_context = "\n=== PRODUCT INFORMATION (Primary Source) ===\n"
            for i, chunk in enumerate(product_chunks, 1):
                section_info = f" [{chunk.get('section', '').upper()}]" if chunk.get('section') else ""
                product_context += f"\n[P{i}] {chunk['doc_id']} (Page {chunk['page']}){section_info}\n"
                product_context += f"{chunk['text']}\n"

        # PDF/Traditional knowledge section
        pdf_context = ""
        if pdf_chunks:
            pdf_context = "\n=== TRADITIONAL AYURVEDIC KNOWLEDGE (For Validation & Context) ===\n"
            for i, chunk in enumerate(pdf_chunks, 1):
                pdf_context += f"\n[T{i}] {chunk['doc_id']} (Page {chunk['page']})\n"
                pdf_context += f"{chunk['text']}\n"

        prompt = f"""{system_prompt}

RETRIEVED CONTEXT:
{product_context}
{pdf_context}

USER QUERY: {query}

INSTRUCTIONS:
- Answer primarily using PRODUCT INFORMATION
- Use TRADITIONAL KNOWLEDGE only to:
  * Validate claims made in product info
  * Add classical Ayurvedic context
  * Explain theoretical principles
- DO NOT repeat product information with PDF chunks
- Cite sources using [P1], [P2] for product info and [T1], [T2] for traditional texts
- If information is not in context, say "I don't have that information"
- Avoid medical claims not explicitly stated in sources
- Use warm, educational tone appropriate for {config.BRAND_NAME}

ANSWER:"""

        return prompt

    def call_llm(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 1500,
        temperature: float = 0.3
    ) -> str:
        """Call Groq LLM"""
        model = model or config.LLM_MODEL

        response = self.groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content

    def extract_citations(self, text: str, product_chunks: List[Dict], pdf_chunks: List[Dict]) -> List[Dict]:
        """Extract citation markers and map to sources"""
        citations = []

        # Product citations [P1], [P2]
        product_pattern = r'\[P(\d+)\]'
        matches = re.finditer(product_pattern, text)
        seen_indices = set()

        for match in matches:
            idx = int(match.group(1)) - 1
            if idx < len(product_chunks) and idx not in seen_indices:
                chunk = product_chunks[idx]
                citations.append({
                    'label': f'P{idx + 1}',
                    'type': 'product',
                    'doc_id': chunk['doc_id'],
                    'page': chunk['page'],
                    'section': chunk.get('section', '')
                })
                seen_indices.add(idx)

        # Traditional/PDF citations [T1], [T2]
        traditional_pattern = r'\[T(\d+)\]'
        matches = re.finditer(traditional_pattern, text)
        seen_indices = set()

        for match in matches:
            idx = int(match.group(1)) - 1
            if idx < len(pdf_chunks) and idx not in seen_indices:
                chunk = pdf_chunks[idx]
                citations.append({
                    'label': f'T{idx + 1}',
                    'type': 'traditional',
                    'doc_id': chunk['doc_id'],
                    'page': chunk['page'],
                    'section': ''
                })
                seen_indices.add(idx)

        return citations

    def answer_query(
        self,
        query: str,
        use_product_direct: bool = True
    ) -> Dict:
        """
        Complete RAG pipeline with intent-aware retrieval

        Args:
            query: User question
            use_product_direct: If True, prioritize product info as primary answer
        """
        # Retrieve with separation
        retrieval_result = self.intent_aware_retrieve(query, use_product_direct=use_product_direct)

        product_chunks = retrieval_result['product_chunks']
        pdf_chunks = retrieval_result['pdf_chunks']

        # Build system prompt
        system_prompt = f"""You are an expert Ayurvedic content writer for {config.BRAND_NAME},
specializing in {config.PRODUCT_NAME}.

Your responses must:
1. Prioritize PRODUCT INFORMATION as the primary answer source
2. Use TRADITIONAL KNOWLEDGE only for validation and enrichment
3. Never repeat product details using PDF chunks
4. Use warm, educational tone
5. Avoid medical claims unless explicitly stated in sources
6. Cite sources using [P1], [P2] for product and [T1], [T2] for traditional texts
7. Recommend consulting practitioners when appropriate"""

        # Build and call LLM
        prompt = self.build_prompt(query, product_chunks, pdf_chunks, system_prompt)
        answer = self.call_llm(prompt)

        # Extract citations
        citations = self.extract_citations(answer, product_chunks, pdf_chunks)

        return {
            'query': query,
            'answer': answer,
            'citations': citations,
            'product_chunks': product_chunks,
            'pdf_chunks': pdf_chunks,
            'retrieval_mode': 'product_direct' if use_product_direct else 'balanced'
        }

    def get_knowledge_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        stats = {
            'total_chunks': len(self.chunks),
            'product_chunks': sum(1 for c in self.chunks if c['doc_id'] == 'nalpamaradi_product_info'),
            'pdf_chunks': sum(1 for c in self.chunks if c['doc_id'] != 'nalpamaradi_product_info'),
            'by_section': {},
            'by_language': {},
            'by_corpus_type': {},
            'is_persistent': self._is_corpus_ready(),
            'processed_files': list(self.corpus_metadata.get('processed_files', {}).keys())
        }

        for chunk in self.chunks:
            # Count by section
            section = chunk.get('metadata', {}).get('section', 'general')
            stats['by_section'][section] = stats['by_section'].get(section, 0) + 1

            # Count by language
            lang = chunk.get('metadata', {}).get('language', 'unknown')
            stats['by_language'][lang] = stats['by_language'].get(lang, 0) + 1

            # Count by corpus type
            corpus_type = chunk.get('corpus_type', 'unknown')
            stats['by_corpus_type'][corpus_type] = stats['by_corpus_type'].get(corpus_type, 0) + 1

        return stats

    def reset_corpus(self):
        """Reset the entire corpus (use with caution)"""
        logger.warning("Resetting corpus...")
        try:
            self.chroma_client.delete_collection(config.CHROMA_COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(
                name=config.CHROMA_COLLECTION_NAME
            )
            self.chunks = []
            self.corpus_metadata = {
                'processed_files': {},
                'product_knowledge_version': None,
                'last_updated': None
            }
            self._save_corpus_metadata()
            logger.info("✓ Corpus reset complete")
        except Exception as e:
            logger.error(f"Failed to reset corpus: {e}")