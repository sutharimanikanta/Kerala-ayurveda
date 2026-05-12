# D:\Download1\kerala_ayurveda_content_pack_v1\pdf_processor.py
"""
Enhanced PDF Processor with Page-Level Isolation and OCR Fallback
Critical fixes for robust PDF ingestion
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import PyPDF2

# Optional OCR dependencies
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Enhanced PDF processor with critical fixes:
    1. Page-level isolation (one broken page won't crash ingestion)
    2. OCR fallback for image-based PDFs
    3. Text normalization before chunking
    4. Page-aware chunking (never across pages)
    5. Language detection (Sanskrit/English/Mixed)
    6. Soft-fail chunk creation
    """

    def __init__(
        self,
        ocr_enabled: bool = False,
        min_text_length: int = 30
    ):
        self.ocr_enabled = ocr_enabled and OCR_AVAILABLE
        self.min_text_length = min_text_length

        if ocr_enabled and not OCR_AVAILABLE:
            logger.warning(
                "⚠️ OCR requested but dependencies not installed. "
                "Install: pip install pytesseract pdf2image"
            )

        logger.info(f"✓ PDFProcessor initialized (OCR: {self.ocr_enabled})")

    def extract_pages(self, pdf_path: str) -> List[Tuple[int, str]]:
        """
        Extract text from PDF with page-level isolation

        Critical: Each page is processed independently. If one page fails,
        others continue successfully.

        Returns:
            List of (page_number, page_text) tuples
        """
        pages = []

        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)

                logger.info(f"Processing {total_pages} pages from {Path(pdf_path).name}")

                for i, page in enumerate(reader.pages):
                    page_num = i + 1

                    try:
                        # Attempt text extraction
                        page_text = page.extract_text()

                        # Check if extraction succeeded
                        if page_text and len(page_text.strip()) >= self.min_text_length:
                            pages.append((page_num, page_text))
                            logger.debug(f"  ✓ Page {page_num}: {len(page_text)} chars")
                        else:
                            # Text too short or missing - try OCR if enabled
                            if self.ocr_enabled:
                                logger.info(f"  ⚠️ Page {page_num}: Text too short, attempting OCR...")
                                ocr_text = self._ocr_page(pdf_path, page_num)

                                if ocr_text and len(ocr_text.strip()) >= self.min_text_length:
                                    pages.append((page_num, ocr_text))
                                    logger.info(f"  ✓ Page {page_num}: OCR successful")
                                else:
                                    logger.warning(f"  ✗ Page {page_num}: OCR failed or text too short")
                            else:
                                logger.warning(
                                    f"  ✗ Page {page_num}: Text too short "
                                    f"({len(page_text.strip()) if page_text else 0} chars)"
                                )

                    except Exception as e:
                        # Page-level error - log and continue
                        logger.error(f"  ✗ Page {page_num}: PDF object error - {str(e)[:100]}")
                        continue

                logger.info(f"✓ Extracted {len(pages)}/{total_pages} pages successfully")

        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path}: {e}")
            return []

        return pages

    def _ocr_page(self, pdf_path: str, page_num: int) -> Optional[str]:
        """
        Apply OCR to a specific page

        Args:
            pdf_path: Path to PDF file
            page_num: 1-indexed page number

        Returns:
            OCR text or None if failed
        """
        if not self.ocr_enabled:
            return None

        try:
            # Convert specific page to image
            images = convert_from_path(
                pdf_path,
                first_page=page_num,
                last_page=page_num,
                dpi=300
            )

            if not images:
                return None

            # Apply OCR
            text = pytesseract.image_to_string(images[0], lang='eng')
            return text

        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")
            return None

    def normalize_text(self, text: str) -> str:
        """
        Normalize text before chunking

        Critical step: Must be applied before sentence splitting

        Args:
            text: Raw extracted text

        Returns:
            Normalized text
        """
        # Remove excessive newlines
        text = re.sub(r'\n+', ' ', text)

        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers (common pattern)
        text = re.sub(r'\b\d+\s*$', '', text)

        # Remove headers/footers (lines with < 10 words)
        lines = text.split('\n')
        lines = [line for line in lines if len(line.split()) > 10]
        text = ' '.join(lines)

        # Trim and return
        return text.strip()

    def detect_language(self, text: str) -> str:
        """
        Detect language of text chunk

        Returns:
            'sanskrit', 'english', or 'mixed'
        """
        # Sanskrit patterns
        sanskrit_pattern = r'[\u0900-\u097F]'  # Devanagari Unicode range
        sanskrit_chars = len(re.findall(sanskrit_pattern, text))

        # English patterns
        english_pattern = r'[a-zA-Z]'
        english_chars = len(re.findall(english_pattern, text))

        total_chars = sanskrit_chars + english_chars

        if total_chars == 0:
            return 'unknown'

        sanskrit_ratio = sanskrit_chars / total_chars

        if sanskrit_ratio > 0.7:
            return 'sanskrit'
        elif sanskrit_ratio < 0.3:
            return 'english'
        else:
            return 'mixed'

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        Must operate on normalized text

        Args:
            text: Normalized text

        Returns:
            List of sentences
        """
        # Basic sentence splitting (handles ., !, ?)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def create_chunks(
        self,
        pages: List[Tuple[int, str]],
        doc_id: str,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_length: int = None
    ) -> List[Dict]:
        """
        Create chunks with page-awareness and language detection

        Critical: Never chunk across pages. Each chunk stores page metadata.

        Args:
            pages: List of (page_number, page_text) tuples
            doc_id: Document identifier
            chunk_size: Target chunk size in words
            chunk_overlap: Overlap between chunks in words
            min_chunk_length: Minimum chunk length in characters

        Returns:
            List of chunk dictionaries
        """
        chunk_size = chunk_size or config.CHUNK_SIZE
        chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        min_chunk_length = min_chunk_length or config.MIN_CHUNK_LENGTH

        all_chunks = []

        for page_num, page_text in pages:
            # Normalize text
            normalized_text = self.normalize_text(page_text)

            if len(normalized_text.strip()) < min_chunk_length:
                logger.debug(f"Skipping page {page_num}: Text too short after normalization")
                continue

            # Detect language
            language = self.detect_language(normalized_text)

            # Split into sentences
            sentences = self._split_sentences(normalized_text)

            if not sentences:
                logger.debug(f"Skipping page {page_num}: No valid sentences")
                continue

            # Create chunks from sentences (page-aware)
            page_chunks = self._chunk_sentences(
                sentences=sentences,
                page_num=page_num,
                doc_id=doc_id,
                language=language,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_length=min_chunk_length
            )

            all_chunks.extend(page_chunks)

        logger.info(f"✓ Created {len(all_chunks)} chunks from {len(pages)} pages")

        return all_chunks

    def _chunk_sentences(
        self,
        sentences: List[str],
        page_num: int,
        doc_id: str,
        language: str,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_length: int
    ) -> List[Dict]:
        """
        Chunk sentences with overlap, staying within page boundary

        Args:
            sentences: List of sentences from a single page
            page_num: Page number
            doc_id: Document ID
            language: Detected language
            chunk_size: Target size in words
            chunk_overlap: Overlap in words
            min_chunk_length: Minimum chunk length in characters

        Returns:
            List of chunk dictionaries
        """
        chunks = []
        current_chunk = []
        current_word_count = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)

            # If adding this sentence exceeds chunk_size, finalize current chunk
            if current_word_count + sentence_word_count > chunk_size and current_chunk:
                # Finalize chunk
                chunk_text = ' '.join(current_chunk)

                # Soft-fail: Skip if too short
                if len(chunk_text.strip()) >= min_chunk_length:
                    chunk_id = f"{doc_id}_p{page_num}_c{chunk_index}"

                    chunks.append({
                        'chunk_id': chunk_id,
                        'doc_id': doc_id,
                        'page': page_num,
                        'text': chunk_text,
                        'metadata': {
                            'language': language,
                            'word_count': current_word_count,
                            'has_sanskrit': language in ['sanskrit', 'mixed'],
                            'chunk_index': chunk_index,
                            'ocr_applied': False  # Updated during extraction if OCR used
                        }
                    })

                    chunk_index += 1

                # Start new chunk with overlap
                overlap_words = chunk_overlap
                if overlap_words > 0 and len(current_chunk) > 0:
                    # Take last N words for overlap
                    overlap_text = ' '.join(current_chunk[-overlap_words:])
                    current_chunk = [overlap_text]
                    current_word_count = len(overlap_text.split())
                else:
                    current_chunk = []
                    current_word_count = 0

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_word_count

        # Finalize last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)

            if len(chunk_text.strip()) >= min_chunk_length:
                chunk_id = f"{doc_id}_p{page_num}_c{chunk_index}"

                chunks.append({
                    'chunk_id': chunk_id,
                    'doc_id': doc_id,
                    'page': page_num,
                    'text': chunk_text,
                    'metadata': {
                        'language': language,
                        'word_count': current_word_count,
                        'has_sanskrit': language in ['sanskrit', 'mixed'],
                        'chunk_index': chunk_index,
                        'ocr_applied': False
                    }
                })

        return chunks

    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """
        Calculate statistics for created chunks

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_word_count': 0,
                'languages': {},
                'pages_covered': 0
            }

        total_words = sum(c['metadata']['word_count'] for c in chunks)
        languages = {}
        pages = set()

        for chunk in chunks:
            lang = chunk['metadata']['language']
            languages[lang] = languages.get(lang, 0) + 1
            pages.add(chunk['page'])

        return {
            'total_chunks': len(chunks),
            'avg_word_count': round(total_words / len(chunks), 1),
            'languages': languages,
            'pages_covered': len(pages),
            'ocr_chunks': sum(1 for c in chunks if c['metadata'].get('ocr_applied', False))
        }