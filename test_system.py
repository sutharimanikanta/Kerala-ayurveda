# D:\Download1\kerala_ayurveda_content_pack_v1\test_system.py
"""
System Testing Script
Tests all components of the Nalpamaradi Keram RAG system
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported"""
    logger.info("Testing module imports...")

    try:
        logger.info("  ✓ config.py")
    except Exception as e:
        logger.error(f"  ✗ config.py: {e}")
        return False

    try:
        logger.info("  ✓ pdf_processor.py")
    except Exception as e:
        logger.error(f"  ✗ pdf_processor.py: {e}")
        return False

    try:
        logger.info("  ✓ rag_pipeline.py")
    except Exception as e:
        logger.error(f"  ✗ rag_pipeline.py: {e}")
        return False

    try:
        logger.info("  ✓ agents.py")
    except Exception as e:
        logger.error(f"  ✗ agents.py: {e}")
        return False

    try:
        logger.info("  ✓ orchestrator.py")
    except Exception as e:
        logger.error(f"  ✗ orchestrator.py: {e}")
        return False

    try:
        logger.info("  ✓ utils.py")
    except Exception as e:
        logger.error(f"  ✗ utils.py: {e}")
        return False

    logger.info("✅ All modules imported successfully")
    return True


def test_config():
    """Test configuration"""
    logger.info("Testing configuration...")

    try:
        import config

        # Check API key
        if config.GROQ_API_KEY == "your_groq_api_key_here":
            logger.error("  ✗ GROQ_API_KEY not set in config.py")
            return False
        else:
            logger.info("  ✓ GROQ_API_KEY configured")

        # Check other settings
        logger.info(f"  ✓ Embedding model: {config.EMBEDDING_MODEL}")
        logger.info(f"  ✓ LLM model: {config.LLM_MODEL}")
        logger.info(f"  ✓ Chunk size: {config.CHUNK_SIZE}")
        logger.info(f"  ✓ Max retries: {config.MAX_RETRIES}")

        logger.info("✅ Configuration valid")
        return True

    except Exception as e:
        logger.error(f"  ✗ Configuration error: {e}")
        return False


def test_pdf_processor():
    """Test PDF processor"""
    logger.info("Testing PDF processor...")

    try:
        from pdf_processor import PDFProcessor

        processor = PDFProcessor(ocr_enabled=False)
        logger.info("  ✓ PDFProcessor initialized")

        # Test text normalization
        test_text = "This  is   a\n\ntest   text"
        normalized = processor.normalize_text(test_text)
        assert len(normalized) < len(test_text)
        logger.info("  ✓ Text normalization works")

        # Test sentence splitting
        test_text = "First sentence. Second sentence! Third sentence?"
        sentences = processor._split_sentences(test_text)
        assert len(sentences) == 3
        logger.info("  ✓ Sentence splitting works")

        logger.info("✅ PDF processor functional")
        return True

    except Exception as e:
        logger.error(f"  ✗ PDF processor error: {e}")
        return False


def test_utils():
    """Test utility functions"""
    logger.info("Testing utilities...")

    try:
        from utils import validate_medical_claims, calculate_readability_score, format_citations

        # Test medical claim detection
        text_with_claim = "This product cures diseases"
        result = validate_medical_claims(text_with_claim)
        assert result['has_medical_claims'] == True
        assert result['severity'] == 'high'
        logger.info("  ✓ Medical claim detection works")

        text_without_claim = "This product may support wellness"
        result = validate_medical_claims(text_without_claim)
        assert result['has_medical_claims'] == False
        logger.info("  ✓ Safe text passes validation")

        # Test readability calculation
        test_text = "This is a test sentence. Another test sentence."
        readability = calculate_readability_score(test_text)
        assert 'word_count' in readability
        assert 'readability_level' in readability
        logger.info("  ✓ Readability calculation works")

        # Test citation formatting
        citations = [
            {'label': 1, 'doc_id': 'test_doc', 'page': 10},
            {'label': 2, 'doc_id': 'another_doc', 'page': 25}
        ]
        formatted = format_citations(citations)
        assert 'test_doc' in formatted
        assert 'Page 10' in formatted
        logger.info("  ✓ Citation formatting works")

        logger.info("✅ Utilities functional")
        return True

    except Exception as e:
        logger.error(f"  ✗ Utilities error: {e}")
        return False


def test_dependencies():
    """Test required dependencies"""
    logger.info("Testing dependencies...")

    dependencies = [
        ('streamlit', 'Streamlit'),
        ('chromadb', 'ChromaDB'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('rank_bm25', 'BM25'),
        ('PyPDF2', 'PyPDF2'),
        ('groq', 'Groq'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas')
    ]

    all_ok = True
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            logger.info(f"  ✓ {display_name}")
        except ImportError:
            logger.error(f"  ✗ {display_name} not installed")
            all_ok = False

    # Check optional OCR dependencies
    try:
        import pytesseract
        import pdf2image
        logger.info("  ✓ OCR support available")
    except ImportError:
        logger.warning("  ⚠ OCR support not available (optional)")

    if all_ok:
        logger.info("✅ All required dependencies installed")
    else:
        logger.error("❌ Some dependencies missing. Run: pip install -r requirements.txt")

    return all_ok


def run_all_tests():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Nalpamaradi Keram RAG System - System Tests")
    logger.info("=" * 60)

    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("PDF Processor", test_pdf_processor),
        ("Utilities", test_utils)
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info("")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}  {test_name}")

    logger.info("=" * 60)

    all_passed = all(results.values())

    if all_passed:
        logger.info("🎉 All tests passed! System is ready to use.")
        logger.info("Run: streamlit run app.py")
    else:
        logger.error("⚠️  Some tests failed. Please fix issues before running.")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()