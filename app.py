# # # D:\Download1\kerala_ayurveda_content_pack_v1\app.py
"""
Nalpamaradi Keram Agentic Content Generation System
Main Streamlit Application with Persistent Corpus Support
"""

import streamlit as st
import json
import time

import config
from rag_pipeline import RAGPipeline
from orchestrator import MainOrchestrator, WorkflowMode
from utils import format_citations

# Page configuration MUST be first
st.set_page_config(
    page_title=f"{config.PRODUCT_NAME} Content Agent",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize ALL session state at the top
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "result" not in st.session_state:
    st.session_state.result = None
if "execution_trace" not in st.session_state:
    st.session_state.execution_trace = []
if "corpus_loaded" not in st.session_state:
    st.session_state.corpus_loaded = False

# Custom CSS
st.markdown(
    """
<style>
.step-box {
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 4px solid;
}
.step-idle { border-color: #gray; background: #f0f0f0; }
.step-active { border-color: #1E88E5; background: #E3F2FD; }
.step-complete { border-color: #43A047; background: #E8F5E9; }
.step-error { border-color: #E53935; background: #FFEBEE; }
.info-box {
    padding: 12px;
    border-radius: 6px;
    background: #E8F4F8;
    border-left: 4px solid #2196F3;
    margin: 10px 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# Helper function to check if pipeline is ready
def is_pipeline_ready():
    return (
        st.session_state.pipeline is not None
        and st.session_state.orchestrator is not None
    )


# Header
st.title(f"🌿 {config.PRODUCT_NAME} Content Generation System")
st.markdown("*Autonomous Agentic RAG Pipeline with Persistent Corpus*")

# Sidebar - System Configuration
with st.sidebar:
    st.header("⚙️ System Configuration")

    # API Key Status
    api_status = (
        "🟢 Configured"
        if config.GROQ_API_KEY != "your_groq_api_key_here"
        else "🔴 Not Configured"
    )
    st.info(f"API Key: {api_status}")

    if api_status == "🔴 Not Configured":
        st.error("Please set GROQ_API_KEY in config.py")
        st.stop()

    # Check for existing corpus
    corpus_exists = (
        config.CHROMA_DIR.exists() and len(list(config.CHROMA_DIR.iterdir())) > 0
    )

    if corpus_exists:
        st.success("📦 Persistent Corpus Detected")

        # Load existing corpus
        if st.button(
            "🔄 Load Existing Corpus", type="primary", use_container_width=True
        ):
            with st.spinner("Loading corpus from persistent storage..."):
                try:
                    st.session_state.pipeline = RAGPipeline()
                    st.session_state.orchestrator = MainOrchestrator(
                        st.session_state.pipeline
                    )
                    st.session_state.corpus_loaded = True
                    st.success("✅ Corpus loaded successfully!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading corpus: {e}")
                    st.session_state.pipeline = None
                    st.session_state.orchestrator = None

    # PDF Upload Section
    st.subheader("📄 Knowledge Base Management")

    uploaded_files = st.file_uploader(
        "Upload Additional PDFs" if corpus_exists else "Upload Ayurvedic PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload reference PDFs (Charaka Samhita, product info, etc.)",
    )

    col1, col2 = st.columns(2)

    with col1:
        # Process new PDFs
        if st.button(
            "➕ Add PDFs" if corpus_exists else "🚀 Initialize",
            type="secondary" if corpus_exists else "primary",
            use_container_width=True,
        ):
            if not uploaded_files:
                st.error("Please upload at least one PDF")
            else:
                with st.spinner("Processing PDFs..."):
                    try:
                        # Save uploaded files
                        pdf_paths = []
                        for uploaded_file in uploaded_files:
                            file_path = config.PDF_DIR / uploaded_file.name
                            file_path.write_bytes(uploaded_file.read())
                            pdf_paths.append(str(file_path))

                        # Initialize or update pipeline
                        if st.session_state.pipeline is None:
                            st.session_state.pipeline = RAGPipeline(pdf_paths=pdf_paths)
                        else:
                            # Add to existing pipeline
                            st.session_state.pipeline = RAGPipeline(
                                pdf_paths=pdf_paths, force_reingest=False
                            )

                        st.session_state.orchestrator = MainOrchestrator(
                            st.session_state.pipeline
                        )
                        st.session_state.corpus_loaded = True
                        st.success("✅ PDFs processed successfully!")
                        time.sleep(1)
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error: {e}")

    with col2:
        # Reset corpus
        if corpus_exists:
            if st.button("🗑️ Reset", type="secondary", use_container_width=True):
                if st.session_state.pipeline:
                    st.session_state.pipeline.reset_corpus()
                    st.session_state.pipeline = None
                    st.session_state.orchestrator = None
                    st.session_state.corpus_loaded = False
                    st.warning("Corpus reset. Please reinitialize.")
                    time.sleep(1)
                    st.rerun()

    # Pipeline Status
    st.divider()
    if is_pipeline_ready():
        st.success("🟢 Pipeline Active")

        # Get knowledge stats
        stats = st.session_state.pipeline.get_knowledge_stats()

        st.metric("Total Chunks", stats["total_chunks"])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Product", stats["product_chunks"])
        with col2:
            st.metric("PDFs", stats["pdf_chunks"])

        # Show processed files
        if stats.get("processed_files"):
            with st.expander("📚 Processed Files"):
                for filename in stats["processed_files"]:
                    st.text(f"  ✓ {filename}")

        # Show sections
        if stats["by_section"]:
            with st.expander("📑 Sections"):
                for section, count in sorted(stats["by_section"].items()):
                    st.text(f"  {section}: {count}")

        # Persistence indicator
        if stats.get("is_persistent"):
            st.caption("💾 Using persistent storage")

    else:
        st.warning("🔴 Pipeline Not Initialized")

    st.divider()
    st.caption(f"{config.PRODUCT_NAME} v2.1")
    st.caption("Persistent Multi-Agent System")

# Main Content - Show different UI based on pipeline state
if not is_pipeline_ready():
    st.info("👈 Please initialize the RAG pipeline from the sidebar to begin")

    # System Architecture
    st.subheader("🏗️ System Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Persistent Corpus Features:**
        - ✅ One-time PDF processing
        - ✅ Instant corpus loading on restart
        - ✅ Incremental PDF addition
        - ✅ Deduplication by file hash
        - ✅ Product knowledge always available

        **Autonomous Multi-Agent Pipeline:**
        1. **Main Orchestrator** - Controls workflow automatically
        2. **Outline Agent** - Structured planning with RAG validation
        3. **Writer Agent** - Draft generation with citations
        4. **Fact-Checker Agent** - Validates claims against corpus
        5. **Finalization Agent** - Polishes and prepares output
        """)

    with col2:
        st.markdown("""
        **Intent-Aware Q&A:**
        - Product info as primary source
        - PDF chunks for validation & context
        - No repetition between sources
        - Separate citation tracking

        **Enhanced PDF Processing:**
        - ✅ Page-level isolation
        - ✅ OCR fallback
        - ✅ Text normalization
        - ✅ Language detection
        - ✅ Persistent vector DB
        - ✅ Fast BM25 + semantic hybrid search
        """)

else:
    # Pipeline is ready - show tabs
    tab1, tab2 = st.tabs(["📝 Content Generation", "❓ Question & Answer"])

    # TAB 1: Content Generation
    with tab1:
        st.header("Autonomous Content Generation")
        st.markdown("Single-click workflow with automatic agent orchestration")

        # Input Form
        with st.form("content_generation_form"):
            st.subheader("Content Brief")

            col1, col2 = st.columns([2, 1])

            with col1:
                brief_text = st.text_area(
                    "Content Brief",
                    height=150,
                    placeholder=f"Example: Write an 800-word article about {config.PRODUCT_NAME} for skin radiance...",
                    help="Describe what you want the article to cover",
                )

            with col2:
                audience = st.selectbox(
                    "Target Audience",
                    [
                        "General consumers",
                        "Ayurveda enthusiasts",
                        "Working professionals",
                        "Skincare conscious individuals",
                    ],
                )

                word_count = st.slider("Target Word Count", 500, 1500, 800, 100)

                tone = st.selectbox(
                    "Tone",
                    [
                        "Warm and educational",
                        "Professional",
                        "Conversational",
                        "Traditional",
                    ],
                )

            submitted = st.form_submit_button(
                "🚀 Run Complete Workflow", type="primary", use_container_width=True
            )

        # Execute Workflow
        if submitted:
            if not brief_text:
                st.error("Please provide a content brief")
            else:
                # Clear previous results
                st.session_state.result = None
                st.session_state.execution_trace = []

                user_input = {
                    "brief": brief_text,
                    "product": config.PRODUCT_NAME,
                    "audience": audience,
                    "word_count": word_count,
                    "tone": tone,
                }

                progress_container = st.container()

                with progress_container:
                    st.subheader("🔄 Workflow Progress")
                    status_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    status_placeholder.info("🧠 Starting workflow...")

                try:
                    result = st.session_state.orchestrator.execute_workflow(
                        mode=WorkflowMode.CONTENT_GENERATION.value,
                        user_input=user_input,
                    )

                    # Store successful result
                    st.session_state.result = result
                    st.session_state.execution_trace = result.get("execution_trace", [])

                    progress_bar.progress(100)
                    status_placeholder.success("✅ Workflow completed successfully!")

                    # Force rerun to show results
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    progress_bar.progress(100)
                    status_placeholder.error(f"❌ Workflow failed: {str(e)}")

                    # Capture partial results if available
                    if (
                        st.session_state.orchestrator
                        and st.session_state.orchestrator.execution_trace
                    ):
                        st.session_state.execution_trace = (
                            st.session_state.orchestrator.execution_trace
                        )

                    # Show error details in expander
                    with st.expander("Error Details", expanded=True):
                        st.exception(e)

        # Display Results
        if (
            st.session_state.result
            and st.session_state.result.get("mode") == "content_generation"
        ):
            st.divider()
            st.subheader("📊 Execution Trace")

            # Show execution steps
            with st.expander("View Detailed Execution Log", expanded=False):
                for entry in st.session_state.execution_trace:
                    step_type = entry["step_type"]
                    message = entry["message"]

                    if step_type == "ERROR":
                        st.error(f"🔴 {message}")
                    elif step_type in ["COMPLETED", "FACT_CHECK_PASSED"]:
                        st.success(f"✅ {message}")
                    elif "COMPLETE" in step_type:
                        st.info(f"✓ {message}")
                    else:
                        st.text(f"• {message}")

            # Show retry count
            total_retries = st.session_state.result.get("total_retries", 0)
            if total_retries > 0:
                st.warning(
                    f"⚠️ Required {total_retries} revision(s) to pass fact-checking"
                )

            st.divider()

            # Display Final Output
            result = st.session_state.result
            final_output = result["final_output"]

            st.subheader("📄 Final Article")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Status", final_output["status"])
            col2.metric("Paragraphs", len(result["draft"]["paragraphs"]))
            col3.metric("Citations", len(final_output["citations"]))
            col4.metric("Revisions", total_retries)

            # Article Text
            st.markdown("---")
            st.markdown(final_output["text"])
            st.markdown("---")

            # Citations
            st.subheader("📚 Citations")
            if final_output["citations"]:
                citation_text = format_citations(final_output["citations"])
                st.code(citation_text, language="text")
            else:
                st.info("No citations (narrative content only)")

            # Editor Notes
            if final_output.get("notes_for_editor"):
                st.subheader("📝 Notes for Editor")
                for note in final_output["notes_for_editor"]:
                    st.info(note)

            # Export Options
            st.subheader("💾 Export")

            col1, col2, col3 = st.columns(3)

            with col1:
                # Download Markdown
                md_content = f"""# {config.PRODUCT_NAME} Article

{final_output["text"]}

## Citations

"""
                for cit in final_output["citations"]:
                    md_content += (
                        f"[{cit['label']}] {cit['doc_id']} (Page {cit['page']})\n"
                    )

                st.download_button(
                    "📥 Download Markdown",
                    md_content,
                    f"{config.PRODUCT_NAME.lower().replace(' ', '_')}_article.md",
                    "text/markdown",
                    use_container_width=True,
                )

            with col2:
                # Download JSON
                st.download_button(
                    "📥 Download JSON",
                    json.dumps(result, indent=2, default=str),
                    f"{config.PRODUCT_NAME.lower().replace(' ', '_')}_article.json",
                    "application/json",
                    use_container_width=True,
                )

            with col3:
                if st.button("🔄 Start New Article", use_container_width=True):
                    st.session_state.result = None
                    st.session_state.execution_trace = []
                    st.rerun()

    # TAB 2: Question & Answer
    with tab2:
        st.header("Question & Answer")
        st.markdown("Intent-aware retrieval with corpus type separation")

        # Show pipeline status at top of tab
        stats = st.session_state.pipeline.get_knowledge_stats()
        st.info(
            f"✅ Pipeline Ready | {stats['total_chunks']} chunks ({stats['product_chunks']} product + {stats['pdf_chunks']} PDFs)"
        )

        # Retrieval mode selector
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**Retrieval Strategy:**")
        use_product_direct = st.checkbox(
            "Use product info as primary answer source",
            value=True,
            help="When enabled: Product chunks provide the main answer, PDF chunks add context. When disabled: Blend both equally.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        query = st.text_input(
            "Your Question",
            placeholder=f"Example: What are the key ingredients in {config.PRODUCT_NAME}?",
        )

        if st.button("🔍 Search", type="primary"):
            if not query:
                st.error("Please enter a question")
            else:
                with st.spinner("Searching knowledge base..."):
                    try:
                        result = st.session_state.pipeline.answer_query(
                            query, use_product_direct=use_product_direct
                        )

                        # Display answer
                        st.subheader("Answer")
                        st.markdown(result["answer"])

                        # Show retrieval mode
                        mode_icon = (
                            "🎯"
                            if result["retrieval_mode"] == "product_direct"
                            else "⚖️"
                        )
                        st.caption(
                            f"{mode_icon} Retrieval mode: {result['retrieval_mode']}"
                        )

                        # Citations
                        if result["citations"]:
                            st.subheader("Sources")

                            # Separate citations by type
                            product_cites = [
                                c
                                for c in result["citations"]
                                if c.get("type") == "product"
                            ]
                            trad_cites = [
                                c
                                for c in result["citations"]
                                if c.get("type") == "traditional"
                            ]

                            if product_cites:
                                st.markdown("**Product Information:**")
                                for cit in product_cites:
                                    section = (
                                        f" - {cit['section']}"
                                        if cit.get("section")
                                        else ""
                                    )
                                    st.text(
                                        f"[{cit['label']}] {cit['doc_id']} (Page {cit['page']}){section}"
                                    )

                            if trad_cites:
                                st.markdown("**Traditional Knowledge:**")
                                for cit in trad_cites:
                                    st.text(
                                        f"[{cit['label']}] {cit['doc_id']} (Page {cit['page']})"
                                    )

                        # Retrieved chunks (organized by type)
                        with st.expander("View Retrieved Context"):
                            if result.get("product_chunks"):
                                st.markdown("**📦 Product Chunks (Primary):**")
                                for i, chunk in enumerate(result["product_chunks"], 1):
                                    st.markdown(
                                        f"**[P{i}] {chunk['doc_id']} - {chunk.get('section', 'general')}** (Similarity: {chunk['similarity']:.3f})"
                                    )
                                    st.text(chunk["text"][:300] + "...")
                                    st.divider()

                            if result.get("pdf_chunks"):
                                st.markdown("**📚 Traditional Knowledge (Context):**")
                                for i, chunk in enumerate(result["pdf_chunks"], 1):
                                    st.markdown(
                                        f"**[T{i}] {chunk['doc_id']} (Page {chunk['page']})** (Similarity: {chunk['similarity']:.3f})"
                                    )
                                    st.text(chunk["text"][:300] + "...")
                                    st.divider()

                    except Exception as e:
                        st.error(f"Error: {e}")
                        with st.expander("Error Details"):
                            st.exception(e)

# Footer
st.divider()
st.caption(
    f"{config.PRODUCT_NAME} Agentic Content System | {config.BRAND_NAME} RAG Pipeline v2.1"
)
st.caption("Persistent Multi-Agent Orchestration with Intent-Aware Retrieval")
# •	What are the key ingredients in Nalpamaradi keram
# •	can a pregnant women use  Nalpamaradi keram
# “What is Nalpamaradi Keram?”
# “How do I use it on face?”
# “What is Nalpamara in Ayurveda?”
# “Does it cure eczema?”
