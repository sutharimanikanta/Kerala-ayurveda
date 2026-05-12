"""
Nalpamaradi Keram Agentic Content Generation System
Main Streamlit Application with Autonomous Orchestration
"""

import streamlit as st
import json
from pathlib import Path
import time

import config
from rag_pipeline import RAGPipeline
from orchestrator import MainOrchestrator, WorkflowMode
from utils import format_citations

# Page configuration
st.set_page_config(
    page_title=f"{config.PRODUCT_NAME} Content Agent",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for progress indicators
st.markdown("""
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'result' not in st.session_state:
    st.session_state.result = None
if 'execution_trace' not in st.session_state:
    st.session_state.execution_trace = []

# Header
st.title(f"🌿 {config.PRODUCT_NAME} Content Generation System")
st.markdown(f"*Autonomous Agentic RAG Pipeline for {config.BRAND_NAME}*")

# Sidebar - System Configuration
with st.sidebar:
    st.header("⚙️ System Configuration")

    # API Key Status
    api_status = "🟢 Configured" if config.GROQ_API_KEY != "your_groq_api_key_here" else "🔴 Not Configured"
    st.info(f"API Key: {api_status}")

    if api_status == "🔴 Not Configured":
        st.error("Please set GROQ_API_KEY in config.py")
        st.stop()

    # PDF Upload
    st.subheader("📄 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Ayurvedic PDFs",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload reference PDFs (Charaka Samhita, product info, etc.)"
    )

    # Initialize Pipeline
    if st.button("🚀 Initialize RAG Pipeline", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("Please upload at least one PDF")
        else:
            with st.spinner("Initializing RAG pipeline..."):
                try:
                    # Save uploaded files
                    pdf_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = Path(f"temp_{uploaded_file.name}")
                        file_path.write_bytes(uploaded_file.read())
                        pdf_paths.append(str(file_path))

                    # Initialize pipeline
                    st.session_state.pipeline = RAGPipeline(pdf_paths=pdf_paths)
                    st.session_state.orchestrator = MainOrchestrator(st.session_state.pipeline)

                    st.success("✅ Pipeline initialized successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")

    # Pipeline Status
    st.divider()
    if st.session_state.pipeline:
        st.success("🟢 Pipeline Active")
        st.metric("Total Chunks", len(st.session_state.pipeline.chunks))

        # Show chunk distribution
        if st.session_state.pipeline.chunks:
            chunk_langs = {}
            for chunk in st.session_state.pipeline.chunks:
                lang = chunk['metadata'].get('language', 'unknown')
                chunk_langs[lang] = chunk_langs.get(lang, 0) + 1

            st.caption("Language Distribution:")
            for lang, count in chunk_langs.items():
                st.text(f"  {lang}: {count}")
    else:
        st.warning("🔴 Pipeline Not Initialized")

    st.divider()
    st.caption(f"{config.PRODUCT_NAME} v2.0")
    st.caption("Autonomous Multi-Agent System")

# Main Content
if not st.session_state.pipeline:
    st.info("👈 Please initialize the RAG pipeline from the sidebar to begin")

    # System Architecture
    st.subheader("🏗️ System Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Autonomous Multi-Agent Pipeline:**
        1. **Main Orchestrator** - Controls workflow automatically
        2. **Outline Agent** - Structured planning with RAG validation
        3. **Writer Agent** - Draft generation with citations
        4. **Fact-Checker Agent** - Validates claims against corpus
        5. **Finalization Agent** - Polishes and prepares output

        **Key Features:**
        - Single-button execution
        - Automatic retry loops
        - Real-time progress tracking
        - Page-level PDF parsing with OCR fallback
        - Language-aware chunking
        """)

    with col2:
        st.markdown("""
        **Enhanced PDF Processing:**
        - ✅ Page-level isolation (one broken page won't crash ingestion)
        - ✅ OCR fallback for image-based PDFs
        - ✅ Text normalization before chunking
        - ✅ Page-aware chunking (never across pages)
        - ✅ Language detection (Sanskrit/English/Mixed)
        - ✅ Soft-fail chunk creation

        **Safety Features:**
        - Medical claim validation
        - Citation tracking
        - Fact-checking loops
        - Max retry enforcement
        """)

else:
    # Tabs for workflows
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
                    help="Describe what you want the article to cover"
                )

            with col2:
                audience = st.selectbox(
                    "Target Audience",
                    ["General consumers", "Ayurveda enthusiasts",
                     "Working professionals", "Skincare conscious individuals"]
                )

                word_count = st.slider("Target Word Count", 500, 1500, 800, 100)

                tone = st.selectbox(
                    "Tone",
                    ["Warm and educational", "Professional",
                     "Conversational", "Traditional"]
                )

            submitted = st.form_submit_button(
                "🚀 Run Complete Workflow",
                type="primary",
                use_container_width=True
            )

        # Execute Workflow
        if submitted:
            if not brief_text:
                st.error("Please provide a content brief")
            else:
                # Clear previous results
                st.session_state.result = None
                st.session_state.execution_trace = []

                # Prepare input
                user_input = {
                    'brief': brief_text,
                    'product': config.PRODUCT_NAME,
                    'audience': audience,
                    'word_count': word_count,
                    'tone': tone
                }

                # Progress display
                progress_container = st.container()
                trace_container = st.container()

                with progress_container:
                    st.subheader("🔄 Workflow Progress")

                    # Progress placeholders
                    status_placeholder = st.empty()
                    progress_bar = st.progress(0)

                    status_placeholder.info("🧠 Starting workflow...")

                try:
                    # Execute workflow
                    result = st.session_state.orchestrator.execute_workflow(
                        mode=WorkflowMode.CONTENT_GENERATION.value,
                        user_input=user_input
                    )

                    st.session_state.result = result
                    st.session_state.execution_trace = result.get('execution_trace', [])

                    progress_bar.progress(100)
                    status_placeholder.success("✅ Workflow completed successfully!")

                    # Force rerun to show results
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    progress_bar.progress(100)
                    status_placeholder.error(f"❌ Workflow failed: {str(e)}")

                    if st.session_state.orchestrator.execution_trace:
                        st.session_state.execution_trace = st.session_state.orchestrator.execution_trace

        # Display Results
        if st.session_state.result and st.session_state.result.get('mode') == 'content_generation':
            st.divider()
            st.subheader("📊 Execution Trace")

            # Show execution steps
            with st.expander("View Detailed Execution Log", expanded=False):
                for entry in st.session_state.execution_trace:
                    step_type = entry['step_type']
                    message = entry['message']

                    if step_type == 'ERROR':
                        st.error(f"🔴 {message}")
                    elif step_type in ['COMPLETED', 'FACT_CHECK_PASSED']:
                        st.success(f"✅ {message}")
                    elif 'COMPLETE' in step_type:
                        st.info(f"✓ {message}")
                    else:
                        st.text(f"• {message}")

            # Show retry count
            total_retries = st.session_state.result.get('total_retries', 0)
            if total_retries > 0:
                st.warning(f"⚠️ Required {total_retries} revision(s) to pass fact-checking")

            st.divider()

            # Display Final Output
            result = st.session_state.result
            final_output = result['final_output']

            st.subheader("📄 Final Article")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Status", final_output['status'])
            col2.metric("Paragraphs", len(result['draft']['paragraphs']))
            col3.metric("Citations", len(final_output['citations']))
            col4.metric("Revisions", total_retries)

            # Article Text
            st.markdown("---")
            st.markdown(final_output['text'])
            st.markdown("---")

            # Citations
            st.subheader("📚 Citations")
            if final_output['citations']:
                citation_text = format_citations(final_output['citations'])
                st.code(citation_text, language="text")
            else:
                st.info("No citations (narrative content only)")

            # Editor Notes
            if final_output.get('notes_for_editor'):
                st.subheader("📝 Notes for Editor")
                for note in final_output['notes_for_editor']:
                    st.info(note)

            # Export Options
            st.subheader("💾 Export")

            col1, col2, col3 = st.columns(3)

            with col1:
                # Download Markdown
                md_content = f"""# {config.PRODUCT_NAME} Article

{final_output['text']}

## Citations

"""
                for cit in final_output['citations']:
                    md_content += f"[{cit['label']}] {cit['doc_id']} (Page {cit['page']})\n"

                st.download_button(
                    "📥 Download Markdown",
                    md_content,
                    f"{config.PRODUCT_NAME.lower().replace(' ', '_')}_article.md",
                    "text/markdown",
                    use_container_width=True
                )

            with col2:
                # Download JSON
                st.download_button(
                    "📥 Download JSON",
                    json.dumps(result, indent=2, default=str),
                    f"{config.PRODUCT_NAME.lower().replace(' ', '_')}_article.json",
                    "application/json",
                    use_container_width=True
                )

            with col3:
                if st.button("🔄 Start New Article", use_container_width=True):
                    st.session_state.result = None
                    st.session_state.execution_trace = []
                    st.rerun()

    # TAB 2: Question & Answer
    with tab2:
        st.header("Question & Answer")
        st.markdown("Ask questions about the knowledge base")

        query = st.text_input(
            "Your Question",
            placeholder=f"Example: What are the key ingredients in {config.PRODUCT_NAME}?"
        )

        if st.button("🔍 Search", type="primary"):
            if not query:
                st.error("Please enter a question")
            else:
                with st.spinner("Searching knowledge base..."):
                    try:
                        result = st.session_state.orchestrator.execute_workflow(
                            mode=WorkflowMode.QUESTION_ANSWERING.value,
                            user_input={'query': query}
                        )

                        # Display answer
                        qa_result = result['result']

                        st.subheader("Answer")
                        st.markdown(qa_result['answer'])

                        # Citations
                        if qa_result['citations']:
                            st.subheader("Sources")
                            citation_text = format_citations(qa_result['citations'])
                            st.code(citation_text, language="text")

                        # Retrieved chunks
                        with st.expander("View Retrieved Context"):
                            for i, chunk in enumerate(qa_result['retrieved_chunks'], 1):
                                st.markdown(f"**[{i}] {chunk['doc_id']} (Page {chunk['page']})** - Similarity: {chunk['similarity']:.3f}")
                                st.text(chunk['text'][:300] + "...")
                                st.divider()

                    except Exception as e:
                        st.error(f"Error: {e}")

# Footer
st.divider()
st.caption(f"{config.PRODUCT_NAME} Agentic Content System | {config.BRAND_NAME} RAG Pipeline v2.0")
st.caption("Autonomous Multi-Agent Orchestration with Enhanced PDF Processing")