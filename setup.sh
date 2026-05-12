#!/bin/bash

# Nalpamaradi Keram Agentic System - Setup Script
# Python Version Required: 3.10+

echo "================================================"
echo "Nalpamaradi Keram Agentic Content System Setup"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python $required_version or higher is required"
    echo "   Current version: $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to create virtual environment"
    exit 1
fi

echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Failed to upgrade pip, continuing anyway..."
else
    echo "✅ Pip upgraded"
fi
echo ""

# Install requirements
echo "Installing dependencies..."
echo "(This may take several minutes...)"
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install dependencies"
    echo "   Please check requirements.txt and try again"
    exit 1
fi

echo "✅ All dependencies installed"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p pdfs
mkdir -p outputs

echo "✅ Directories created"
echo ""

# Create .env template if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env template..."
    cat > .env << EOF
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Optional: Model Configuration
# LLM_MODEL=llama-3.1-70b-versatile
# EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EOF
    echo "✅ .env template created"
    echo "   ⚠️  Please edit .env and add your Groq API key"
else
    echo "ℹ️  .env file already exists"
fi
echo ""

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo "Creating .gitignore..."
    cat > .gitignore << EOF
# Virtual environment
venv/
env/

# API keys
.env

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# Temporary files
temp_*.pdf
*.tmp

# ChromaDB
chroma_data/

# Outputs
outputs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF
    echo "✅ .gitignore created"
else
    echo "ℹ️  .gitignore file already exists"
fi
echo ""

# Create README if it doesn't exist
if [ ! -f README.md ]; then
    echo "Creating README.md..."
    cat > README.md << 'EOF'
# Nalpamaradi Keram Agentic Content System

AI-powered content generation system for Ayurvedic products using RAG and multi-agent architecture.

## Quick Start

1. Add your Groq API key to `.env`:
   ```
   GROQ_API_KEY=your_actual_key_here
   ```

2. Place Ayurvedic PDF documents in the `pdfs/` folder

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Open browser to `http://localhost:8501`

## Documentation

See `SYSTEM_DOCUMENTATION.md` for complete details.

## Python Version

Requires Python 3.10 or higher.

## Key Features

- RAG-based content generation
- 4-agent workflow (Outline → Writer → Fact-Checker → Finalizer)
- Medical claim detection
- Citation tracking
- Streamlit web interface

EOF
    echo "✅ README.md created"
else
    echo "ℹ️  README.md file already exists"
fi
echo ""

# Test imports
echo "Testing imports..."
python3 << 'PYTHON_TEST'
try:
    import streamlit
    import chromadb
    import sentence_transformers
    import groq
    import PyPDF2
    print("✅ All critical imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
PYTHON_TEST

if [ $? -ne 0 ]; then
    echo "❌ Error: Import test failed"
    exit 1
fi
echo ""

# Final instructions
echo "================================================"
echo "Setup Complete! 🎉"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Edit .env file and add your Groq API key:"
echo "   nano .env"
echo ""
echo "2. Place PDF documents in the pdfs/ folder:"
echo "   cp /path/to/charaka_samhita.pdf pdfs/"
echo ""
echo "3. Run the application:"
echo "   streamlit run app.py"
echo ""
echo "4. Open http://localhost:8501 in your browser"
echo ""
echo "For detailed documentation, see:"
echo "   SYSTEM_DOCUMENTATION.md"
echo ""
echo "================================================"