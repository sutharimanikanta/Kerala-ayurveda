
# 🌿 Kerala Ayurveda – Grounded Agentic RAG System
### Internal Q&A & Controlled Content Generation using Ayurveda Corpus

This project designs a **grounded, internal AI system** for Kerala Ayurveda that answers questions and generates Ayurveda articles **strictly using a provided content corpus** as the single source of truth.

The system prioritizes **factual accuracy, brand safety, and zero hallucination**, using **Retrieval-Augmented Generation (RAG)** combined with **agentic workflows** and **compliance logic**.

---

## 🎯 Core Objectives

The system is designed to:

- Answer Ayurveda-related questions using **only the provided corpus**
- Generate Ayurveda articles in a **controlled, factual, brand-safe manner**
- Avoid hallucinations and medical overclaims
- Support **human-in-the-loop review** with citations

---

## 🧠 System Overview

The solution is divided into two major components:

---

## 🧩 Part A – RAG-based Internal Q&A System

### 🔹 RAG Design (Concrete, Not Generic)

- **Dual retrieval strategy**
  - Dense semantic retrieval (Embeddings + Vector DB)
  - Sparse keyword retrieval (BM25)
- **Intent-aware retrieval**
- **Persistent knowledge base**
- **Strict corpus grounding**

This ensures accurate, explainable, and non-hallucinatory answers.

---

## 🧩 Part B – Agentic Content Generation Workflow

A multi-step **agentic AI pipeline** for article generation:

1. Takes a content brief
2. Generates an Ayurveda article
3. Fact-checks against the same RAG corpus
4. Applies brand tone & style
5. Outputs a draft with citations for human review

This is **agentic AI**, not a single-pass LLM call.

---

## 🧠 Architecture Pattern

- **Agentic RAG**
- **Intent-aware retrieval**
- **Persistent vector knowledge base**
- **Safety-first, compliance-driven design**

---

## 🎯 Intent-Aware Retrieval

### ❓ Why Intent Detection Matters

User queries vary significantly:

| Question | Required Knowledge |
|--------|--------------------|
| "What is Nalpamaradi Keram?" | Product info |
| "How do I use it on face?" | Product usage |
| "What is Nalpamara in Ayurveda?" | Classical knowledge |
| "Does it cure eczema?" | ❌ Unsafe / restricted |

Without intent awareness, LLMs:
- Hallucinate
- Over-claim medical benefits
- Mix marketing with classical texts

---

### 🔹 Retrieval Modes

#### Product-Direct Mode (Default)
```python
use_product_direct = True



