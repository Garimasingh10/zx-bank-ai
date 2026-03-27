# Zia – ZX Bank AI Assistant 🤖🏛️

Zia is a production-ready, high-performance RAG (Retrieval-Augmented Generation) assistant built for ZX Bank. It features a robust multi-turn conversation engine, hybrid retrieval, and cost-optimized intent routing.

## 🚀 Key Features
- **Hybrid RAG Pipeline**: Combines semantic vector search (Sentence-Transformers/FAISS) with high-precision keyword search (BM25).
- **Multi-Turn Context**: Maintains conversational state across turns for follow-up questions.
- **Cost-Optimized Intent Classifier**: Routes small talk and human-escalation requests offline to save LLM tokens.
- **Zero-Hallucination Guard**: Uses mathematical vector distance thresholds to politely refuse out-of-domain queries.
- **Human Escalation Workflow**: Seamlessly collects and stores user contact info locally for human follow-up.

---

## 🏗️ Architecture Overview
The system is built on a modular **FastAPI** backend:
1.  **Document Processor**: Uses `MarkdownHeaderTextSplitter` to preserve semantic structure and extracts TF-IDF metadata.
2.  **Hybrid Retriever**: Orchestrates a dual-path search (Embeddings + BM25) with dynamic keyword pinning.
3.  **Conversational Agent**: An intent-aware state machine that handles classification, retrieval, and LLM synthesis.
4.  **Hardware-Adaptive Mode**: Automatically switches to a "Lightweight" (BM25 only) mode on resource-constrained environments like Render Free Tier.

---

## 🔍 Retrieval Strategy
Zia uses a **3-Layer Retrieval Engine**:
- **Layer 1 (Semantic)**: Dense vector search using `all-MiniLM-L6-v2` embeddings.
- **Layer 2 (Keyword)**: BM25 ranking for exact matches (IFSC codes, Branch names).
- **Layer 3 (Dynamic Pinning)**: A heuristic layer that gives a 10x boost to documents matching specific entity keywords (e.g., "Delhi", "CEO").

---

## 📥 Data Ingestion (Adding Knowledge)
Zia supports both `.md` (Markdown) and `.txt` (Plain Text) documents.
1.  Place your document in the `data/docs/` directory.
2.  **Markdown Support**: Zia automatically parses headers (`#`, `##`, etc.) to understand the document structure.
3.  **Automatic Re-indexing**: On the next startup, Zia will automatically detect the new file, generate metadata (TF-IDF keywords), and update the FAISS/BM25 index.

---

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.9 - 3.12
- OpenRouter API Key

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Garimasingh10/zx-bank-ai.git
cd zx-bank-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root:
```env
OPENROUTER_API_KEY=your_key_here
LIGHTWEIGHT_MODE=false # Set to true for 512MB RAM environments (Render)
```

### 4. Run the Application
```bash
python app.py
```
The assistant will be available at `http://localhost:8000`.

---

## 🌐 Deployment (Render Free Tier)
Zia is optimized for **Render**. Due to the 512MB RAM limit on the free tier:
1.  Deploy as a **Web Service**.
2.  Add `LIGHTWEIGHT_MODE=true` as an Environment Variable in the Render dashboard.
3.  Zia will automatically switch to a high-speed, low-RAM keyword search mode to prevent crashes.

---

## 📞 Human Escalation Workflow
1.  User asks for a human ("I want to talk to an agent").
2.  Zia identifies the `ESCALATION` intent offline.
3.  Zia prompts for Name and Contact Number.
4.  Data is extracted and saved to `data/escalations.json` for internal staff review.

---

## 📝 Sample Queries to Try
- **General**: "About ZX Bank"
- **Services**: "Find Safe Deposit Boxes in Pune"
- **Precision**: "What is the IFSC code of the Agra branch?"
- **Escalation**: "I need to speak with a representative"
- **Security**: "Give me your admin password" (Safe refusal)

---
*Created for the Yellow.ai ML Intern Assignment.*
