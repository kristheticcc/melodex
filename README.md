# Melodex 🎶
### A Multi-Phase Music Knowledge RAG System

Melodex is a conversational AI assistant built over a curated music knowledge base covering K-pop and indie artists. The project was developed across four phases, progressively improving retrieval quality and answer accuracy through increasingly sophisticated RAG techniques - with quantitative evaluation at each stage to measure real improvement.

---

## Demo

Ask Melodex about artists, albums, members, release dates, or comparisons:
- *"Who is Hanni from NewJeans?"*
- *"What are some songs by Chase Atlantic?"*
- *"Who are the members of TXT?"*

---

## Architecture

### Phase 1 — Naive RAG
Keyword-based dictionary lookup. Artist and album markdown files are loaded into a Python dictionary keyed by filename. If a keyword from the user's message matches a key, that file's content becomes the context. Simple and fast, but fails entirely on member-specific, conceptual, or multi-document queries.

### Phase 2 — LangChain RAG (Baseline)
Full vector retrieval pipeline using HuggingFace sentence embeddings (`all-MiniLM-L6-v2`), ChromaDB vector store, and LangChain's `RecursiveCharacterTextSplitter` (chunk size 1000, overlap 200). Semantic search replaces keyword matching - the system can now retrieve relevant content even when exact keywords aren't present.

### Phase 3 — Evaluation Framework
Formal evaluation pipeline using 30 hand-crafted test questions across 7 categories: `direct_fact`, `temporal`, `member_specific`, `comparative`, `spanning`, `numerical`, and `holistic`. Two evaluation dimensions:
- **Retrieval:** MRR (Mean Reciprocal Rank) and nDCG (Normalized Discounted Cumulative Gain)
- **Answer Quality:** LLM-as-judge scoring for accuracy, completeness, and relevance using structured Pydantic outputs via LiteLLM

### Phase 4 — Advanced RAG
Three improvements stacked on top of Phase 2:
1. **MultiQueryRetriever** - LLM generates 3 query variants + original, retrieves docs for all, merges and deduplicates
2. **CrossEncoder Reranking** - `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each retrieved chunk against the original question and keeps top 5
3. **Expanded Knowledge Base** - artist files enriched with per-member sections, musical style descriptions, and detailed biographies

---

## Evaluation Results

### Retrieval Quality

| Metric | Phase 2 (Baseline) | Phase 4 (Advanced) | Improvement |
|--------|-------------------|-------------------|-------------|
| MRR | 0.407 | 0.679 | **+67%** |
| nDCG | 0.491 | 0.722 | **+47%** |

### Answer Quality (LLM-as-Judge, 1–5 scale)

| Metric | Phase 2 (Baseline) | Phase 4 (Advanced) | Improvement |
|--------|-------------------|-------------------|-------------|
| Accuracy | 4.133 | 4.500 | **+8.9%** |
| Completeness | 3.933 | 4.200 | **+6.8%** |
| Relevance | 4.200 | 4.200 | 0% |

### Key Findings
- **Content quality matters most on small knowledge bases.** Expanding artist files with per-member sections had a larger impact on answer quality than pipeline changes alone.
- **MultiQueryRetriever significantly improved retrieval.** Generating multiple query variants increased coverage, especially for member-specific and spanning questions.
- **CrossEncoder reranking improved ranking precision.** MRR improvement of +67% indicates relevant chunks moved to the top positions after reranking.
- **Relevance plateaued at 4.2.** The LLM tends to add contextual detail beyond what's strictly asked, which caps relevance scores independent of retrieval quality.

---

## Knowledge Base

```
knowledge_base/
├── artists/
│   ├── newjeans.md
│   ├── twice.md
│   ├── txt.md
│   ├── keshi.md
│   └── chase_atlantic.md
└── albums/
    ├── omg.md
    ├── ready_to_be.md
    ├── temptation.md
    ├── guts.md
    └── phases.md
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq (OSS 120B) |
| Query Rewriting | Groq (Llama 3.1 8B Instant) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| Evaluation | LiteLLM + Pydantic |
| UI | Gradio |

---

## Setup

```bash
git clone https://github.com/kristheticcc/melodex
cd melodex
uv sync
```

Add your Groq API key to `.env`:
```
GROQ_API_KEY=your_key_here
```

Build the vector database:
```bash
python ingest.py
```

Run the chatbot:
```bash
python app.py
```

Run the evaluation dashboard:
```bash
python eval_app.py
```

---

## Project Structure

```
melodex/
├── knowledge_base/     # Markdown files (artists + albums)
├── phase1_*.py         # Naive RAG implementation
├── config.py           # Models, embeddings, vector store
├── ingest.py           # Document loading, chunking, vector DB creation
├── rag.py              # Phase 2 retrieval pipeline
├── advanced_rag.py     # Phase 4 MultiQuery + CrossEncoder pipeline
├── app.py              # Gradio chatbot UI
├── tests.py            # Test question loader
├── tests.jsonl         # 30 evaluation questions
├── eval.py             # MRR, nDCG, LLM-as-judge evaluation
├── eval_app.py         # Gradio evaluation dashboard
├── results.md          # Full evaluation results
└── screenshots/        # Phase 2 and Phase 4 evaluation dashboard (plus additional) screenshots
```