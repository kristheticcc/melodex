# Melodex - Evaluation Results

## Phase 1 - Naive RAG (Keyword-Based)
Dictionary lookup using filename stems as keys. If the user message contains a keyword matching a filename, that file's content is injected as context. No vector embeddings, no semantic search.

**Limitations:** Fails on member-specific queries, conceptual questions, and anything requiring semantic understanding. Only works when the exact artist/album name appears in the message.

---

## Phase 2 - LangChain RAG (Baseline)
HuggingFace sentence embeddings (`all-MiniLM-L6-v2`), ChromaDB vector store, RecursiveCharacterTextSplitter (chunk size 1000, overlap 200), default retriever (k=4). LLM: Groq OSS 120B.

| Metric | Score |
|--------|-------|
| MRR | 0.407 |
| nDCG | 0.491 |
| Accuracy | 4.133 / 5 |
| Completeness | 3.933 / 5 |
| Relevance | 4.200 / 5 |

---

## Phase 3 - Evaluation Framework
Formal evaluation using 30 test questions across 7 categories: `direct_fact`, `temporal`, `member_specific`, `comparative`, `spanning`, `numerical`, `holistic`. Retrieval scored with MRR and nDCG. Answer quality scored with LLM-as-judge (llama-3.3-70b-versatile) using structured Pydantic outputs for accuracy, completeness, and relevance.

---

## Phase 4 - Advanced RAG
MultiQueryRetriever (llama-3.1-8b-instant generates 3 query variants + original), CrossEncoderReranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`, top_n=5), expanded knowledge base with per-member sections, musical style descriptions, and detailed album content.

| Metric | Score |
|--------|-------|
| MRR | 0.679 |
| nDCG | 0.722 |
| Accuracy | 4.500 / 5 |
| Completeness | 4.200 / 5 |
| Relevance | 4.200 / 5 |

---

## Phase 2 → Phase 4 Improvement

| Metric | Phase 2 | Phase 4 | Improvement |
|--------|---------|---------|-------------|
| MRR | 0.407 | 0.679 | +67% |
| nDCG | 0.491 | 0.722 | +47% |
| Accuracy | 4.133 | 4.500 | +8.9% |
| Completeness | 3.933 | 4.200 | +6.8% |
| Relevance | 4.200 | 4.200 | 0% |

---

## Key Findings

- **Content quality matters most on small knowledge bases.** Expanding artist files with per-member sections and musical style descriptions had a larger impact on answer quality than retrieval strategy alone.
- **MultiQueryRetriever significantly improved retrieval.** Generating multiple query variants increased the chance of hitting relevant chunks, especially for member-specific and spanning questions.
- **CrossEncoder reranking improved ranking quality.** MRR improvement (+67%) indicates the most relevant chunks moved to the top after reranking, giving the LLM better context to work with.
- **Relevance plateaued.** The LLM tends to add contextual information beyond what's strictly asked, which caps relevance scores regardless of retrieval quality.