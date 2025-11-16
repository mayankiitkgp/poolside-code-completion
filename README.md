# Context-Aware Code Completion via Retrieval-Augmented Generation

**Empirical investigation: Can hybrid retrieval (BM25 + vector embeddings) improve code completion accuracy by 25%+ vs zero-shot LLM prompting?**

**Author:** Mayank Mahawar ([LinkedIn](https://www.linkedin.com/in/mayank-mahawar-46032a4b/))  
**Timeline:** Nov 17 - Dec 7, 2025 (21 days)  
**Inspiration:** [Poolside AI](https://poolside.ai)'s approach to private codebase intelligence

---

## Research Question

**Hypothesis:** Retrieval-Augmented Generation (RAG) with codebase-specific context can boost code completion accuracy to 80%+ at <$0.01/completion, outperforming zero-shot LLM prompting by 25%.

**Target benchmark:** Match GPT-4's 67% on HumanEval using Qwen 2.5 Coder (32B) + retrieved context, at 10x lower cost.

---

## Architecture

flowchart TD
A[Python Codebase
FastAPI ~50K LoC] -->|AST Parse| B[Function Chunks
+ docstrings + signatures]
B -->|Embed via OpenAI API| C[pgvector Database
1536D vectors]
D[User Query
Partial Code Snippet] -->|Extract tokens| E[Hybrid Retrieval]
C --> E
E -->|1. BM25 keyword scoring| F[Top-20 Candidates]
F -->|2. Cosine similarity rerank| G[Top-5 Context Functions]
G -->|Inject into prompt| H[Qwen 2.5 Coder 32B
via Groq API]
H --> I[Completion + Citations]

style A fill:#e1f5fe
style C fill:#fff3e0
style H fill:#f3e5f5
style I fill:#e8f5e9


**Pipeline:**
1. **Index:** AST-parse codebase → embed functions with `text-embedding-3-small` (1536D) → store in pgvector
2. **Retrieve:** Hybrid search (BM25 keyword + cosine semantic) → top-5 context
3. **Complete:** Inject context into Qwen 2.5 Coder prompt → generate completion + citations

---

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Embeddings** | text-embedding-3-small | $0.02/1M tokens (5x cheaper than ada-002), 1536D standard |
| **Vector DB** | pgvector (Postgres) | Zero setup, HNSW index for <1M vectors. Production: Qdrant/Weaviate |
| **Retrieval** | BM25 + cosine hybrid | BM25 for exact matches, cosine for semantics. Test vs pure vector. |
| **LLM** | Qwen 2.5 Coder 32B | 65.9% HumanEval, $0.27/1M tokens via Groq API (400 tok/sec) |
| **UI** | Streamlit | Fast prototyping. Production: VSCode extension or REST API |

**Estimated cost:** <$5 for 21-day experiment (500 completions, 50K embeddings)

---

## Week 1 Progress

- [x] **Day 1 (Nov 17):** Architecture design, repo setup
- [ ] **Day 2 (Nov 18):** AST-based code crawler, chunk extraction
- [ ] **Day 3 (Nov 19):** Embedding generation, pgvector setup
- [ ] **Day 4 (Nov 20):** BM25 indexing with `rank_bm25`
- [ ] **Day 5 (Nov 21):** Hybrid retrieval pipeline
- [ ] **Day 6 (Nov 22):** Qwen integration via Groq API
- [ ] **Day 7 (Nov 23):** Prompt engineering, initial completions

_Week 2-3 progress will be added as work proceeds._

---

## Relation to Poolside's RLCEF

Poolside uses Reinforcement Learning from Code Execution Feedback: generate code → run tests → update model based on pass/fail.

**This work:** RAG-based context injection (no execution loop, no fine-tuning). Faster to prototype, lower infrastructure needs. Trade-off: No automatic error correction from runtime feedback.

**Future direction:** Combine RAG retrieval + execution feedback (Week 3 analysis).

---

## Quick Start

Clone repo
git clone https://github.com/mayankiitkgp/poolside-code-completion
cd poolside-code-completion

Install dependencies
pip install -r requirements.txt

Configure API keys
cp .env.example .env

Add OPENAI_API_KEY and GROQ_API_KEY
Index a codebase (coming Day 2)
python src/crawler/index_repo.py --repo /path/to/codebase

Run demo (coming Week 2)
streamlit run src/ui/app.py


---

## References

1. Poolside: Code Execution Environment
2. [Qwen 2.5 Coder Report](https://qwenlm.github.io/blog/qwen2.5-coder/)
3. [HumanEval Benchmark](https://github.com/openai/human-eval)

---

## License

MIT License - see [LICENSE](LICENSE).

**Contact:** [LinkedIn](https://www.linkedin.com/in/mayank-mahawar-46032a4b/) or open an issue.
