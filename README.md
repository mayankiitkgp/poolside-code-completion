# Context-Aware Code Completion via Retrieval-Augmented Generation

**A 21-day empirical investigation of hybrid retrieval (BM25 + vector embeddings) for codebase-aware code generation.**

**Author:** Mayank Mahawar ([LinkedIn](https://www.linkedin.com/in/mayank-mahawar-46032a4b/))  
**Duration:** Nov 17 - Dec 7, 2025  
**Inspiration:** [Poolside AI](https://poolside.ai)'s approach to private codebase intelligence

---

## Research Question

**Can hybrid retrieval over codebases (keyword + semantic search) improve code completion accuracy by 25%+ compared to zero-shot LLM prompting, while maintaining sub-$0.01 cost per completion?**

### Hypothesis

Generic code LLMs (GitHub Copilot, GPT-4) complete based on internet-scale patterns but lack project-specific context. Retrieval-Augmented Generation (RAG) addresses this by:
1. Indexing codebase functions/classes with semantic embeddings
2. Retrieving top-K relevant snippets for a given partial code input
3. Injecting context into LLM prompts to guide completion

Expected outcome: 80%+ functional accuracy (completion compiles and matches intent) at <$0.01/request.

---

## Architecture

┌─────────────────┐
│ Python Codebase │
└────────┬────────┘
│ AST parse (ast module)
▼
┌─────────────────────────┐
│ Function Chunks │
│ (with docstrings + sig) │
└────────┬────────────────┘
│ Embed (text-embedding-3-small)
▼
┌──────────────────┐
│ pgvector (1536D) │
└────────┬─────────┘
│
┌────┴────┐
│ Query │
└────┬────┘
│
┌────▼────────────────┐
│ Hybrid Retrieval: │
│ 1. BM25 (keyword) │
│ 2. Cosine (semantic)│
│ → Top-5 context │
└────┬────────────────┘
│
┌────▼──────────────────┐
│ Prompt Engineering: │
│ Context + Query │
└────┬──────────────────┘
│
┌────▼────────────────┐
│ Qwen 2.5 Coder 32B │
│ (via Groq API) │
└────┬────────────────┘
│
┌────▼────────────────┐
│ Completion │
│ + Source Citations │
└─────────────────────┘


---

## Methodology

### Phase 1: Indexing (Days 1-3)
1. **Corpus Selection:** FastAPI framework (~50K LoC, 1,200 functions) - chosen for clear separation of concerns, extensive docstrings
2. **Chunking Strategy:** AST-based function extraction (maintains syntactic boundaries)
3. **Embedding Model:** OpenAI `text-embedding-3-small` (1536D, $0.02/1M tokens)
   - **Alternative to test (Week 2):** Microsoft CodeBERT (`microsoft/codebert-base`) - pretrained on code-specific corpus
4. **Storage:** PostgreSQL + pgvector extension
   - **Scale consideration:** 1 repo = ~10K vectors (pgvector sufficient). Production (10M+ vectors) would require Qdrant or Weaviate.

### Phase 2: Retrieval (Days 4-5)
1. **Query Processing:** Given partial code snippet, extract semantic intent + keyword tokens
2. **Hybrid Search:**
   - BM25 (via `rank_bm25` library): Score top-20 candidates on keyword overlap
   - Vector similarity: Rerank top-20 with cosine similarity (L2-normalized)
   - Return top-5 (configurable K)
3. **Baseline Comparison:** Pure vector search vs hybrid vs BM25-only

### Phase 3: Completion (Days 6-10)
1. **Model:** Qwen 2.5 Coder 32B (65.9% on HumanEval vs GPT-4's 67%)
2. **Inference:** Groq API (400+ tokens/sec on LPU hardware vs ~50 tok/sec on GPU)
3. **Prompt Template:**

Context (retrieved from codebase):
[Function 1: def foo(...): ...]
[Function 2: def bar(...): ...]

Task: Complete the following Python code:
{partial_code}

Completion:

4. **Citation:** Return source function names that contributed to completion

### Phase 4: Evaluation (Days 11-21)
**Metrics:**
1. **Functional Correctness:** Does completion parse (AST validation)?
2. **Semantic Match:** Manual eval - does it solve intended task? (n=50 samples)
3. **Retrieval Quality:** Precision@5 (are retrieved functions relevant?)
4. **Cost:** Embedding + LLM inference per completion
5. **Latency:** End-to-end time (target: <2 sec)

**Benchmark:** HumanEval subset (164 problems) - compare:
- Zero-shot Qwen (no context)
- RAG-enhanced Qwen (this work)
- GPT-4 zero-shot (reference)

**Target:** Match GPT-4 accuracy (67%) at 10x lower cost ($0.008 vs $0.08/completion)

---

## Technical Specifications

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Embedding Model** | text-embedding-3-small (1536D) | Cost: $0.02/1M tokens (5x cheaper than ada-002). Quality: [0.48 MTEB score](https://platform.openai.com/docs/guides/embeddings). Will compare to CodeBERT (768D, code-pretrained). |
| **Vector Database** | pgvector (Postgres extension) | Zero infra setup, SQL-native. HNSW index for <1M vectors. Limitations: ANN recall degrades at 10M+ scale → would migrate to Qdrant (gRPC streaming) for production. |
| **Retrieval** | Hybrid (BM25 + cosine) | BM25 handles exact API name matches (keyword precision), cosine captures semantic similarity. Hypothesis: hybrid > pure vector for code (test in Phase 2). |
| **LLM** | Qwen 2.5 Coder 32B | Open weights, 65.9% HumanEval (SOTA among open models). Groq API: $0.27/1M tokens, 400 tok/sec (vs $15/1M + 50 tok/sec on AWS g5.xlarge). |
| **Serving** | Groq LPU API | Custom silicon for LLM inference - 8x faster than GPU at lower cost. Avoids infrastructure management. |
| **UI** | Streamlit | Rapid prototyping. Production: VSCode extension or REST API. |

---

## Relation to Poolside's Approach

### Poolside's RLCEF (Reinforcement Learning from Code Execution Feedback)
Poolside uses [code execution as a reward signal](https://poolside.ai/designing-a-world-class-code-execution-environment):
1. Generate completion → run tests → measure pass rate → update model weights
2. Advantages: Ground truth from execution (no manual labels), catches runtime errors (not just syntax)
3. Challenges: Requires sandboxed execution environment, test suite per repo

### This Work: RAG-Based Approach
**What's similar:**
- Both inject codebase-specific context (Poolside: execution traces, this work: retrieved functions)
- Both optimize for private codebases (not internet-scale data)

**What's different:**
- **No execution feedback loop** (out of scope for 21 days) - RAG is static context injection
- **No model fine-tuning** - using off-the-shelf Qwen + prompt engineering
- **Simpler evaluation** - manual correctness checks, not automated test pass rates

**Future work:** Combine RAG retrieval with execution feedback - use retrieved tests as part of prompt, run completions in sandbox, refine based on pass/fail.

---

## Progress Tracker

**Week 1: Core Pipeline**
- [x] Nov 17 (Day 1): Architecture design, repo setup
- [ ] Nov 18 (Day 2): AST-based code crawler, chunk extraction
- [ ] Nov 19 (Day 3): Embedding generation, pgvector setup
- [ ] Nov 20 (Day 4): BM25 indexing with `rank_bm25`
- [ ] Nov 21 (Day 5): Hybrid retrieval pipeline, quality eval
- [ ] Nov 22 (Day 6): Qwen integration via Groq API
- [ ] Nov 23 (Day 7): Prompt engineering, context injection

**Week 2: UI + Demo**
- [ ] Nov 24-26: Streamlit interface, user testing
- [ ] Nov 27-28: Benchmarking on HumanEval subset
- [ ] Nov 29-30: Cost analysis, latency profiling

**Week 3: Iteration + Outreach**
- [ ] Dec 1-3: CodeBERT comparison, hybrid search tuning
- [ ] Dec 4-5: Documentation, demo video
- [ ] Dec 6-7: Final evaluation, publish results

---

## Preliminary Results

_To be updated weekly. Target: 25%+ accuracy gain over zero-shot, <$0.01/completion._

| Metric | Baseline (Zero-shot) | RAG-Enhanced | GPT-4 Reference |
|--------|----------------------|--------------|-----------------|
| HumanEval Accuracy | TBD (Week 2) | TBD | 67% |
| Avg Latency (sec) | TBD | TBD | ~3s |
| Cost per Completion | $0.004 | TBD | $0.08 |
| Retrieval Precision@5 | N/A | TBD | N/A |

---

## Dependencies

See [`requirements.txt`](requirements.txt) for full list. Key libraries:
- `openai` (embeddings)
- `groq` (LLM inference)
- `pgvector` (vector storage)
- `rank-bm25` (keyword search)
- `streamlit` (UI)
- `pytest` (testing)

---

## Reproducibility

**Setup:**
Clone repo
git clone https://github.com/mayankiitkgp/poolside-code-completion
cd poolside-code-completion

Install dependencies
pip install -r requirements.txt

Configure API keys
cp .env.example .env

Add OPENAI_API_KEY and GROQ_API_KEY to .env
Run indexing
python src/crawler/index_repo.py --repo /path/to/fastapi

Start demo
streamlit run src/ui/app.py


**Estimated cost:** <$5 for 21-day experiment (500 completions, 50K embeddings)

---

## References

1. [Poolside: Designing a World-Class Code Execution Environment](https://poolside.ai/designing-a-world-class-code-execution-environment)
2. [Qwen 2.5 Coder Technical Report](https://qwenlm.github.io/blog/qwen2.5-coder/)
3. [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155)
4. [HumanEval: Hand-Written Evaluation Set for Code Generation](https://github.com/openai/human-eval)
5. [Groq LPU Inference Whitepaper](https://groq.com/)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Contact:** For questions or collaboration, open an issue or reach out via [LinkedIn](https://www.linkedin.com/in/mayank-mahawar-46032a4b/).
