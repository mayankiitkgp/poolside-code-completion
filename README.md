# Poolside-Inspired Code Completion

**Context-aware Python code completion using RAG + Qwen 2.5 Coder**

Inspired by [Poolside](https://poolside.ai)'s approach to code generation, this project demonstrates how retrieval-augmented generation (RAG) improves LLM completions by providing relevant codebase context.

**Built by:** Mayank Mahawar | [LinkedIn](https://www.linkedin.com/in/mayank-mahawar-46032a4b/)

---

##  Goal

Explore whether semantic search over codebases (vector embeddings + BM25 hybrid retrieval) can boost code completion accuracy beyond generic LLM prompting.

## Architecture

[Code Repository]
↓ (AST-based chunking)
[Function Embeddings] → [pgvector]
↓ (Hybrid Search: BM25 + Cosine)
[Retrieved Context (Top-5)]
↓ (Injected into prompt)
[Qwen 2.5 Coder LLM] → [Completion + Citations]


## Current Status

- [x] Day 1: Repo setup, architecture design
- [ ] Day 2: Code crawler + AST chunking
- [ ] Day 3: Embedding generation (text-embedding-3-small)
- [ ] Day 4-5: RAG retrieval pipeline
- [ ] Day 6-7: LLM integration (Qwen via Groq API)
- [ ] Day 8-10: Streamlit UI + deployment
- [ ] Day 11+: Benchmarking, optimization

**Follow along:** I'm building in public over the next 3 weeks. Check back for updates!

## Tech Stack

- **Embeddings:** OpenAI text-embedding-3-small ($0.02/1M tokens)
- **Vector DB:** PostgreSQL + pgvector
- **LLM:** Qwen 2.5 Coder 32B (via Groq API, $0.27/1M tokens)
- **Retrieval:** Hybrid BM25 (keyword) + cosine similarity (semantic)
- **UI:** Streamlit

## Demo

_Coming soon (Week 2) - live Streamlit app TBA_

## License

MIT - feel free to fork and build on this!

## Acknowledgments

Inspired by:
- Poolside's blog on RLCEF
- semantic search work
- Research on RAG for code

---

**Questions or feedback?** Open an issue or DM me on [LinkedIn](https://www.linkedin.com/in/mayank-mahawar-46032a4b/).



