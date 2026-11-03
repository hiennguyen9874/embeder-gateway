# The gateway renders rerank prompts itself instead of using llama.cpp `/v1/rerank`

The Instruction of a reranking Served Model must be configurable, but llama.cpp `/v1/rerank` always uses the rerank template baked into the GGUF (fixed Instruction, and it differs from the official Qwen default). So for every rerank request the gateway renders the full query–document prompt with the Served Model's Prompt Format, sends one Item per document to the Backend's `/v1/embeddings` (rank pooling, `embd_normalize: -1`), reads `P("yes")` as `relevance_score`, then sorts and applies `top_n` itself. We always take this path, even with the default Instruction, so there is a single code path whose correctness is pinned by a parity test against the official transformers implementation.

## Consequences

- The GGUF's `tokenizer.chat_template.rerank` is unused; the Prompt Format must reproduce the official prompt exactly, including image placement.
- Backend presets for rerankers still need `--reranking --pooling rank`.
