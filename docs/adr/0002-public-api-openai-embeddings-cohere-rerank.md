# Public API: OpenAI-compatible embeddings (with vLLM `messages`) and Cohere/Jina-compatible rerank with {text, image} Items

The gateway exposes `POST /v1/embeddings` and `POST /v1/rerank`.

- **Embeddings** follow the OpenAI shape and accept two input forms: `input` (a string, string[], or Items `{"text"?, "image"?}`, the Jina/SGLang "object" variant) and vLLM's `messages` (one conversation, or a 2-D batch of conversations). `messages` is required because WeKnora's `generic` provider automatically sends text and images for any `*qwen3-vl-embedding*` model as vLLM `messages` (`text_as_messages`, `batch_messages`). A `system` message, when present, is used as the Instruction; otherwise the Served Model's Instruction for the Input Role applies.
- **Rerank** follows the Cohere/Jina shape: `query` and `documents` are strings or Items, and the response is `results[].index/relevance_score`. This matches WeKnora's `cohere-rerank` path, including its Jina-style `{"image": data URI}` documents.

Other vendor-native schemas (DashScope, Ark, Gemini, NIM, TEI) are deliberately not supported. They can be added later as extra request adapters without changing this contract.

## Consequences

- WeKnora tells the gateway which Input Role an input has only if the operator adds `{"input_type_field": "input_type"}` to the row's compat. Without it, every WeKnora input is treated as a Document.
- Vendor knobs WeKnora sends (`truncate_prompt_tokens`, `add_special_tokens`, `truncate`) are accepted and ignored. Over-long input is rejected with 400.
