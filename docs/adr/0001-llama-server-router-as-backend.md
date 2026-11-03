# llama-server router mode is the Backend; the gateway is a stateless translator

We run a single `llama-server` in router mode with a `models-preset.ini` (one preset per Checkpoint, `load-on-startup`, `models-max` ≥ number of presets so nothing is evicted) instead of having the gateway spawn and supervise one `llama-server` per model. llama.cpp already provides routing on `model`, lifecycle and health, so the gateway only owns protocol translation and Prompt Formats. Calling the router directly from clients was rejected because `/v1/embeddings` applies no Instruction/chat wrapping and does not accept the `{"image": ...}` Item shape WeKnora sends.

## Consequences

- Per-Checkpoint llama-server flags (pooling, device, ctx) live in the preset file, not in gateway code.
- By default the gateway starts the router as a child process (one `uv run gateway` command); it can instead point at an externally managed router.
- If router mode proves immature, the fallback is gateway-managed processes behind the same Backend interface.
