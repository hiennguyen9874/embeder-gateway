# Embedder Gateway

A Python gateway that serves multimodal (text + image) embedding and reranking models from llama.cpp (router mode) behind OpenAI/Cohere-compatible APIs, mainly for WeKnora. The spec is GitHub issue #1 and the tickets are its sub-issues. The architecture is in `docs/adr/`.

## Stack

Python, managed with uv. FastAPI + httpx; pytest.

## Testing

- Default suite: drive the gateway's public HTTP API in-process with a fake Backend (httpx `MockTransport`). Assert on the responses and on the exact requests sent to the Backend.
- Tests that need a real llama-server or GPU are marked `gpu` and are opt-in. These are the smoke tests and the parity tests against the official Qwen transformers scripts.
- The GPUs are shared with other workloads. Pick an idle one with `nvidia-smi` and pin to it.

## Reference material (read-only)

- `llama.cpp/`: vendored checkout with prebuilt binaries (`llama-server`, `llama-mtmd-cli`, …). It is the source of truth for Backend behaviour: `tools/server/server-context.cpp`, `tools/server/server-common.cpp` and `tools/server/README.md` (router mode, presets). Its own `AGENTS.md` covers upstream contributions and does not apply here.
- `references/WeKnora/`: the source of truth for what WeKnora sends. See `internal/models/api/`, `internal/models/providers/generic.go`, `internal/models/catalog/data/models.generated.json`, and `website-docs/03-features/06-models.md`. Golden and wire tests there give exact request bodies to use as fixtures.
- `docs/embedding/` and `docs/reranking/`: per-model research notes, written against an older llama.cpp (~b10472). Check every llama.cpp claim against the vendored source. For example, `/v1/rerank` is now multimodal.
- `docs/api-schema.md`: a summary of WeKnora's outbound wire formats. `references/WeKnora/` is authoritative when the two differ.

## Models

`models/` is gitignored.
- `models/Qwen3-VL-*-2B/` are Hugging Face reference checkouts holding LFS pointers only. They are useful for their README, chat templates and the official `scripts/` used by the parity tests.
- GGUF Checkpoints go in `models/<org>--<repo>/`. Fetch only the files you need: `GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/<org>/<repo>`, then `git lfs pull --include "<file>"`.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Triage uses the default five-role label vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

Domain documentation uses a single-context layout. See `docs/agents/domain.md`.
