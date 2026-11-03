# Embedder Gateway

An OpenAI/Cohere-compatible gateway for multimodal (text + image) embedding and reranking models served by llama.cpp. It is built mainly for [WeKnora](docs/weknora.md).

The gateway runs a single `llama-server` in router mode as its **Backend** ([ADR 0001](docs/adr/0001-llama-server-router-as-backend.md)) and exposes:

- `POST /v1/embeddings`: the OpenAI shape. It accepts `input` (a string, strings, or `{"text"?, "image"?}` Items) or vLLM `messages` (one conversation or a 2-D batch) ([ADR 0002](docs/adr/0002-public-api-openai-embeddings-cohere-rerank.md)).
- `POST /v1/rerank`: the Cohere/Jina shape, with `query`/`documents` given as strings or Items. The gateway renders each rerank prompt itself ([ADR 0003](docs/adr/0003-gateway-renders-rerank-prompts.md)).
- `GET /v1/models`: every Served Model, with its kind, Modalities and dimensions.
- `GET /health`: returns 200 only when every Checkpoint is loaded. It needs no API key.

Terms in **bold**, such as Served Model, Checkpoint and Prompt Format, are defined in [GLOSSARY.md](GLOSSARY.md).

## Quick start

You need:
- Linux with an NVIDIA GPU with about 14 GB of free VRAM for both default Checkpoints; lower `ctx-size` to use less
- [uv](https://docs.astral.sh/uv/)
- `git` and [`git-lfs`](https://git-lfs.com/)
- a CUDA build of `llama-server` at build 12072 or newer

### 1. Get `llama-server`

The gateway needs router mode (`--models-preset`), multimodal `/v1/embeddings`, and rank pooling. These are in llama.cpp build 12072 (`ca8daad19`), the build this repo is tested against. Build `llama-server` with CUDA by following [docs/llama-cpp-install.md](docs/llama-cpp-install.md), or download a release from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/releases).

The default location is `llama.cpp/llama-server`. To keep the binary elsewhere, set `backend.llama_server` in `gateway.yaml`. Check the binary and the GPUs it can see:

```bash
./llama.cpp/llama-server --version        # build 12072 or newer
./llama.cpp/llama-server --list-devices
```

### 2. Install and download the Checkpoints

```bash
uv sync
uv run gateway download
```

`download` fetches exactly the GGUF and mmproj files named in `gateway.yaml` into `models/<org>--<repo>/`. The defaults are a 5 GB download and take about 10 GB on disk, because git-lfs keeps a second copy under `.git/lfs`. It uses `GIT_LFS_SKIP_SMUDGE=1 git clone` and then `git lfs pull --include <files>`. Running it again is safe: Checkpoints that are already complete are skipped. The Hugging Face reference checkouts (`models/Qwen3-VL-*-2B/`) are never touched.

### 3. Run

```bash
export GATEWAY_API_KEYS="$(openssl rand -hex 24)"   # comma-separated; clients send one as a Bearer token
CUDA_VISIBLE_DEVICES=0 uv run gateway --host 0.0.0.0 --port 8000
```

On startup the gateway:
1. generates `models-preset.ini` from `gateway.yaml`;
2. starts the router on `127.0.0.1` with a random internal key;
3. waits until every Checkpoint is loaded (up to 10 minutes);
4. serves the API.

Stopping the gateway (Ctrl-C or SIGTERM) also stops the router. `CUDA_VISIBLE_DEVICES` controls which physical GPU(s) the router sees. Inside it, the first visible GPU is `CUDA0`, the default `device` in `gateway.yaml`.

`uv run gateway --help` lists the options: `serve` (default) or `download`, `--config` (default `gateway.yaml`, or `$GATEWAY_CONFIG`), `--host` (default `127.0.0.1`), and `--port` (default `8000`).

### 4. Try it

```bash
KEY="${GATEWAY_API_KEYS%%,*}"
curl -s localhost:8000/health
curl -s localhost:8000/v1/models -H "Authorization: Bearer $KEY"

curl -s localhost:8000/v1/embeddings -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d '{
  "model": "Qwen3-VL-Embedding-2B",
  "input": ["Paris is the capital of France.", {"text": "A llama", "image": "data:image/png;base64,..."}],
  "dimensions": 1024
}'

curl -s localhost:8000/v1/rerank -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d '{
  "model": "Qwen3-VL-Reranker-2B",
  "query": "What is the capital of France?",
  "documents": ["Paris is the capital of France.", "Bananas are yellow."],
  "top_n": 1, "return_documents": true
}'
```

Each embedding input has an **Input Role**: a Query or a Document. The gateway takes it from the first of these that is present:
1. `input_type` (`query`/`document`, or `search_query`/`search_document`/`passage`);
2. Jina `task` (`*.query` / `*.passage`);
3. the Served Model's `role` (for example `Qwen3-VL-Embedding-2B-query`).

If none is present, the input is a Document. In `messages`, a `system` message replaces the **Instruction**. Images must be data URIs or raw base64; http(s) URLs and video are rejected with 400. Errors use the OpenAI shape `{"error": {"message", "type", "code"}}`.

## Configuration

`gateway.yaml` is the single source of truth. Each section is commented in the file itself.

| Section | What it sets |
| --- | --- |
| `models_dir` | Where `download` puts Checkpoints (`models/` by default, relative to the YAML file). |
| `backend` | `llama_server`: the binary the gateway starts. Set `url` to use an externally managed router instead; the gateway reads that router's key from `GATEWAY_BACKEND_API_KEY`. |
| `limits` | Per-request limits: `max_embedding_items` (64), `max_rerank_documents` (100), `max_image_mb` (10), `max_body_mb` (50). Going over one returns 400 (body size: 413). |
| `checkpoints` | One router preset per **Checkpoint**. The key is the router's model id. `repo`, `model` and `mmproj` name the Hugging Face files. `args` holds `llama-server` flags (long names without dashes, such as `pooling`, `device`, `ctx-size`, `parallel`, `image-max-tokens`). Defaults: `parallel: 4`, `ctx-size: 32768`, i.e. 8192 tokens per slot. Every preset gets `load-on-startup`, and the router runs with `--models-max` equal to the number of Checkpoints, so nothing is evicted. |
| `served_models` | The names clients put in `model`. Each one names a `checkpoint` and a Prompt Format (`format`), and sets the accepted `modalities`, optional `instructions` overrides and an optional fixed `role`. Embedding models also set `max_dimensions`, `min_dimensions` (the MRL range) and the default `dimensions`. |

API keys come from `GATEWAY_API_KEYS` (comma-separated), `GATEWAY_API_KEY_FILE` (one key per line, `#` for comments), or both. The gateway refuses to start without a key. To rotate a key, add the new key, restart, then remove the old one.

### External router

To run the router yourself, start `llama-server` with a preset equivalent to the one the gateway generates. Its model ids must match the `checkpoints` keys. Then set `backend.url`:

```bash
llama-server --models-preset models-preset.ini --models-max 2 --host 127.0.0.1 --port 8080 --api-key-file router.key
GATEWAY_BACKEND_API_KEY="$(cat router.key)" uv run gateway   # with backend.url: http://127.0.0.1:8080
```

The `render_preset` function in `src/gateway/presets.py` produces the preset text from `gateway.yaml`:

```bash
uv run python -c 'from gateway.config import load_registry; from gateway.presets import render_preset; print(render_preset(load_registry("gateway.yaml")))'
```

### Adding a Served Model

Most new models need no code, only YAML:

1. Add a `checkpoints` entry with its `repo`, `model`, `mmproj` and `args`. Embedders use `embedding: true`, `pooling: last`, `embd-normalize: 2`. Rerankers use `reranking: true`, `pooling: rank`.
2. Add a `served_models` entry that uses an existing `format` (`qwen3-vl-embedding` or `qwen3-vl-reranker`). Override `instructions` if needed: the keys are `document`/`query` for embedding and `rerank` for reranking.
3. Run `uv run gateway download` and restart.

The same Checkpoint can back several Served Models, for example `Qwen3-VL-Embedding-2B` and its `-query` alias with `role: query`. Placement is per Checkpoint: set `args.device` to put Checkpoints on different GPUs, and make those GPUs visible through `CUDA_VISIBLE_DEVICES`. Before shipping a new Checkpoint or quant, run the GPU smoke and parity tests (see [Tests](#tests)).

### Adding a Prompt Format

Add a model with new input conventions as a new Prompt Format:

1. Write a `render` function in `src/gateway/prompts.py`. For embedding it takes `(item, instruction)`; for reranking it takes `(query, document, instruction)`. It returns one Backend `/v1/embeddings` input of the form `{"content": [text and image_url parts]}`. The rendered prompt must match the model's official preprocessing exactly, including image placement and empty input.
2. Register it in `FORMATS` with its `kind` (`embedding` or `reranking`) and its default Instructions.
3. Add tests that assert the exact content sent to the Backend (see `tests/test_embeddings.py` and `tests/test_rerank.py`), and add a parity test against the official implementation.

The API layer, Request Adapters and post-processing stay unchanged.

## Tests

```bash
uv run pytest                     # default suite: in-process, Backend faked with httpx MockTransport, no GPU
```

The tests marked `gpu` run against a real router and are opt-in. The GPUs are shared, so pick an idle one with `nvidia-smi` and pin to it. These tests need the downloaded Checkpoints and a llama.cpp source checkout at `llama.cpp/`, because they use images from `llama.cpp/media/`.

```bash
# Smoke: normalized vectors, separated rerank scores, images change outputs
CUDA_VISIBLE_DEVICES=<idle gpu> uv run pytest -m gpu tests/test_gpu_smoke.py

# Parity with the official Qwen3-VL transformers scripts (one-time setup first)
uv sync --group parity
git -C models/Qwen3-VL-Embedding-2B lfs pull --include model.safetensors,tokenizer.json
git -C models/Qwen3-VL-Reranker-2B lfs pull --include model.safetensors,tokenizer.json
CUDA_VISIBLE_DEVICES=<idle gpu> uv run --group parity pytest -m gpu tests/test_gpu_parity.py
```

The parity test needs the Hugging Face reference checkouts at `models/Qwen3-VL-Embedding-2B/` and `models/Qwen3-VL-Reranker-2B/`. Get them with `GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/<name> models/<name>`.

## More

- [docs/weknora.md](docs/weknora.md): how to configure WeKnora's embedding and rerank rows.
- [docs/embedding/Qwen3-VL-Embedding.md](docs/embedding/Qwen3-VL-Embedding.md) and [docs/reranking/Qwen3-VL-Reranker.md](docs/reranking/Qwen3-VL-Reranker.md): research notes on the models and on llama.cpp (in Vietnamese).
- [docs/adr/](docs/adr/): architecture decisions.
