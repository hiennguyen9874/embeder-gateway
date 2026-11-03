# Using the gateway from WeKnora

This guide sets up two WeKnora model rows against the gateway: an embedding row (text and image) and a rerank row (text and image documents). Every setting below was checked against the WeKnora source in `references/WeKnora/`; each section cites the files. Terms in **bold** are defined in [GLOSSARY.md](../GLOSSARY.md).

Both rows use the **Custom (OpenAI-compatible)** provider (`generic`, 自定义 (OpenAI兼容接口)). This is the path [ADR 0002](adr/0002-public-api-openai-embeddings-cohere-rerank.md) is designed around.

## Common settings

| Setting | Value |
| --- | --- |
| Provider | Custom (OpenAI-compatible), id `generic` |
| Base URL | `http://<gateway-host>:<port>/v1`, e.g. `http://10.0.0.5:8000/v1`. WeKnora appends `/embeddings` or `/rerank`. |
| API Key | One of the gateway's keys (`GATEWAY_API_KEYS` / `GATEWAY_API_KEY_FILE`). WeKnora sends it as `Authorization: Bearer <key>`. The `generic` provider marks the key optional, but the gateway rejects requests without one. |

The gateway listens on `127.0.0.1` by default. When WeKnora runs on another host or in a container, start it with `--host 0.0.0.0`; see the [README](../README.md). WeKnora also checks every Base URL for SSRF when the row is saved. If it rejects a private gateway address, add the host to WeKnora's `SSRF_WHITELIST`.

Sources: `internal/models/providers/generic.go`; `internal/models/api/openaiembeddings/client.go` (`defaultPath = "/embeddings"`); `internal/models/api/cohererank/client.go` (`defaultPath = "/rerank"`).

## Embedding row

| Setting | Value |
| --- | --- |
| Type | Embedding |
| Model name | `Qwen3-VL-Embedding-2B`. The name **must contain `qwen3-vl-embedding`** (case-insensitive) so that WeKnora's built-in `*qwen3-vl-embedding*` preset applies. |
| Dimension | `2048` |
| Custom Output Dimension | Off, which returns 2048. To get shorter MRL vectors, turn it on and set Dimension to a value from 64 to 2048: WeKnora then sends `dimensions` and the gateway truncates and re-normalizes. |
| Image Input | On. WeKnora detects this from the preset; you do not need to set it. |
| Protocol compat override (JSON) | `{"input_type_field": "input_type", "max_batch_size": 64, "max_image_bytes": 10485760, "image_mime_types": ["image/png", "image/jpeg"]}` |

The knowledge base must also have **image vector retrieval** (图片向量检索) turned on before images are embedded. This switch is off by default.

### What the preset sends

For `*qwen3-vl-embedding*` on `generic`, WeKnora sends every text and image as a vLLM `messages` conversation (`text_as_messages`). It batches several conversations as a 2-D `messages` array (`batch_messages`), sends at most 5 images per request, and adds `encoding_format: "float"`, `add_special_tokens: true` and, by default, `truncate_prompt_tokens`. The gateway accepts all of this:
- Each conversation becomes one Item, rendered with the official Qwen3-VL-Embedding Prompt Format.
- The vendor knobs are accepted and ignored. Input that is too long fails with 400 instead of being truncated.

You need no compat to make text and image embedding work. The gateway renders `input` and `messages` identically, so the `add_special_tokens` caveat in WeKnora's docs (which applies to vLLM) does not apply here.

### Why each compat key

- **`input_type_field: "input_type"`: query asymmetry.** Without it, WeKnora does not say whether a text is a query or a document, so the gateway treats every input as a Document and uses the Document Instruction. With it, WeKnora adds `input_type: "query"` when it embeds a search query (text or image) and `input_type: "document"` when it indexes. The gateway then uses the Query Instruction (`Retrieve relevant documents for the query.`) for queries. Keep the default values: do not set `input_type_values`, because `query`/`document` are what the gateway expects.
- **`max_batch_size: 64`** and **`max_image_bytes: 10485760`** match the gateway's default `limits` (`max_embedding_items`, `max_image_mb`). With them, WeKnora splits large batches and shrinks large images before sending, instead of getting a 400. Change both together if you change the gateway's `limits`.
- **`image_mime_types: ["image/png", "image/jpeg"]`**: WeKnora re-encodes any other format to JPEG before sending. The gateway accepts more formats, but the Backend decodes WebP only when ffmpeg is available (`llama.cpp/tools/mtmd/mtmd-helper.cpp`). PNG and JPEG always work.

### Re-indexing

WeKnora uses one model name for indexing and for searching, and it stores only Document-side vectors: query embeddings are made at search time (`internal/application/service/knowledgebase_search.go`). With the default `input_type_values`, Documents get the Document Instruction whether or not `input_type_field` is set. Adding or removing `input_type_field` therefore changes only how queries are embedded, and existing vectors stay valid.

These changes do alter stored vectors, so the knowledge base must be re-embedded after any of them:
- changing the Dimension (or Custom Output Dimension);
- changing the Checkpoint or its quantization;
- changing the Served Model's Document Instruction (`instructions.document` in `gateway.yaml`);
- setting `input_type_values` so that documents are sent with any value other than `document`;
- moving the row to the gateway from a server that embedded inputs differently, such as plain llama.cpp or vLLM without `text_as_messages`.

Do not point the embedding row at `Qwen3-VL-Embedding-2B-query`. That Served Model embeds every input, including indexed documents, with the Query Instruction. It exists for clients that cannot send `input_type`.

Sources: `internal/models/catalog/data/models.generated.json` (the `generic` entry `*qwen3-vl-embedding*`); `internal/models/catalog/catalog.go` (case-insensitive glob matching); `internal/models/api/embeddings_settings.go` (`input_type_field`, `InputTypeValue`); `internal/models/api/openaiembeddings/client.go` (`body`, `postMessages`); `internal/models/imageprep/prepare.go`; `internal/models/api/embeddings.go` (`EmbedDocument = "document"`, `EmbedQuery = "query"`).

## Rerank row

| Setting | Value |
| --- | --- |
| Type | Rerank |
| Model name | `Qwen3-VL-Reranker-2B` |
| Rerank score scale | **Already normalized relevance (0..1)** (`score_scale: "probability"`). This is also the default. Do not choose `logit`: WeKnora would apply a sigmoid to a probability and the relevance thresholds would stop meaning anything. |
| Protocol compat override (JSON) | `{"image_field": "image", "max_documents": 100, "max_image_bytes": 10485760, "image_mime_types": ["image/png", "image/jpeg"]}` |
| Model input (`spec.input`) | `["text", "image"]`: needed for image documents, see below |

The gateway returns `results[].index` and `relevance_score` = `P("yes")` in [0, 1], sorted in descending order. This is the Cohere shape that WeKnora's `cohere-rerank` protocol reads.

### Image documents

WeKnora sends image documents to a rerank row only when both of these are true:
1. the compat sets `image_field`. With `"image"`, documents go out as `{"image": "data:image/...;base64,..."}`, which the gateway accepts as Items;
2. the row's model input includes `image`. `Qwen3-VL-Reranker-2B` is not in WeKnora's catalog, so the row has to declare this itself.

The model editor only shows an *Image Input* switch for embedding rows. For a rerank row, declare it in one of two ways.

**Option A: through the REST API** (`POST /api/v1/models`):

```json
{
  "name": "Qwen3-VL-Reranker-2B",
  "type": "Rerank",
  "source": "remote",
  "parameters": {
    "provider": "generic",
    "base_url": "http://<gateway-host>:8000/v1",
    "api_key": "<gateway key>",
    "extra_config": {"score_scale": "probability"},
    "spec": {
      "input": ["text", "image"],
      "compat": {"image_field": "image", "max_documents": 100, "max_image_bytes": 10485760, "image_mime_types": ["image/png", "image/jpeg"]}
    }
  }
}
```

**Option B: through a deployment catalog overlay** (`config/models.json`, or the file named by `MODELS_CONFIG`). The overlay adds a `generic` catalog entry, so any row with this model name picks up the input and compat. WeKnora must be restarted after the change.

```json
{
  "providers": {
    "generic": {
      "models": [
        {"id": "Qwen3-VL-Reranker-2B", "type": "Rerank", "input": ["text", "image"],
         "compat": {"image_field": "image", "max_documents": 100, "max_image_bytes": 10485760, "image_mime_types": ["image/png", "image/jpeg"]}}
      ]
    }
  }
}
```

WeKnora reranks text and image documents in separate requests, and sends each image document in a request of its own (`max_image_batch_size` defaults to 1). The Qwen3-VL-Reranker scores every query–document pair on its own, so text and image scores for the same query are directly comparable and can be merged. `max_documents: 100` matches the gateway's `max_rerank_documents`, so larger candidate sets are split instead of rejected.

Sources: `internal/models/api/rerank_settings.go` (`DefaultRerank`: `ScoreProbability`); `internal/models/runtime/resolve.go` (`resolveRerank`: `cohere-rerank` default, `score_scale` from `extra_config`); `internal/models/rerank/reranker.go` (images only when the client has `image_field` *and* `spec.input` has `image`); `internal/models/api/cohererank/client.go` (`BuildImageRequestBody`); `internal/handler/model.go` and `internal/types/model.go` (`parameters.spec`); `config/models.json.example`; `website-docs/03-features/06-models.md`.

## Checking the setup

1. Use *Test connection* on each row. A 401 means the API key is wrong. A 404 lists the gateway's Served Models, which helps when the model name has a typo.
2. `curl http://<gateway-host>:8000/health` returns 200 only when both Checkpoints are loaded.
3. The gateway's default test suite replays WeKnora's exact request bodies: `tests/test_multimodal.py` (the `test_weknora_*` tests, with bodies from `internal/models/embedding/image_wire_test.go`) and `tests/test_rerank.py::test_weknora_jina_style_image_body_is_scored_one_request_per_image`.
