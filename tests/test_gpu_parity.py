"""Parity of the gateway (Q8_0 Checkpoints + Prompt Formats) with the official Qwen3-VL transformers scripts (opt-in).

The reference is `scripts/` of the Hugging Face checkouts in models/, run in float32 with the official default
Instructions. Every input and every rerank pair is its own test case, so a failure names what diverged and by how much;
`--junitxml` also records each measured value as a property.

Setup, once:
    uv sync --group parity
    uv run gateway download
    git -C models/Qwen3-VL-Embedding-2B lfs pull --include model.safetensors,tokenizer.json
    git -C models/Qwen3-VL-Reranker-2B lfs pull --include model.safetensors,tokenizer.json
Run with an idle GPU pinned:
    CUDA_VISIBLE_DEVICES=<idle gpu> uv run --group parity pytest -m gpu tests/test_gpu_parity.py
"""

import base64
import gc
import importlib.util
import math

import pytest
from conftest import CONFIG
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.backend import BackendClient
from gateway.config import load_registry
from gateway.download import _is_downloaded
from gateway.supervisor import run_backend

pytestmark = pytest.mark.gpu
AUTH = {"Authorization": "Bearer parity"}
MODELS = CONFIG.parent / "models"
EMBEDDER = MODELS / "Qwen3-VL-Embedding-2B"
RERANKER = MODELS / "Qwen3-VL-Reranker-2B"
# The checkouts' LFS files: Hugging Face clones without smudging leave them as pointers.
REFERENCE_FILES = ("model.safetensors", "tokenizer.json")

# Tolerances for Q8_0 weights + f16 mmproj against the float32 reference, with headroom over what was measured
# (llama.cpp b12072, A100). Images are resized by llama.cpp's own bicubic, so they get a looser bound than text.
MIN_TEXT_COSINE = 0.99  # measured >= 0.998
MIN_IMAGE_COSINE = 0.97  # measured >= 0.982 (image only), >= 0.991 (image+text)
MAX_RERANK_DELTA = 0.06  # measured <= 0.042 (with the f16 GGUF: <= 0.003 for text, <= 0.024 with images)
# Rankings must agree on every two candidates the reference separates by more than this; closer ones are ties at
# Q8_0 precision (measured flips at reference gaps of 0.016 for embedding similarity and 0.005 for rerank scores).
RANK_TIE_MARGIN = 0.03

# The official Instructions: the embedder script's default, the Query one from the model card, the reranker script's default.
DOCUMENT_INSTRUCTION = "Represent the user's input."
QUERY_INSTRUCTION = "Retrieve relevant documents for the query."

# From the vendored llama.cpp checkout: a 500x500 cartoon llama with a transparent background, and a 2875x1500 diagram
# that both sides must downscale.
LLAMA_PNG = CONFIG.parent / "llama.cpp" / "media" / "llama1-icon.png"
MATMUL_PNG = CONFIG.parent / "llama.cpp" / "media" / "matmul.png"
LLAMA, MATMUL = "llama", "matmul"

QUERIES = {
    "capital": {"text": "What is the capital of France?"},
    "energy": {"text": "How do cells produce energy?"},
    "llama": {"text": "A cartoon llama."},
    "matmul": {"text": "A diagram of matrix multiplication."},
}
DOCUMENTS = {
    "paris-text": {"text": "Paris is the capital and largest city of France."},
    "mitochondria-text": {"text": "Mitochondria generate most of the chemical energy needed by a cell."},
    "llama-image": {"image": LLAMA},
    "matmul-image": {"image": MATMUL},
    "llama-image+text": {"image": LLAMA, "text": "The llama.cpp mascot."},
}
RERANKS = {
    "capital": ({"text": "What is the capital of France?"}, [
        {"text": "Paris is the capital and most populous city of France."},
        {"text": "Bananas are a yellow tropical fruit rich in potassium."},
        {"text": "The Eiffel Tower is a famous landmark located in Paris, France."},
        {"image": LLAMA},
    ]),
    "matmul-text-query": ({"text": "A matrix multiplication diagram."}, [
        {"image": MATMUL},
        {"image": LLAMA},
        {"image": LLAMA, "text": "The llama.cpp mascot."},
    ]),
    "llama-image-query": ({"image": LLAMA}, [
        {"text": "A diagram of matrix multiplication."},
        {"text": "A cartoon llama."},
        {"text": "Paris is the capital of France."},
    ]),
}
PAIRS = [(case, i) for case, (_, documents) in RERANKS.items() for i in range(len(documents))]
IMAGES = {LLAMA: LLAMA_PNG, MATMUL: MATMUL_PNG}


def _for_gateway(item: dict) -> dict:
    if "image" not in item:
        return item
    return {**item, "image": "data:image/png;base64," + base64.b64encode(IMAGES[item["image"]].read_bytes()).decode()}


def _for_reference(item: dict) -> dict:
    # A PIL image, not a path, so the reference decodes the very same bytes.
    from PIL import Image

    return {**item, "image": Image.open(IMAGES[item["image"]])} if "image" in item else item


def _load_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free(torch) -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


@pytest.fixture(scope="module")
def torch():
    for module in ("transformers", "qwen_vl_utils", "scipy"):
        pytest.importorskip(module, reason="reference dependencies missing: uv sync --group parity")
    torch = pytest.importorskip("torch", reason="reference dependencies missing: uv sync --group parity")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    for checkout in (EMBEDDER, RERANKER):
        if not all(_is_downloaded(checkout / f) for f in REFERENCE_FILES):
            pytest.skip(f"reference weights missing: git -C {checkout} lfs pull --include {','.join(REFERENCE_FILES)}")
    registry = load_registry(CONFIG)
    if not all(_is_downloaded(c.dir / f) for c in registry.checkpoints.values() for f in c.files):
        pytest.skip("Checkpoints missing: uv run gateway download")
    if not registry.llama_server.is_file():
        pytest.skip(f"llama-server not found at {registry.llama_server}")
    return torch


@pytest.fixture(scope="module")
def reference_embeddings(torch) -> dict[str, list[float]]:
    script = _load_script(EMBEDDER / "scripts" / "qwen3_vl_embedding.py")
    embedder = script.Qwen3VLEmbedder(str(EMBEDDER), dtype=torch.float32)
    vectors = {}
    for prefix, items, instruction in (("query", QUERIES, QUERY_INSTRUCTION), ("document", DOCUMENTS, DOCUMENT_INSTRUCTION)):
        for name, item in items.items():
            (vector,) = embedder.process([{**_for_reference(item), "instruction": instruction}]).tolist()
            vectors[f"{prefix}:{name}"] = vector
    del embedder
    _free(torch)
    return vectors


@pytest.fixture(scope="module")
def reference_scores(torch) -> dict[str, list[float]]:
    script = _load_script(RERANKER / "scripts" / "qwen3_vl_reranker.py")
    reranker = script.Qwen3VLReranker(str(RERANKER), dtype=torch.float32)
    scores = {
        case: reranker.process({"query": _for_reference(query), "documents": [_for_reference(d) for d in documents]})
        for case, (query, documents) in RERANKS.items()
    }
    del reranker
    _free(torch)
    return scores


@pytest.fixture(scope="module")
def gateway(torch, reference_embeddings, reference_scores):
    # Started after the references are computed and freed, so both never hold the GPU at once.
    registry = load_registry(CONFIG)
    with run_backend(registry) as (url, key):
        with TestClient(create_app(registry, BackendClient(url, key), {"parity"})) as client:
            yield client


@pytest.fixture(scope="module")
def gateway_embeddings(gateway) -> dict[str, list[float]]:
    vectors = {}
    for prefix, items in (("query", QUERIES), ("document", DOCUMENTS)):
        r = gateway.post("/v1/embeddings", headers=AUTH, json={
            "model": "Qwen3-VL-Embedding-2B", "input_type": prefix, "input": [_for_gateway(i) for i in items.values()],
        })
        assert r.status_code == 200, r.text
        vectors |= {f"{prefix}:{name}": d["embedding"] for name, d in zip(items, r.json()["data"])}
    return vectors


@pytest.fixture(scope="module")
def gateway_scores(gateway) -> dict[str, list[float]]:
    scores = {}
    for case, (query, documents) in RERANKS.items():
        r = gateway.post("/v1/rerank", headers=AUTH, json={
            "model": "Qwen3-VL-Reranker-2B", "query": _for_gateway(query), "documents": [_for_gateway(d) for d in documents],
        })
        assert r.status_code == 200, r.text
        by_index = {x["index"]: x["relevance_score"] for x in r.json()["results"]}
        scores[case] = [by_index[i] for i in range(len(documents))]
    return scores


def _ranking(scores: dict) -> list:
    return sorted(scores, key=scores.get, reverse=True)


def _assert_same_ranking(what: str, ours: dict, theirs: dict) -> None:
    swapped = [(a, b) for a in theirs for b in theirs if theirs[a] > theirs[b] + RANK_TIE_MARGIN and ours[a] <= ours[b]]
    assert not swapped, (
        f"{what} ranks differently: {', '.join(f'{a} above {b}' for a, b in swapped)} in the reference only\n"
        f"  gateway:   {[(k, round(ours[k], 4)) for k in _ranking(ours)]}\n"
        f"  reference: {[(k, round(theirs[k], 4)) for k in _ranking(theirs)]}"
    )


@pytest.mark.parametrize("name", [f"query:{n}" for n in QUERIES] + [f"document:{n}" for n in DOCUMENTS])
def test_embedding_matches_reference(name, gateway_embeddings, reference_embeddings, record_property):
    item = (QUERIES | DOCUMENTS)[name.split(":", 1)[1]]
    threshold = MIN_IMAGE_COSINE if "image" in item else MIN_TEXT_COSINE
    cosine = _cosine(gateway_embeddings[name], reference_embeddings[name])
    record_property("cosine", cosine)

    assert cosine >= threshold, f"{name}: cosine(gateway, reference) = {cosine:.5f} < {threshold}"


@pytest.mark.parametrize("query", QUERIES)
def test_retrieval_ranking_matches_reference(query, gateway_embeddings, reference_embeddings, record_property):
    def similarities(vectors):
        return {d: _cosine(vectors[f"query:{query}"], vectors[f"document:{d}"]) for d in DOCUMENTS}

    ours, theirs = similarities(gateway_embeddings), similarities(reference_embeddings)
    record_property("gateway", ours)
    record_property("reference", theirs)

    _assert_same_ranking(f"query:{query}", ours, theirs)


@pytest.mark.parametrize(("case", "index"), PAIRS, ids=[f"{c}-doc{i}" for c, i in PAIRS])
def test_rerank_score_matches_reference(case, index, gateway_scores, reference_scores, record_property):
    ours, theirs = gateway_scores[case][index], reference_scores[case][index]
    record_property("gateway", ours)
    record_property("reference", theirs)

    assert abs(ours - theirs) <= MAX_RERANK_DELTA, (
        f"{case} document {index}: gateway {ours:.4f} vs reference {theirs:.4f}, "
        f"|Δ| = {abs(ours - theirs):.4f} > {MAX_RERANK_DELTA}"
    )


@pytest.mark.parametrize("case", RERANKS)
def test_rerank_ordering_matches_reference(case, gateway_scores, reference_scores):
    ours, theirs = dict(enumerate(gateway_scores[case])), dict(enumerate(reference_scores[case]))

    _assert_same_ranking(case, ours, theirs)
