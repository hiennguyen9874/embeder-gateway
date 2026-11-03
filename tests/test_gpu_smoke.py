"""Smoke test against a real llama-server router (opt-in).

Run with an idle GPU pinned, after `uv run gateway download`:
    CUDA_VISIBLE_DEVICES=<idle gpu> uv run pytest -m gpu
"""

import base64
import math

import pytest
from conftest import CONFIG
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.backend import BackendClient
from gateway.config import load_registry
from gateway.supervisor import run_backend

pytestmark = pytest.mark.gpu
AUTH = {"Authorization": "Bearer smoke"}


@pytest.fixture(scope="module")
def gateway():
    registry = load_registry(CONFIG)
    with run_backend(registry) as (url, key):
        with TestClient(create_app(registry, BackendClient(url, key), {"smoke"})) as client:
            yield client


# A cartoon llama head and a matrix-multiplication diagram, from the vendored llama.cpp checkout.
LLAMA_PNG = CONFIG.parent / "llama.cpp" / "media" / "llama1-icon.png"
MATMUL_PNG = CONFIG.parent / "llama.cpp" / "media" / "matmul.png"


def uri(path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def embed(gateway, model: str, inputs: list, **extra) -> list[list[float]]:
    r = gateway.post("/v1/embeddings", headers=AUTH, json={"model": model, "input": inputs, **extra})
    assert r.status_code == 200, r.text
    return [d["embedding"] for d in r.json()["data"]]


def dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def rerank(gateway, query, documents) -> dict[int, float]:
    r = gateway.post("/v1/rerank", headers=AUTH, json={"model": "Qwen3-VL-Reranker-2B", "query": query, "documents": documents})
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    scores = [x["relevance_score"] for x in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 1 for s in scores), scores
    return {x["index"]: x["relevance_score"] for x in results}


def test_health_is_ok_once_the_router_has_loaded_the_preset(gateway):
    assert gateway.get("/health").status_code == 200


def test_vectors_are_normalized_and_retrieve_the_related_document(gateway):
    related, unrelated = embed(gateway, "Qwen3-VL-Embedding-2B", [
        "Paris is the capital and largest city of France.",
        "Mitochondria generate most of the chemical energy needed by a cell.",
    ])
    (query,) = embed(gateway, "Qwen3-VL-Embedding-2B-query", ["What is the capital of France?"])

    for vec in (related, unrelated, query):
        assert len(vec) == 2048
        assert math.isclose(math.sqrt(dot(vec, vec)), 1.0, abs_tol=1e-4)
    assert dot(query, related) > dot(query, unrelated) + 0.1, (dot(query, related), dot(query, unrelated))


def test_the_query_instruction_reaches_the_model(gateway):
    (as_document,) = embed(gateway, "Qwen3-VL-Embedding-2B", ["What is the capital of France?"])
    (as_query,) = embed(gateway, "Qwen3-VL-Embedding-2B-query", ["What is the capital of France?"])

    assert dot(as_document, as_query) < 0.999


def test_image_is_closer_to_a_matching_caption_than_an_unrelated_one(gateway):
    uri = "data:image/png;base64," + base64.b64encode(LLAMA_PNG.read_bytes()).decode()
    image, with_text = embed(gateway, "Qwen3-VL-Embedding-2B", [{"image": uri}, {"image": uri, "text": "A llama."}])
    matching, unrelated = embed(gateway, "Qwen3-VL-Embedding-2B", ["A cartoon llama.", "A bowl of ramen noodles."])

    for vec in (image, with_text):
        assert len(vec) == 2048
        assert math.isclose(math.sqrt(dot(vec, vec)), 1.0, abs_tol=1e-4)
    assert dot(image, matching) > dot(image, unrelated) + 0.05, (dot(image, matching), dot(image, unrelated))
    # The text part of a text+image Item reaches the model.
    assert dot(image, with_text) < 0.999


def test_rerank_scores_the_relevant_document_clearly_higher(gateway):
    # The bealore model card's example: about 0.73, 0.27 and 0.51 with its own (different) Instruction.
    scores = rerank(gateway, "What is the capital of France?", [
        "Paris is the capital and most populous city of France.",
        "Bananas are a yellow tropical fruit rich in potassium.",
        "The Eiffel Tower is a famous landmark located in Paris, France.",
    ])

    assert scores[0] > scores[1] + 0.3, scores
    assert scores[2] > scores[1], scores


def test_rerank_scores_image_documents_and_image_queries(gateway):
    llama, matmul = uri(LLAMA_PNG), uri(MATMUL_PNG)
    # (A text llama query barely prefers the stylized llama icon, natively in llama.cpp too, so it is not used.)
    by_image = rerank(gateway, "A matrix multiplication diagram.", [{"image": matmul}, {"image": llama}])
    by_text = rerank(gateway, {"image": llama}, ["A diagram of matrix multiplication.", "A cartoon llama."])

    assert by_image[0] > by_image[1] + 0.2, by_image
    assert by_text[1] > by_text[0] + 0.2, by_text


def test_weknora_messages_batch_and_mrl_dimensions(gateway):
    uri = "data:image/png;base64," + base64.b64encode(LLAMA_PNG.read_bytes()).decode()
    r = gateway.post("/v1/embeddings", headers=AUTH, json={
        "model": "Qwen3-VL-Embedding-2B", "encoding_format": "float", "truncate_prompt_tokens": 511,
        "add_special_tokens": True, "dimensions": 256,
        "messages": [
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}]}],
            [{"role": "user", "content": [{"type": "text", "text": "A cartoon llama."}]}],
        ],
    })
    assert r.status_code == 200, r.text
    image, caption = [d["embedding"] for d in r.json()["data"]]
    (full_image,) = embed(gateway, "Qwen3-VL-Embedding-2B", [{"image": uri}])

    assert len(image) == len(caption) == 256
    assert math.isclose(math.sqrt(dot(image, image)), 1.0, abs_tol=1e-4)
    # The `messages` path and the Item path render the same prompt.
    truncated = full_image[:256]
    assert dot(image, truncated) / math.sqrt(dot(truncated, truncated)) > 0.999
