"""Smoke test against a real llama-server router (opt-in).

Run with an idle GPU pinned, after `uv run gateway download`:
    CUDA_VISIBLE_DEVICES=<idle gpu> uv run pytest -m gpu
"""

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


def embed(gateway, model: str, texts: list[str]) -> list[list[float]]:
    r = gateway.post("/v1/embeddings", headers=AUTH, json={"model": model, "input": texts})
    assert r.status_code == 200, r.text
    return [d["embedding"] for d in r.json()["data"]]


def dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


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
