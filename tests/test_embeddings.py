import math

import pytest
from conftest import AUTH


def rendered(instruction: str, text: str) -> str:
    # The official Qwen3-VL-Embedding chat template with add_generation_prompt=True.
    return (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


DOCUMENT = "Represent the user's input."
QUERY = "Retrieve relevant documents for the query."


def test_string_input_is_embedded_with_the_document_instruction(gateway, backend):
    r = gateway.post("/v1/embeddings", headers=AUTH, json={"model": "Qwen3-VL-Embedding-2B", "input": "Follow the white rabbit."})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "Qwen3-VL-Embedding-2B"
    assert [d["index"] for d in body["data"]] == [0]
    assert body["data"][0]["object"] == "embedding"
    vec = body["data"][0]["embedding"]
    assert len(vec) == 2048
    assert vec[:3] == [0.6, 0.8, 0.0]
    assert math.isclose(sum(x * x for x in vec), 1.0)
    assert body["usage"] == {"prompt_tokens": 7, "total_tokens": 7}

    assert backend.embedding_bodies == [{
        "model": "Qwen3-VL-Embedding-2B-Q8_0",
        "input": [{"content": [{"type": "text", "text": rendered(DOCUMENT, "Follow the white rabbit.")}]}],
        "encoding_format": "float",
    }]


def backend_texts(backend) -> list[str]:
    return [i["content"][0]["text"] for body in backend.embedding_bodies for i in body["input"]]


def test_string_array_input_returns_one_vector_per_input_in_order(gateway, backend):
    r = gateway.post("/v1/embeddings", headers=AUTH, json={
        "model": "Qwen3-VL-Embedding-2B", "input": ["first", "second", "third"], "encoding_format": "float",
    })

    assert r.status_code == 200, r.text
    assert [d["index"] for d in r.json()["data"]] == [0, 1, 2]
    assert backend_texts(backend) == [rendered(DOCUMENT, t) for t in ["first", "second", "third"]]
    assert r.json()["usage"] == {"prompt_tokens": 21, "total_tokens": 21}


def test_backend_order_is_mapped_back_by_index(gateway, backend):
    import httpx

    backend.embeddings_response = httpx.Response(200, json={"data": [
        {"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [2.0, 0.0]},
    ]})
    r = gateway.post("/v1/embeddings", headers=AUTH, json={"model": "Qwen3-VL-Embedding-2B", "input": ["a", "b"]})

    assert [(d["index"], d["embedding"]) for d in r.json()["data"]] == [(0, [1.0, 0.0]), (1, [0.0, 1.0])]


@pytest.mark.parametrize("model, extra, instruction", [
    ("Qwen3-VL-Embedding-2B", {}, DOCUMENT),
    ("Qwen3-VL-Embedding-2B", {"input_type": "query"}, QUERY),
    ("Qwen3-VL-Embedding-2B", {"input_type": "search_query"}, QUERY),
    ("Qwen3-VL-Embedding-2B", {"input_type": "document"}, DOCUMENT),
    ("Qwen3-VL-Embedding-2B", {"task": "retrieval.query"}, QUERY),
    ("Qwen3-VL-Embedding-2B", {"task": "retrieval.passage"}, DOCUMENT),
    ("Qwen3-VL-Embedding-2B-query", {}, QUERY),
    # input_type wins over task, and both win over the Served Model's role.
    ("Qwen3-VL-Embedding-2B", {"input_type": "document", "task": "retrieval.query"}, DOCUMENT),
    ("Qwen3-VL-Embedding-2B-query", {"task": "retrieval.passage"}, DOCUMENT),
    ("Qwen3-VL-Embedding-2B-query", {"input_type": "passage"}, DOCUMENT),
    # Unrecognized signals fall through to the next rule.
    ("Qwen3-VL-Embedding-2B", {"input_type": "classification", "task": "text-matching"}, DOCUMENT),
])
def test_input_role_selects_the_instruction(gateway, backend, model, extra, instruction):
    r = gateway.post("/v1/embeddings", headers=AUTH, json={"model": model, "input": "what is a llama?", **extra})

    assert r.status_code == 200, r.text
    assert r.json()["model"] == model
    assert backend_texts(backend) == [rendered(instruction, "what is a llama?")]
    assert backend.embedding_bodies[0]["model"] == "Qwen3-VL-Embedding-2B-Q8_0"


def test_vendor_knobs_weknora_sends_are_accepted_and_not_forwarded(gateway, backend):
    r = gateway.post("/v1/embeddings", headers=AUTH, json={
        "model": "Qwen3-VL-Embedding-2B", "input": ["a"], "encoding_format": "float",
        "truncate_prompt_tokens": 512, "add_special_tokens": True, "truncate": True, "user": "u1",
    })

    assert r.status_code == 200, r.text
    assert set(backend.embedding_bodies[0]) == {"model", "input", "encoding_format"}


def test_empty_text_is_embedded_as_null_like_the_official_script(gateway, backend):
    gateway.post("/v1/embeddings", headers=AUTH, json={"model": "Qwen3-VL-Embedding-2B", "input": ""})

    assert backend_texts(backend) == [rendered(DOCUMENT, "NULL")]


def test_served_model_can_override_instructions_and_gets_official_punctuation(tmp_path, backend):
    import httpx
    from conftest import BACKEND_KEY, CONFIG
    from fastapi.testclient import TestClient

    from gateway.app import create_app
    from gateway.backend import BackendClient
    from gateway.config import load_registry

    config = tmp_path / "gateway.yaml"
    config.write_text(CONFIG.read_text() + """
  products:
    checkpoint: Qwen3-VL-Embedding-2B-Q8_0
    format: qwen3-vl-embedding
    dimensions: 2048
    modalities: [text]
    instructions: {query: "  Find the product the shopper describes  "}
""")
    client = BackendClient("http://backend", BACKEND_KEY, transport=httpx.MockTransport(backend.handle))
    with TestClient(create_app(load_registry(config), client, {"k"})) as gateway:
        for input_type in ("query", "document"):
            r = gateway.post("/v1/embeddings", headers={"Authorization": "Bearer k"},
                             json={"model": "products", "input": "red shoes", "input_type": input_type})
            assert r.status_code == 200, r.text

    assert backend_texts(backend) == [
        rendered("Find the product the shopper describes.", "red shoes"),
        rendered(DOCUMENT, "red shoes"),
    ]
