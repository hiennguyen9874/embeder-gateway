import pytest
from conftest import AUTH


def assert_openai_error(r, status: int, code: str | None = None) -> dict:
    assert r.status_code == status, r.text
    error = r.json()["error"]
    assert set(error) == {"message", "type", "code"}
    assert isinstance(error["message"], str) and error["message"]
    if code is not None:
        assert error["code"] == code
    return error


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "test-key"}])
@pytest.mark.parametrize("method, path", [("POST", "/v1/embeddings"), ("GET", "/v1/models")])
def test_missing_or_invalid_bearer_key_is_401(gateway, backend, headers, method, path):
    r = gateway.request(method, path, headers=headers, json={"model": "Qwen3-VL-Embedding-2B", "input": "x"})

    assert_openai_error(r, 401, "invalid_api_key")
    assert r.headers["www-authenticate"] == "Bearer"
    assert backend.requests == []


def test_auth_is_checked_before_the_body_is_parsed(gateway):
    r = gateway.post("/v1/embeddings", content=b"{not json", headers={"Content-Type": "application/json"})

    assert_openai_error(r, 401, "invalid_api_key")


def test_unknown_model_is_404_listing_served_models(gateway, backend):
    r = gateway.post("/v1/embeddings", headers=AUTH, json={"model": "qwen3-vl-embeding", "input": "x"})

    error = assert_openai_error(r, 404, "model_not_found")
    assert "Qwen3-VL-Embedding-2B" in error["message"]
    assert "Qwen3-VL-Embedding-2B-query" in error["message"]
    assert backend.requests == []


@pytest.mark.parametrize("body", [
    {"input": "x"},  # no model
    {"model": "Qwen3-VL-Embedding-2B"},  # no input
    {"model": "Qwen3-VL-Embedding-2B", "input": []},
    {"model": "Qwen3-VL-Embedding-2B", "input": 42},
    {"model": "Qwen3-VL-Embedding-2B", "input": "x", "dimensions": 63},
    {"model": "Qwen3-VL-Embedding-2B", "input": "x", "dimensions": 2049},
    {"model": "Qwen3-VL-Embedding-2B", "input": "x", "encoding_format": "int8"},
    {"model": "Qwen3-VL-Embedding-2B", "input": ["x", 7]},
    {"model": "Qwen3-VL-Embedding-2B", "input": "x", "messages": [{"role": "user", "content": "x"}]},
    {"model": "Qwen3-VL-Embedding-2B", "messages": []},
    {"model": "Qwen3-VL-Embedding-2B", "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]},
    {"model": "Qwen3-VL-Embedding-2B", "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]},
    {"model": "Qwen3-VL-Embedding-2B", "messages": [{"role": "user", "content": [{"type": "text", "text": 1}]}]},
    {"model": "Qwen3-VL-Embedding-2B", "input": [{"txt": "typo"}]},
])
def test_invalid_embeddings_request_is_400(gateway, backend, body):
    assert_openai_error(gateway.post("/v1/embeddings", headers=AUTH, json=body), 400)
    assert backend.embedding_bodies == []


def test_malformed_json_is_400(gateway):
    r = gateway.post("/v1/embeddings", headers={**AUTH, "Content-Type": "application/json"}, content=b"{not json")

    assert_openai_error(r, 400)


def test_unknown_route_is_404_in_openai_shape(gateway):
    assert_openai_error(gateway.get("/v1/nope", headers=AUTH), 404)


def test_backend_rejecting_the_input_is_400(gateway, backend):
    import httpx

    backend.embeddings_response = httpx.Response(400, json={"error": {
        "code": 400, "message": "input (9000 tokens) is too large to process", "type": "invalid_request_error",
    }})
    error = assert_openai_error(gateway.post("/v1/embeddings", headers=AUTH, json={"model": "Qwen3-VL-Embedding-2B", "input": "x"}), 400)
    assert error["message"] == "input: input (9000 tokens) is too large to process"


def test_backend_failure_is_502(gateway, backend):
    import httpx

    backend.embeddings_response = httpx.Response(500, json={"error": {"code": 500, "message": "boom", "type": "server_error"}})
    error = assert_openai_error(gateway.post("/v1/embeddings", headers=AUTH, json={"model": "Qwen3-VL-Embedding-2B", "input": "x"}), 502)
    assert error["type"] == "server_error"


def test_models_lists_every_served_model_with_its_capabilities(gateway, backend):
    r = gateway.get("/v1/models", headers=AUTH)

    assert r.status_code == 200, r.text
    assert r.json() == {"object": "list", "data": [
        {"id": "Qwen3-VL-Embedding-2B", "object": "model", "created": 0, "owned_by": "embedder-gateway",
         "kind": "embedding", "modalities": ["text", "image"], "dimensions": 2048, "max_dimensions": 2048},
        {"id": "Qwen3-VL-Embedding-2B-query", "object": "model", "created": 0, "owned_by": "embedder-gateway",
         "kind": "embedding", "modalities": ["text", "image"], "dimensions": 2048, "max_dimensions": 2048},
    ]}
    assert backend.requests == []


def test_health_is_200_when_every_preset_is_loaded_and_needs_no_key(gateway, backend):
    r = gateway.get("/health")

    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ok", "checkpoints": {"Qwen3-VL-Embedding-2B-Q8_0": "loaded"}}


@pytest.mark.parametrize("statuses, reported", [
    ({"Qwen3-VL-Embedding-2B-Q8_0": "loading"}, "loading"),
    ({"Qwen3-VL-Embedding-2B-Q8_0": "unloaded"}, "unloaded"),
    ({"some-other-model": "loaded"}, "missing"),
])
def test_health_is_503_until_every_preset_is_loaded(gateway, backend, statuses, reported):
    backend.statuses = statuses
    r = gateway.get("/health")

    assert r.status_code == 503, r.text
    assert r.json() == {"status": "unavailable", "checkpoints": {"Qwen3-VL-Embedding-2B-Q8_0": reported}}


@pytest.mark.parametrize("status", ["unloaded", "loading", None, "loaded"])
def test_health_checks_checkpoints_without_served_models(tmp_path, backend, status):
    import httpx
    import yaml
    from fastapi.testclient import TestClient

    from conftest import BACKEND_KEY, CONFIG
    from gateway.app import create_app
    from gateway.backend import BackendClient
    from gateway.config import load_registry

    raw = yaml.safe_load(CONFIG.read_text())
    raw["checkpoints"]["extra"] = dict(next(iter(raw["checkpoints"].values())))
    config = tmp_path / "gateway.yaml"
    config.write_text(yaml.safe_dump(raw))
    if status is not None:
        backend.statuses["extra"] = status

    client = BackendClient("http://backend", BACKEND_KEY, transport=httpx.MockTransport(backend.handle))
    with TestClient(create_app(load_registry(config), client, {"k"})) as gateway:
        r = gateway.get("/health")

    ready = status == "loaded"
    assert r.status_code == (200 if ready else 503), r.text
    assert r.json() == {
        "status": "ok" if ready else "unavailable",
        "checkpoints": {"Qwen3-VL-Embedding-2B-Q8_0": "loaded", "extra": status or "missing"},
    }


def test_health_is_503_when_the_backend_is_unreachable():
    import httpx
    from fastapi.testclient import TestClient

    from conftest import CONFIG
    from gateway.app import create_app
    from gateway.backend import BackendClient
    from gateway.config import load_registry

    def refuse(request):
        raise httpx.ConnectError("connection refused")

    app = create_app(load_registry(CONFIG), BackendClient("http://backend", "k", transport=httpx.MockTransport(refuse)), {"k"})
    with TestClient(app) as c:
        r = c.get("/health")

    assert r.status_code == 503
    assert r.json()["status"] == "unavailable"
