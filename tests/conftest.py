import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.backend import BackendClient
from gateway.config import load_registry

CONFIG = Path(__file__).parent.parent / "gateway.yaml"
API_KEY = "test-key"
BACKEND_KEY = "internal-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


class FakeBackend:
    """A llama-server router behind httpx.MockTransport that records every request."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        # Raw, un-normalized vector: the gateway must return it L2-normalized as [0.6, 0.8, 0, ...].
        self.vector = [3.0, 4.0] + [0.0] * 2046
        self.statuses = {"Qwen3-VL-Embedding-2B-Q8_0": "loaded"}
        self.embeddings_response: httpx.Response | None = None
        # Per-input hooks: the vector to return, and an error message to reject the input with (400).
        self.vector_for: Callable[[dict], list[float]] = lambda input: self.vector
        self.reject: Callable[[dict], str | None] = lambda input: None

    @property
    def embedding_bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.path == "/v1/embeddings"]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.headers.get("authorization") != f"Bearer {BACKEND_KEY}":
            return httpx.Response(401, json={"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}})
        if request.method == "GET" and request.url.path == "/models":
            return httpx.Response(200, json={"data": [
                {"id": name, "object": "model", "status": {"value": status}} for name, status in self.statuses.items()
            ]})
        if request.method == "POST" and request.url.path == "/v1/embeddings":
            if self.embeddings_response is not None:
                return self.embeddings_response
            body = json.loads(request.content)
            for input in body["input"]:
                if message := self.reject(input):
                    return httpx.Response(400, json={"error": {"code": 400, "message": message, "type": "exceed_context_size_error"}})
            return httpx.Response(200, json={
                "model": body["model"],
                "object": "list",
                "data": [{"object": "embedding", "index": i, "embedding": self.vector_for(input)} for i, input in enumerate(body["input"])],
                "usage": {"prompt_tokens": 7 * len(body["input"]), "total_tokens": 7 * len(body["input"])},
            })
        return httpx.Response(404, json={"error": {"message": "File Not Found", "type": "not_found_error", "code": 404}})


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def gateway(backend: FakeBackend):
    client = BackendClient("http://backend", BACKEND_KEY, transport=httpx.MockTransport(backend.handle))
    app = create_app(load_registry(CONFIG), client, api_keys={API_KEY})
    with TestClient(app) as c:
        yield c
