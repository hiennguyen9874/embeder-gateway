"""API layer: routes, Bearer auth, OpenAI-style errors."""

import hmac
import math
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException

from gateway.backend import BackendClient, BackendError
from gateway.config import Registry, ServedModel

# input_type values (Cohere/Voyage/NIM style) -> Input Role
INPUT_TYPES = {
    "query": "query",
    "search_query": "query",
    "document": "document",
    "search_document": "document",
    "passage": "document",
}


class APIError(Exception):
    def __init__(self, status: int, message: str, code: str | None = None, type: str = "invalid_request_error"):
        self.status, self.message, self.code, self.type = status, message, code, type


def error_response(status: int, message: str, code: str | None = None, type: str = "invalid_request_error"):
    return JSONResponse({"error": {"message": message, "type": type, "code": code}}, status_code=status)


class EmbeddingsRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float"] = "float"
    dimensions: int | None = None
    input_type: str | None = None
    task: str | None = None
    # Accepted and ignored (ADR 0002): truncate_prompt_tokens, add_special_tokens, truncate, user.


def resolve_role(req: EmbeddingsRequest, model: ServedModel) -> str:
    if req.input_type in INPUT_TYPES:
        return INPUT_TYPES[req.input_type]
    if req.task and req.task.endswith(".query"):
        return "query"
    if req.task and req.task.endswith(".passage"):
        return "document"
    return model.role


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


def create_app(registry: Registry, backend: BackendClient, api_keys: set[str]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await backend.aclose()

    app = FastAPI(title="Embedder Gateway", lifespan=lifespan)

    def served_model(name: str) -> ServedModel:
        if name not in registry.served_models:
            raise APIError(
                404,
                f"The model {name!r} does not exist. Served models: {', '.join(registry.served_models)}",
                code="model_not_found",
            )
        return registry.served_models[name]

    @app.middleware("http")
    async def bearer_auth(request: Request, call_next):
        # Middleware rather than a dependency so a bad key is a 401 even when the body is invalid.
        if request.url.path != "/health":
            scheme, _, key = request.headers.get("authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not any(hmac.compare_digest(key.encode(), k.encode()) for k in api_keys):
                response = error_response(401, "Invalid or missing API key", code="invalid_api_key")
                response.headers["WWW-Authenticate"] = "Bearer"
                return response
        return await call_next(request)

    @app.exception_handler(APIError)
    async def api_error(request: Request, e: APIError):
        return error_response(e.status, e.message, e.code, e.type)

    @app.exception_handler(BackendError)
    async def backend_error(request: Request, e: BackendError):
        # A 400 from the Backend is about the input (e.g. too many tokens); anything else is ours to own.
        if e.status == 400:
            return error_response(400, e.message, code="invalid_input")
        status = 503 if e.status == 503 else 502
        return error_response(status, f"backend error: {e.message}", code="backend_error", type="server_error")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, e: RequestValidationError):
        message = "; ".join(f"{'.'.join(str(p) for p in err['loc'][1:]) or 'body'}: {err['msg']}" for err in e.errors())
        return error_response(400, message, code="invalid_request")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, e: HTTPException):
        return error_response(e.status_code, str(e.detail))

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [
            {
                "id": m.name, "object": "model", "created": 0, "owned_by": "embedder-gateway",
                "kind": m.format.kind, "modalities": list(m.modalities), "dimensions": m.dimensions,
            }
            for m in registry.served_models.values()
        ]}

    @app.get("/health")
    async def health():
        needed = sorted(registry.checkpoints)
        try:
            statuses = await backend.model_statuses()
        except BackendError as e:
            return JSONResponse({"status": "unavailable", "error": e.message}, status_code=503)
        checkpoints = {name: statuses.get(name, "missing") for name in needed}
        ready = all(s == "loaded" for s in checkpoints.values())
        return JSONResponse(
            {"status": "ok" if ready else "unavailable", "checkpoints": checkpoints},
            status_code=200 if ready else 503,
        )

    @app.post("/v1/embeddings")
    async def embeddings(req: EmbeddingsRequest):
        model = served_model(req.model)
        texts = [req.input] if isinstance(req.input, str) else req.input
        if not texts:
            raise APIError(400, "input: must not be empty", code="invalid_request")
        if req.dimensions is not None and req.dimensions != model.dimensions:
            raise APIError(400, f"dimensions: {req.model} only supports {model.dimensions}", code="invalid_request")

        instruction = model.instructions[resolve_role(req, model)]
        result = await backend.embed(model.checkpoint.name, [model.format.render(t, instruction) for t in texts])
        data = sorted(result["data"], key=lambda d: d["index"])
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": l2_normalize(d["embedding"])} for i, d in enumerate(data)],
            "model": model.name,
            "usage": result.get("usage", {"prompt_tokens": 0, "total_tokens": 0}),
        }

    return app
