"""API layer: routes, Bearer auth, OpenAI-style errors."""

import asyncio
import base64
import hmac
import math
import struct
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException

from gateway.backend import BackendClient, BackendError
from gateway.config import Registry, ServedModel
from gateway.items import InvalidItem, Item, embedding_items, rerank_items

# input_type values (Cohere/Voyage/NIM style) -> Input Role
INPUT_TYPES = {
    "query": "query",
    "search_query": "query",
    "document": "document",
    "search_document": "document",
    "passage": "document",
}
ROUTES = {"embedding": "/v1/embeddings", "reranking": "/v1/rerank"}


class APIError(Exception):
    def __init__(self, status: int, message: str, code: str | None = None, type: str = "invalid_request_error"):
        self.status, self.message, self.code, self.type = status, message, code, type


def error_response(status: int, message: str, code: str | None = None, type: str = "invalid_request_error"):
    return JSONResponse({"error": {"message": message, "type": type, "code": code}}, status_code=status)


class EmbeddingsRequest(BaseModel):
    model: str
    # Shapes are checked by the Request Adapters, which name the offending Item.
    input: Any = None
    messages: Any = None
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    input_type: str | None = None
    task: str | None = None
    # Accepted and ignored (ADR 0002): truncate_prompt_tokens, add_special_tokens, truncate, user.


class RerankRequest(BaseModel):
    model: str
    # Required but may be anything: shapes are checked by the Request Adapters.
    query: Any
    documents: Any
    top_n: int | None = Field(None, ge=1)
    return_documents: bool = False
    # Accepted and ignored (ADR 0002): truncate_prompt_tokens, truncate.


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


def encode(vec: list[float], encoding_format: str) -> list[float] | str:
    if encoding_format == "base64":
        return base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode()
    return vec


class BodyLimit:
    """Rejects request bodies over the limit with 413, with or without a Content-Length."""

    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes
        self.message = f"request body is over the limit of {max_bytes} bytes"

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            return await error_response(413, self.message, code="request_too_large")(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                raise HTTPException(413, self.message)  # FastAPI re-raises HTTPExceptions from body reading
            return message

        await self.app(scope, limited_receive, send)


def create_app(registry: Registry, backend: BackendClient, api_keys: set[str]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await backend.aclose()

    app = FastAPI(title="Embedder Gateway", lifespan=lifespan)
    limits = registry.limits
    # Share each Checkpoint's slots across requests; acquire before serializing repeated query images.
    backend_slots = {
        name: asyncio.Semaphore(max(1, int(checkpoint.args["parallel"])))
        for name, checkpoint in registry.checkpoints.items()
    }
    # Added before bearer_auth so it runs inside it: a bad key is a 401 whatever the body size.
    app.add_middleware(BodyLimit, max_bytes=limits.max_body_bytes)

    def served_model(name: str, kind: str) -> ServedModel:
        if name not in registry.served_models:
            raise APIError(
                404,
                f"The model {name!r} does not exist. Served models: {', '.join(registry.served_models)}",
                code="model_not_found",
            )
        model = registry.served_models[name]
        if model.format.kind != kind:
            raise APIError(
                400, f"{name} is a {model.format.kind} model; use {ROUTES[model.format.kind]}", code="invalid_request"
            )
        return model

    def check_modalities(model: ServedModel, items: list[tuple[str, Item]]) -> None:
        for where, item in items:
            given = {"text"} if item.text else set()
            given |= {"image"} if item.images else set()
            if missing := sorted(given - set(model.modalities)):
                raise APIError(400, f"{where}: {model.name} does not accept {' or '.join(missing)} input", code="invalid_request")

    async def embed_each(model: ServedModel, located: list[tuple[str, dict]], **options) -> list[dict]:
        async def embed(input: dict) -> dict:
            async with backend_slots[model.checkpoint.name]:
                return await backend.embed(model.checkpoint.name, [input], **options)

        # One Backend request per input: the Backend fails a batch without saying which input was at fault.
        results = await asyncio.gather(*(embed(i) for _, i in located), return_exceptions=True)
        for (where, _), r in zip(located, results):
            if isinstance(r, BackendError) and r.status == 400:
                raise APIError(400, f"{where}: {r.message}", code="invalid_input")
        for r in results:
            if isinstance(r, BaseException):
                raise r
        return results

    def usage(results: list[dict]) -> dict:
        tokens = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results)
        return {"prompt_tokens": tokens, "total_tokens": tokens}

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

    @app.exception_handler(InvalidItem)
    async def invalid_item(request: Request, e: InvalidItem):
        return error_response(400, str(e), code="invalid_request")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, e: HTTPException):
        return error_response(e.status_code, str(e.detail), code="request_too_large" if e.status_code == 413 else None)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [
            {
                "id": m.name, "object": "model", "created": 0, "owned_by": "embedder-gateway",
                "kind": m.format.kind, "modalities": list(m.modalities),
                **({"dimensions": m.dimensions, "max_dimensions": m.max_dimensions} if m.format.kind == "embedding" else {}),
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
        model = served_model(req.model, "embedding")
        dims = model.dimensions if req.dimensions is None else req.dimensions
        if not model.min_dimensions <= dims <= model.max_dimensions:
            raise APIError(
                400, f"dimensions: {model.name} supports {model.min_dimensions} to {model.max_dimensions}",
                code="invalid_request",
            )
        items = embedding_items(req.input, req.messages, limits.max_embedding_items, limits.max_image_bytes)
        check_modalities(model, items)

        instruction = model.instructions[resolve_role(req, model)]
        results = await embed_each(model, [
            (where, model.format.render(item, instruction if item.instruction is None else item.instruction))
            for where, item in items
        ])
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i,
                 "embedding": encode(l2_normalize(r["data"][0]["embedding"][:dims]), req.encoding_format)}
                for i, r in enumerate(results)
            ],
            "model": model.name,
            "usage": usage(results),
        }

    @app.post("/v1/rerank")
    async def rerank(req: RerankRequest):
        model = served_model(req.model, "reranking")
        items = rerank_items(req.query, req.documents, limits.max_rerank_documents, limits.max_image_bytes)
        check_modalities(model, items)

        (_, query), documents = items[0], items[1:]
        instruction = model.instructions["rerank"]
        # Rank pooling outputs [P("yes"), P("no")]; read it raw, not L2-normalized (ADR 0003).
        results = await embed_each(
            model, [(where, model.format.render(query, doc, instruction)) for where, doc in documents], embd_normalize=-1
        )
        scores = [r["data"][0]["embedding"][0] for r in results]
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])  # stable: ties keep document order
        return {
            "model": model.name,
            "results": [
                {"index": i, "relevance_score": scores[i],
                 # Text only: an image document's payload is never echoed back.
                 **({"document": {"text": documents[i][1].text}} if req.return_documents else {})}
                for i in ranked[:req.top_n]
            ],
            "usage": usage(results),
        }

    return app
