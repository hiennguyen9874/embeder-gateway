"""Backend Client: a thin async client for the llama-server router."""

import httpx


class BackendError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class BackendClient:
    def __init__(self, base_url: str, api_key: str, transport: httpx.AsyncBaseTransport | None = None):
        self.http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
            timeout=httpx.Timeout(300, connect=5),
        )

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            r = await self.http.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise BackendError(503, f"backend unavailable: {e!r}") from e
        if r.status_code != 200:
            try:
                message = r.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                message = r.text
            raise BackendError(r.status_code, message)
        return r.json()

    async def embed(self, model: str, inputs: list, embd_normalize: int | None = None) -> dict:
        """POST /v1/embeddings; returns the OpenAI-shaped response. `embd_normalize` overrides the preset's."""
        body = {"model": model, "input": inputs, "encoding_format": "float"}
        if embd_normalize is not None:
            body["embd_normalize"] = embd_normalize
        return await self._request("POST", "/v1/embeddings", json=body)

    async def model_statuses(self) -> dict[str, str]:
        """Router model id -> status ("loaded", "loading", "unloaded", ...)."""
        body = await self._request("GET", "/models")
        return {m["id"]: m.get("status", {}).get("value", "unknown") for m in body["data"]}

    async def aclose(self) -> None:
        await self.http.aclose()
