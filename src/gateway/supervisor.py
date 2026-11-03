"""Backend Supervisor: runs the llama-server router as a child process, or uses an external one."""

import os
import secrets
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

from gateway.config import Registry
from gateway.presets import render_preset


class RouterError(RuntimeError):
    pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_loaded(proc: subprocess.Popen, url: str, key: str, names: list[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    statuses: dict = {}
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RouterError(f"llama-server router exited with code {proc.returncode}")
        try:
            r = httpx.get(f"{url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=5)
            statuses = {m["id"]: m.get("status", {}) for m in r.json()["data"]} if r.status_code == 200 else {}
        except (httpx.HTTPError, ValueError, KeyError):
            statuses = {}
        failed = [n for n in names if statuses.get(n, {}).get("failed")]
        if failed:
            raise RouterError(f"router failed to load {failed}: {[statuses[n] for n in failed]}")
        if all(statuses.get(n, {}).get("value") == "loaded" for n in names):
            return
        time.sleep(1)
    raise RouterError(f"router not ready after {timeout:.0f}s: { {n: statuses.get(n) for n in names} }")


@contextmanager
def run_backend(registry: Registry, ready_timeout: float = 600) -> Iterator[tuple[str, str]]:
    """Yields the Backend's (base URL, API key) for as long as the context is open."""
    if registry.backend_url:
        yield registry.backend_url, os.environ.get("GATEWAY_BACKEND_API_KEY", "")
        return

    with tempfile.TemporaryDirectory(prefix="embedder-gateway-") as tmp:
        preset = Path(tmp) / "models-preset.ini"
        preset.write_text(render_preset(registry))
        key = secrets.token_urlsafe(32)
        # A key file, not --api-key or LLAMA_API_KEY: argv is world-readable and model instances inherit the env.
        key_file = Path(tmp) / "api-key"
        key_file.touch(mode=0o600)
        key_file.write_text(key + "\n")
        port = _free_port()
        url = f"http://127.0.0.1:{port}"
        proc = subprocess.Popen([
            str(registry.llama_server),
            "--models-preset", str(preset),
            "--models-max", str(len(registry.checkpoints)),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--api-key-file", str(key_file),
        ])
        try:
            _wait_until_loaded(proc, url, key, list(registry.checkpoints), ready_timeout)
            yield url, key
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
