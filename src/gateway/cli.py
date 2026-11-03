"""`gateway` entry point: `serve` (default) and `download`."""

import argparse
import os
import signal
import sys
from pathlib import Path

import uvicorn

from gateway.app import create_app
from gateway.backend import BackendClient
from gateway.config import load_registry
from gateway.download import download
from gateway.supervisor import run_backend


def load_api_keys() -> set[str]:
    """Keys from GATEWAY_API_KEYS (comma-separated) and GATEWAY_API_KEY_FILE (one per line, # comments)."""
    keys = {k.strip() for k in os.environ.get("GATEWAY_API_KEYS", "").split(",")}
    if path := os.environ.get("GATEWAY_API_KEY_FILE"):
        keys |= {line.strip() for line in Path(path).read_text().splitlines() if not line.strip().startswith("#")}
    return keys - {""}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gateway", description="Embedder Gateway")
    parser.add_argument("command", nargs="?", choices=["serve", "download"], default="serve")
    parser.add_argument("--config", default=os.environ.get("GATEWAY_CONFIG", "gateway.yaml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    registry = load_registry(args.config)

    if args.command == "download":
        download(registry)
        return

    api_keys = load_api_keys()
    if not api_keys:
        sys.exit("gateway: no API keys; set GATEWAY_API_KEYS or GATEWAY_API_KEY_FILE")
    # Turn SIGTERM into SystemExit so the router child is stopped even while we wait for it to load.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    with run_backend(registry) as (url, key):
        app = create_app(registry, BackendClient(url, key), api_keys)
        uvicorn.run(app, host=args.host, port=args.port)
