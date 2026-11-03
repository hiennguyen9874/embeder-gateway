"""Config/Registry: loads gateway.yaml into Checkpoints and Served Models."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from gateway.prompts import FORMATS, PromptFormat

ROLES = ("document", "query")
MODALITIES = ("text", "image")
DEFAULT_ARGS = {"parallel": 4, "ctx-size": 32768}
DEFAULT_LIMITS = {"max_embedding_items": 64, "max_rerank_documents": 100, "max_image_mb": 10, "max_body_mb": 50}
MB = 1024 * 1024


@dataclass(frozen=True)
class Checkpoint:
    name: str  # also the router's model id
    repo: str  # Hugging Face <org>/<repo>
    model: str
    mmproj: str | None
    args: dict  # llama-server flags for this Checkpoint's preset
    dir: Path  # models_dir/<org>--<repo>

    @property
    def files(self) -> list[str]:
        return [f for f in (self.model, self.mmproj) if f]


@dataclass(frozen=True)
class ServedModel:
    name: str
    checkpoint: Checkpoint
    format: PromptFormat
    # Embedding only (None for reranking):
    dimensions: int | None  # returned when the request has no `dimensions`
    min_dimensions: int | None  # smallest MRL truncation
    max_dimensions: int | None  # the Checkpoint's native size
    modalities: tuple[str, ...]
    role: str  # Input Role used when the request carries none
    instructions: dict[str, str]  # keyed like the Prompt Format's defaults


@dataclass(frozen=True)
class Limits:
    max_embedding_items: int
    max_rerank_documents: int
    max_image_bytes: int
    max_body_bytes: int


@dataclass(frozen=True)
class Registry:
    checkpoints: dict[str, Checkpoint]
    served_models: dict[str, ServedModel]
    backend_url: str | None  # external router; None means the gateway starts one
    llama_server: Path
    limits: Limits


def load_registry(path: str | Path) -> Registry:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    base = path.parent.resolve()
    models_dir = base / raw.get("models_dir", "models")

    checkpoints = {}
    for name, c in raw["checkpoints"].items():
        if ":" in name:
            raise ValueError(f"checkpoint {name!r}: ':' is not allowed (the router rewrites it as a quant tag)")
        org, _, repo = c["repo"].partition("/")
        checkpoints[name] = Checkpoint(
            name=name,
            repo=c["repo"],
            model=c["model"],
            mmproj=c.get("mmproj"),
            args={**DEFAULT_ARGS, **c.get("args", {})},
            dir=models_dir / f"{org}--{repo}",
        )

    served = {}
    for name, m in raw["served_models"].items():
        if m["checkpoint"] not in checkpoints:
            raise ValueError(f"served model {name!r}: unknown checkpoint {m['checkpoint']!r}")
        if m["format"] not in FORMATS:
            raise ValueError(f"served model {name!r}: unknown format {m['format']!r}; known: {sorted(FORMATS)}")
        fmt = FORMATS[m["format"]]
        role = m.get("role", "document")
        if role not in ROLES:
            raise ValueError(f"served model {name!r}: role must be one of {ROLES}")
        if not set(m["modalities"]) <= set(MODALITIES):
            raise ValueError(f"served model {name!r}: modalities must be among {MODALITIES}")
        instructions = m.get("instructions", {})
        if unknown := sorted(set(instructions) - set(fmt.instructions)):
            raise ValueError(f"served model {name!r}: unknown instructions {unknown}; {m['format']} has {sorted(fmt.instructions)}")
        max_dims = min_dims = dims = None
        if fmt.kind == "embedding":
            max_dims = m["max_dimensions"]
            min_dims = m.get("min_dimensions", max_dims)
            dims = m.get("dimensions", max_dims)
            if not 0 < min_dims <= dims <= max_dims:
                raise ValueError(f"served model {name!r}: needs 0 < min_dimensions <= dimensions <= max_dimensions")
        served[name] = ServedModel(
            name=name,
            checkpoint=checkpoints[m["checkpoint"]],
            format=fmt,
            dimensions=dims,
            min_dimensions=min_dims,
            max_dimensions=max_dims,
            modalities=tuple(m["modalities"]),
            role=role,
            instructions={**fmt.instructions, **instructions},
        )

    backend = raw.get("backend", {})
    limits = {**DEFAULT_LIMITS, **raw.get("limits", {})}
    return Registry(
        checkpoints=checkpoints,
        served_models=served,
        backend_url=backend.get("url"),
        llama_server=base / backend.get("llama_server", "llama.cpp/llama-server"),
        limits=Limits(
            max_embedding_items=limits["max_embedding_items"],
            max_rerank_documents=limits["max_rerank_documents"],
            max_image_bytes=int(limits["max_image_mb"] * MB),
            max_body_bytes=int(limits["max_body_mb"] * MB),
        ),
    )
