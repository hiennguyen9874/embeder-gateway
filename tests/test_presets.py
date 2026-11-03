import configparser
from pathlib import Path

from conftest import CONFIG
from gateway.config import load_registry
from gateway.presets import render_preset


def test_preset_is_generated_from_yaml(tmp_path):
    config = tmp_path / "gateway.yaml"
    config.write_text("""
models_dir: weights
checkpoints:
  Emb-Q8_0:
    repo: org/Emb-GGUF
    model: Emb.Q8_0.gguf
    mmproj: Emb.mmproj-f16.gguf
    args: {embedding: true, pooling: last, embd-normalize: 2, device: CUDA1}
  Small:
    repo: other/Small-GGUF
    model: small.gguf
    args: {parallel: 1, flash-attn: "on", mmproj-offload: false}
served_models:
  emb:
    checkpoint: Emb-Q8_0
    format: qwen3-vl-embedding
    max_dimensions: 2048
    modalities: [text]
""")
    weights = tmp_path / "weights"

    assert render_preset(load_registry(config)) == f"""[Emb-Q8_0]
model = {weights}/org--Emb-GGUF/Emb.Q8_0.gguf
mmproj = {weights}/org--Emb-GGUF/Emb.mmproj-f16.gguf
load-on-startup = true
parallel = 4
ctx-size = 32768
embedding = true
pooling = last
embd-normalize = 2
device = CUDA1

[Small]
model = {weights}/other--Small-GGUF/small.gguf
load-on-startup = true
parallel = 1
ctx-size = 32768
flash-attn = on
mmproj-offload = false
"""


def test_shipped_config_preset_runs_qwen3_vl_embedding_as_a_last_pooled_normalized_embedder():
    ini = configparser.ConfigParser()
    ini.read_string(render_preset(load_registry(CONFIG)))

    preset = ini["Qwen3-VL-Embedding-2B-Q8_0"]
    checkpoint_dir = CONFIG.parent / "models" / "mradermacher--Qwen3-VL-Embedding-2B-GGUF"
    assert dict(preset) == {
        "model": str(checkpoint_dir / "Qwen3-VL-Embedding-2B.Q8_0.gguf"),
        "mmproj": str(checkpoint_dir / "Qwen3-VL-Embedding-2B.mmproj-f16.gguf"),
        "load-on-startup": "true",
        "embedding": "true",
        "pooling": "last",
        "embd-normalize": "2",
        "device": "CUDA0",
        "image-min-tokens": "4",
        "image-max-tokens": "1800",
        "parallel": "4",
        "ctx-size": "32768",
    }
    assert Path(preset["model"]).is_absolute()


def test_shipped_config_preset_runs_qwen3_vl_reranker_with_rank_pooling():
    ini = configparser.ConfigParser()
    ini.read_string(render_preset(load_registry(CONFIG)))

    checkpoint_dir = CONFIG.parent / "models" / "bealore--Qwen3-VL-Reranker-2B-GGUF"
    assert dict(ini["Qwen3-VL-Reranker-2B-Q8_0"]) == {
        "model": str(checkpoint_dir / "Qwen3-VL-Reranker-2B.Q8_0.gguf"),
        "mmproj": str(checkpoint_dir / "Qwen3-VL-Reranker-2B.mmproj-f16.gguf"),
        "load-on-startup": "true",
        "reranking": "true",
        "pooling": "rank",
        "device": "CUDA0",
        "image-min-tokens": "4",
        "image-max-tokens": "1280",
        "parallel": "4",
        "ctx-size": "32768",
    }
