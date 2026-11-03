"""Items, vLLM `messages` (WeKnora's wire formats), MRL dimensions, base64 and limits."""

import base64
import math
import struct
from contextlib import contextmanager

import httpx
import pytest
import yaml
from conftest import AUTH, BACKEND_KEY, CONFIG
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.backend import BackendClient
from gateway.config import load_registry

MODEL = "Qwen3-VL-Embedding-2B"
DOCUMENT = "Represent the user's input."
QUERY = "Retrieve relevant documents for the query."
HEAD = "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n"


def png(n: int) -> bytes:
    # n bytes, so each image is distinguishable (the gateway never decodes the picture itself).
    return bytes([0x89]) * n


def data_uri(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


URI = data_uri(png(1))  # "data:image/png;base64,iQ==", as in WeKnora's tests


def text_content(instruction: str, text: str) -> list[dict]:
    return [{"type": "text", "text": HEAD.format(instruction) + text + TAIL}]


def image_content(instruction: str, url: str, text: str = "") -> list[dict]:
    # Official placement: the image first, then the text, in the user turn.
    return [
        {"type": "text", "text": HEAD.format(instruction)},
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": text + TAIL},
    ]


def backend_contents(backend) -> list[list[dict]]:
    return [i["content"] for body in backend.embedding_bodies for i in body["input"]]


def one_hot(i: int) -> list[float]:
    return [5.0 if j == i else 0.0 for j in range(2048)]


def hot_index(vec: list[float]) -> int:
    return vec.index(1.0)


@contextmanager
def gateway_with(tmp_path, backend, edit):
    raw = yaml.safe_load(CONFIG.read_text())
    edit(raw)
    config = tmp_path / "gateway.yaml"
    config.write_text(yaml.safe_dump(raw))
    client = BackendClient("http://backend", BACKEND_KEY, transport=httpx.MockTransport(backend.handle))
    with TestClient(create_app(load_registry(config), client, {"test-key"})) as c:
        yield c


def embed(gateway, **body):
    return gateway.post("/v1/embeddings", headers=AUTH, json={"model": MODEL, **body})


def assert_400(r, *fragments: str):
    assert r.status_code == 400, r.text
    message = r.json()["error"]["message"]
    for fragment in fragments:
        assert fragment in message, message


# --- input Items ---

@pytest.mark.parametrize("item, content", [
    ({"text": "a red bicycle"}, text_content(DOCUMENT, "a red bicycle")),
    ({"image": URI}, image_content(DOCUMENT, URI)),
    ({"text": "a red bicycle", "image": URI}, image_content(DOCUMENT, URI, "a red bicycle")),
    ({"image": "iQ=="}, image_content(DOCUMENT, "iQ==")),  # raw base64 is forwarded as sent
    ({"image": "data:image/jpeg;base64,/9j/"}, image_content(DOCUMENT, "data:image/jpeg;base64,/9j/")),
    ({}, text_content(DOCUMENT, "NULL")),  # empty Item, as in the official script
])
def test_input_items_render_the_official_prompt_with_images_before_text(gateway, backend, item, content):
    r = embed(gateway, input=[item])

    assert r.status_code == 200, r.text
    assert len(r.json()["data"]) == 1
    assert backend_contents(backend) == [content]


def test_strings_and_items_can_be_mixed_and_keep_their_order(gateway, backend):
    backend.vector_for = lambda input: one_hot(len(input["content"]))
    r = embed(gateway, input=["text", {"image": URI}, {"text": "both", "image": URI}])

    assert r.status_code == 200, r.text
    assert [(d["index"], hot_index(d["embedding"])) for d in r.json()["data"]] == [(0, 1), (1, 3), (2, 3)]
    assert backend_contents(backend) == [
        text_content(DOCUMENT, "text"), image_content(DOCUMENT, URI), image_content(DOCUMENT, URI, "both"),
    ]


def test_each_item_is_its_own_backend_request(gateway, backend):
    embed(gateway, input=["a", "b", {"image": URI}])

    assert [len(body["input"]) for body in backend.embedding_bodies] == [1, 1, 1]
    assert {body["model"] for body in backend.embedding_bodies} == {"Qwen3-VL-Embedding-2B-Q8_0"}


@pytest.mark.parametrize("body, fragments", [
    ({"input": [{"image": "https://example.com/cat.png"}]}, ["input[0].image", "URLs are not supported"]),
    ({"input": [{"image": "http://169.254.169.254/latest"}]}, ["input[0].image", "URLs are not supported"]),
    ({"input": [{"image": "file:///etc/passwd"}]}, ["input[0].image"]),
    ({"input": ["ok", {"video": "data:video/mp4;base64,AAAA"}]}, ["input[1]", "video is not supported"]),
    ({"input": [{"image": "data:video/mp4;base64,AAAA"}]}, ["input[0].image", "video is not supported"]),
    ({"input": [{"image": "data:text/plain;base64,AAAA"}]}, ["input[0].image", "text/plain"]),
    ({"input": [{"image": "data:image/png,rawbytes"}]}, ["input[0].image", "base64"]),
    ({"input": [{"image": "not base64!"}]}, ["input[0].image"]),
    ({"input": [{"image": ""}]}, ["input[0].image"]),
    ({"input": [{"image": 42}]}, ["input[0].image"]),
    ({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]}]},
     ["messages[0].content[0].image_url.url", "URLs are not supported"]),
    ({"messages": [{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}]}]},
     ["messages[0].content[0]", "video is not supported"]),
    ({"messages": [[{"role": "user", "content": "ok"}], [{"role": "user", "content": [
        {"type": "input_video", "input_video": {"data": "AAAA"}}]}]]},
     ["messages[1][0].content[0]", "video is not supported"]),
    ({"messages": [{"role": "system", "content": [{"type": "image_url", "image_url": {"url": URI}}]},
                   {"role": "user", "content": "x"}]},
     ["messages[0].content[0]", "only contain text"]),
])
def test_remote_urls_video_and_bad_images_are_rejected_naming_the_item(gateway, backend, body, fragments):
    assert_400(embed(gateway, **body), *fragments)
    assert backend.embedding_bodies == []


# --- vLLM messages, as WeKnora's generic provider sends them for *qwen3-vl-embedding* ---
# Bodies from references/WeKnora/internal/models/embedding/image_wire_test.go
# (TestSelfHostedQwen3VLEmbeddingSendsTextAndImagesThroughTheTemplate) and
# internal/models/api/openaiembeddings/golden_test.go.

WEKNORA = {"model": MODEL, "encoding_format": "float", "truncate_prompt_tokens": 511, "add_special_tokens": True}


def user(*parts: dict) -> list[dict]:
    return [{"role": "user", "content": list(parts)}]


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def image_part(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def test_weknora_text_batch_returns_one_vector_per_conversation_in_order(gateway, backend):
    backend.vector_for = lambda input: one_hot(len(input["content"][0]["text"]))
    r = gateway.post("/v1/embeddings", headers=AUTH, json={
        **WEKNORA, "messages": [user(text_part("ab")), user(text_part("c"))],
    })

    assert r.status_code == 200, r.text
    lengths = [len(HEAD.format(DOCUMENT) + t + TAIL) for t in ("ab", "c")]
    assert [(d["index"], hot_index(d["embedding"])) for d in r.json()["data"]] == [(0, lengths[0]), (1, lengths[1])]
    assert backend_contents(backend) == [text_content(DOCUMENT, "ab"), text_content(DOCUMENT, "c")]
    assert all(set(body) == {"model", "input", "encoding_format"} for body in backend.embedding_bodies)


def test_weknora_text_batch_with_input_type_field_uses_the_query_instruction(gateway, backend):
    r = gateway.post("/v1/embeddings", headers=AUTH, json={
        **WEKNORA, "input_type": "query", "messages": [user(text_part("ab")), user(text_part("c"))],
    })

    assert r.status_code == 200, r.text
    assert backend_contents(backend) == [text_content(QUERY, "ab"), text_content(QUERY, "c")]


def test_weknora_image_batch_returns_one_vector_per_conversation_in_order(gateway, backend):
    uris = [data_uri(png(n)) for n in range(1, 6)]
    backend.vector_for = lambda input: one_hot(uris.index(input["content"][1]["image_url"]["url"]))
    r = gateway.post("/v1/embeddings", headers=AUTH, json={
        **WEKNORA, "messages": [user(image_part(u)) for u in uris],
    })

    assert r.status_code == 200, r.text
    assert [(d["index"], hot_index(d["embedding"])) for d in r.json()["data"]] == [(i, i) for i in range(5)]
    assert backend_contents(backend) == [image_content(DOCUMENT, u) for u in uris]


def test_weknora_single_image_message_is_a_flat_conversation(gateway, backend):
    uri = data_uri(png(6))
    r = gateway.post("/v1/embeddings", headers=AUTH, json={**WEKNORA, "messages": user(image_part(uri))})

    assert r.status_code == 200, r.text
    assert [d["index"] for d in r.json()["data"]] == [0]
    assert backend_contents(backend) == [image_content(DOCUMENT, uri)]


def test_vllm_image_prompt_after_the_image_becomes_the_items_text(gateway, backend):
    # TestVLLMImageMessageMatchesItsExample: the image part, then a text part.
    r = gateway.post("/v1/embeddings", headers=AUTH, json={
        "model": MODEL, "encoding_format": "float", "truncate_prompt_tokens": 512,
        "messages": user(image_part(URI), text_part("Represent the given image.")),
    })

    assert r.status_code == 200, r.text
    assert backend_contents(backend) == [image_content(DOCUMENT, URI, "Represent the given image.")]


def test_text_parts_are_concatenated_and_images_still_come_first(gateway, backend):
    r = embed(gateway, messages=user(text_part("a photo "), image_part(URI), text_part("of a cat")))

    assert r.status_code == 200, r.text
    assert backend_contents(backend) == [image_content(DOCUMENT, URI, "a photo of a cat")]


def test_string_user_content_is_accepted(gateway, backend):
    r = embed(gateway, messages=[{"role": "user", "content": "plain"}])

    assert r.status_code == 200, r.text
    assert backend_contents(backend) == [text_content(DOCUMENT, "plain")]


# --- Instruction resolution ---

def test_a_system_message_replaces_the_instruction(gateway, backend):
    r = embed(gateway, input_type="query", messages=[
        [{"role": "system", "content": "  Find the matching product photo  "}, *user(image_part(URI))],
        [{"role": "system", "content": [text_part("Classify "), text_part("the sentiment!")]}, *user(text_part("great"))],
        user(text_part("no system message")),
    ])

    assert r.status_code == 200, r.text
    assert backend_contents(backend) == [
        image_content("Find the matching product photo.", URI),  # official punctuation rule applies
        text_content("Classify the sentiment!", "great"),
        text_content(QUERY, "no system message"),  # falls back to the Input Role's Instruction
    ]


def test_an_empty_system_message_falls_back_to_the_official_default(gateway, backend):
    embed(gateway, input_type="query", messages=[{"role": "system", "content": ""}, *user(text_part("x"))])

    assert backend_contents(backend) == [text_content(DOCUMENT, "x")]


# --- dimensions (MRL) and encodings ---

def test_dimensions_truncates_then_renormalizes(gateway, backend):
    backend.vector = [3.0, 4.0] + [0.0] * 62 + [1.0] * 1984
    r = embed(gateway, input=["x", {"image": URI}], dimensions=64)

    assert r.status_code == 200, r.text
    for d in r.json()["data"]:
        vec = d["embedding"]
        assert len(vec) == 64
        assert vec[:3] == [0.6, 0.8, 0.0]
        assert math.isclose(sum(x * x for x in vec), 1.0)


@pytest.mark.parametrize("dimensions", [64, 1000, 2048])
def test_every_dimension_in_the_mrl_range_is_accepted(gateway, backend, dimensions):
    backend.vector = [1.0] * 2048
    r = embed(gateway, input="x", dimensions=dimensions)

    assert r.status_code == 200, r.text
    vec = r.json()["data"][0]["embedding"]
    assert len(vec) == dimensions
    assert math.isclose(sum(x * x for x in vec), 1.0)


@pytest.mark.parametrize("dimensions", [0, -1, 63, 2049])
def test_dimensions_outside_the_mrl_range_is_400(gateway, backend, dimensions):
    assert_400(embed(gateway, input="x", dimensions=dimensions), "dimensions", "64", "2048")
    assert backend.embedding_bodies == []


def test_a_served_model_can_default_to_a_shorter_size(tmp_path, backend):
    def edit(raw):
        raw["served_models"][MODEL]["dimensions"] = 512

    with gateway_with(tmp_path, backend, edit) as gateway:
        default = embed(gateway, input="x")
        explicit = embed(gateway, input="x", dimensions=2048)
        models = gateway.get("/v1/models", headers=AUTH).json()["data"]

    assert len(default.json()["data"][0]["embedding"]) == 512
    assert len(explicit.json()["data"][0]["embedding"]) == 2048
    assert models[0]["dimensions"] == 512 and models[0]["max_dimensions"] == 2048


def test_base64_encoding_is_little_endian_float32(gateway, backend):
    backend.vector = [3.0, 4.0, -1.5] + [0.0] * 2045
    floats = embed(gateway, input=["x", "y"], dimensions=128).json()["data"]
    encoded = embed(gateway, input=["x", "y"], dimensions=128, encoding_format="base64").json()["data"]

    assert [d["index"] for d in encoded] == [0, 1]
    for f, b in zip(floats, encoded):
        raw = base64.b64decode(b["embedding"])
        assert len(raw) == 128 * 4
        decoded = struct.unpack("<128f", raw)
        assert decoded == pytest.approx(f["embedding"], rel=1e-6)


# --- Modalities ---

def test_an_image_for_a_text_only_served_model_is_400(tmp_path, backend):
    def edit(raw):
        raw["served_models"][MODEL]["modalities"] = ["text"]

    with gateway_with(tmp_path, backend, edit) as gateway:
        assert_400(embed(gateway, input=["fine", {"text": "caption", "image": URI}]), "input[1]", "image")
        assert_400(embed(gateway, messages=[user(text_part("fine")), user(image_part(URI))]), "messages[1]", "image")
        assert embed(gateway, input=["fine", {"text": "only text"}]).status_code == 200

    assert len(backend.embedding_bodies) == 2


# --- limits ---

def test_more_than_64_items_is_400(gateway, backend):
    assert embed(gateway, input=["x"] * 64).status_code == 200
    backend.requests.clear()

    assert_400(embed(gateway, input=["x"] * 65), "65", "64")
    assert_400(embed(gateway, messages=[user(text_part("x"))] * 65), "65", "64")
    assert backend.embedding_bodies == []


def test_an_image_over_the_size_limit_is_400(tmp_path, backend):
    def edit(raw):
        raw["limits"] = {"max_image_mb": 100 / (1024 * 1024)}  # 100 bytes

    with gateway_with(tmp_path, backend, edit) as gateway:
        assert embed(gateway, input=[{"image": data_uri(png(100))}]).status_code == 200
        assert_400(embed(gateway, input=["x", {"image": data_uri(png(101))}]), "input[1].image", "101", "100")
        assert_400(embed(gateway, messages=user(image_part(data_uri(png(101))))), "messages[0].content[0]", "101")


def test_default_limits_allow_a_10_mib_image(gateway, backend):
    assert embed(gateway, input=[{"image": data_uri(png(10 * 1024 * 1024))}]).status_code == 200
    assert_400(embed(gateway, input=[{"image": data_uri(png(10 * 1024 * 1024 + 1))}]), "input[0].image")


@pytest.fixture
def small_body_gateway(tmp_path, backend):
    def edit(raw):
        raw["limits"] = {"max_body_mb": 1000 / (1024 * 1024)}  # 1000 bytes

    with gateway_with(tmp_path, backend, edit) as gateway:
        yield gateway


def test_a_body_over_the_limit_is_413(small_body_gateway, backend):
    ok = small_body_gateway.post("/v1/embeddings", headers=AUTH, json={"model": MODEL, "input": "x" * 900})
    too_big = small_body_gateway.post("/v1/embeddings", headers=AUTH, json={"model": MODEL, "input": "x" * 1000})

    assert ok.status_code == 200, ok.text
    assert too_big.status_code == 413, too_big.text
    assert too_big.json()["error"]["code"] == "request_too_large"
    assert len(backend.embedding_bodies) == 1


def test_a_chunked_body_over_the_limit_is_413(small_body_gateway, backend):
    def chunks():
        yield b'{"model": "Qwen3-VL-Embedding-2B", "input": "'
        for _ in range(20):
            yield b"x" * 100
        yield b'"}'

    r = small_body_gateway.post("/v1/embeddings", headers={**AUTH, "Content-Type": "application/json"}, content=chunks())

    assert "content-length" not in {k.lower() for k in r.request.headers}
    assert r.status_code == 413, r.text
    assert r.json()["error"]["code"] == "request_too_large"
    assert backend.embedding_bodies == []


def test_a_bad_key_is_401_even_for_an_oversized_body(small_body_gateway):
    r = small_body_gateway.post("/v1/embeddings", json={"model": MODEL, "input": "x" * 5000})

    assert r.status_code == 401


# --- Backend errors ---

def test_an_over_long_item_is_400_naming_the_item(gateway, backend):
    backend.reject = lambda input: (
        "input (9020 tokens) is larger than the max context size (8192 tokens). skipping"
        if "very long" in input["content"][-1]["text"] else None
    )

    assert_400(embed(gateway, input=["short", "very long", {"image": URI, "text": "very long"}]),
               "input[1]: input (9020 tokens) is larger than the max context size")
    assert_400(embed(gateway, messages=[user(text_part("short")), user(text_part("very long"))]), "messages[1]: ")
    assert_400(embed(gateway, input="very long"), "input: ")
