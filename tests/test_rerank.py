"""POST /v1/rerank: Cohere-style results computed from gateway-rendered Qwen3-VL-Reranker prompts (ADR 0003)."""

import asyncio
import base64

import httpx
import pytest
from conftest import AUTH
from test_multimodal import URI, data_uri, gateway_with, png

MODEL = "Qwen3-VL-Reranker-2B"
CHECKPOINT = "Qwen3-VL-Reranker-2B-Q8_0"
INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
# The official format_mm_instruction() rendered with the chat template (add_generation_prompt=True).
SYSTEM = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
TAIL = "<|im_end|>\n<|im_start|>assistant\n"


def prompt(query: str, document: str, instruction: str = INSTRUCTION) -> str:
    return f"{SYSTEM}<Instruct>: {instruction}<Query>:{query}\n<Document>:{document}{TAIL}"


def text(t: str) -> dict:
    return {"type": "text", "text": t}


def image(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def contents(backend) -> list[list[dict]]:
    return [i["content"] for body in backend.embedding_bodies for i in body["input"]]


def score_by_document(backend, scores: dict[str, float]):
    """The fake rank head answers [P(yes), P(no), <garbage>...] by which document the prompt holds."""
    def vector_for(input):
        rendered = "".join(p.get("text", "") for p in input["content"])
        document = rendered.rsplit("<Document>:", 1)[1].removesuffix(TAIL)
        p = scores.get(document, 0.5)
        return [p, 1 - p] + [7.0] * 2046
    backend.vector_for = vector_for


def rerank(gateway, **body):
    return gateway.post("/v1/rerank", headers=AUTH, json={"model": MODEL, **body})


def assert_400(r, *fragments: str):
    assert r.status_code == 400, r.text
    error = r.json()["error"]
    assert set(error) == {"message", "type", "code"}
    for fragment in fragments:
        assert fragment in error["message"]


FRANCE = "What is the capital of France?"
PARIS = "Paris is the capital and most populous city of France."
BANANAS = "Bananas are a yellow tropical fruit rich in potassium."
EIFFEL = "The Eiffel Tower is a famous landmark located in Paris, France."


def test_text_rerank_is_sorted_by_relevance_with_indexes_into_documents(gateway, backend):
    score_by_document(backend, {PARIS: 0.73, BANANAS: 0.27, EIFFEL: 0.51})
    r = rerank(gateway, query=FRANCE, documents=[BANANAS, PARIS, EIFFEL])

    assert r.status_code == 200, r.text
    assert r.json() == {
        "model": MODEL,
        "results": [
            {"index": 1, "relevance_score": 0.73},
            {"index": 2, "relevance_score": 0.51},
            {"index": 0, "relevance_score": 0.27},
        ],
        "usage": {"prompt_tokens": 21, "total_tokens": 21},
    }
    # One Backend request per document, rank head read raw (no L2 normalization).
    assert backend.embedding_bodies == [
        {"model": CHECKPOINT, "input": [{"content": [text(prompt(FRANCE, d))]}],
         "encoding_format": "float", "embd_normalize": -1}
        for d in [BANANAS, PARIS, EIFFEL]
    ]


@pytest.mark.parametrize("top_n, indexes", [(1, [1]), (2, [1, 2]), (3, [1, 2, 0]), (10, [1, 2, 0])])
def test_top_n_keeps_only_the_best_documents(gateway, backend, top_n, indexes):
    score_by_document(backend, {PARIS: 0.73, BANANAS: 0.27, EIFFEL: 0.51})
    r = rerank(gateway, query=FRANCE, documents=[BANANAS, PARIS, EIFFEL], top_n=top_n)

    assert r.status_code == 200, r.text
    assert [x["index"] for x in r.json()["results"]] == indexes
    assert len(backend.embedding_bodies) == 3  # every document is scored


def test_equal_scores_keep_document_order(gateway, backend):
    score_by_document(backend, {"a": 0.5, "b": 0.9, "c": 0.5, "d": 0.5})
    r = rerank(gateway, query="q", documents=["a", "b", "c", "d"])

    assert [x["index"] for x in r.json()["results"]] == [1, 0, 2, 3]


def test_return_documents_echoes_the_document_text(gateway, backend):
    score_by_document(backend, {PARIS: 0.73, BANANAS: 0.27})
    r = rerank(gateway, query=FRANCE, documents=[BANANAS, {"text": PARIS}], return_documents=True, top_n=2)

    assert r.status_code == 200, r.text
    assert r.json()["results"] == [
        {"index": 1, "relevance_score": 0.73, "document": {"text": PARIS}},
        {"index": 0, "relevance_score": 0.27, "document": {"text": BANANAS}},
    ]


@pytest.mark.parametrize("extra", [{}, {"return_documents": False}])
def test_documents_are_not_echoed_unless_asked(gateway, backend, extra):
    score_by_document(backend, {})
    r = rerank(gateway, query=FRANCE, documents=[PARIS], **extra)

    assert r.status_code == 200, r.text
    assert r.json()["results"] == [{"index": 0, "relevance_score": 0.5}]


def test_image_documents_never_echo_their_payload(gateway, backend):
    raw = base64.b64encode(png(2)).decode()
    r = rerank(gateway, query="a red pixel", return_documents=True,
               documents=[{"image": URI}, {"image": raw, "text": "caption"}])

    assert r.status_code == 200, r.text
    assert sorted((x["index"], x["document"]) for x in r.json()["results"]) == [
        (0, {"text": ""}), (1, {"text": "caption"}),
    ]
    assert URI.split(",")[1] not in r.text and raw not in r.text


def test_weknora_jina_style_image_body_is_scored_one_request_per_image(gateway, backend):
    # WeKnora's image_wire_test.go body for a Cohere-shaped row with image_field "image".
    uris = [data_uri(png(n)) for n in (1, 2, 3)]
    by_image = {u: s for u, s in zip(uris, (0.1, 0.8, 0.4))}
    backend.vector_for = lambda input: [by_image[input["content"][1]["image_url"]["url"]], 0.0]
    r = gateway.post("/v1/rerank", headers=AUTH, json={
        "model": MODEL, "query": "which chart shows revenue", "return_documents": False, "top_n": 3,
        "documents": [{"image": u} for u in uris],
    })

    assert r.status_code == 200, r.text
    assert r.json()["results"] == [
        {"index": 1, "relevance_score": 0.8},
        {"index": 2, "relevance_score": 0.4},
        {"index": 0, "relevance_score": 0.1},
    ]
    # Official placement: the image right after "<Document>:", and no "NULL" for an image-only document.
    assert contents(backend) == [
        [text(f"{SYSTEM}<Instruct>: {INSTRUCTION}<Query>:which chart shows revenue\n<Document>:"), image(u), text(TAIL)]
        for u in uris
    ]


def test_image_query_and_text_plus_image_document_put_images_before_text(gateway, backend):
    raw = base64.b64encode(png(2)).decode()
    r = rerank(gateway, query={"image": URI, "text": "this animal"},
               documents=[{"text": "A llama.", "image": raw}, "A bowl of ramen."])

    assert r.status_code == 200, r.text
    head = f"{SYSTEM}<Instruct>: {INSTRUCTION}<Query>:"
    assert contents(backend) == [
        [text(head), image(URI), text("this animal\n<Document>:"), image(raw), text("A llama." + TAIL)],
        [text(head), image(URI), text("this animal\n<Document>:A bowl of ramen." + TAIL)],
    ]


def test_image_only_query(gateway, backend):
    rerank(gateway, query={"image": URI}, documents=["d"])

    assert contents(backend) == [
        [text(f"{SYSTEM}<Instruct>: {INSTRUCTION}<Query>:"), image(URI), text("\n<Document>:d" + TAIL)],
    ]


def test_empty_query_or_document_is_null_like_the_official_script(gateway, backend):
    rerank(gateway, query="", documents=["", {"text": ""}, "d"])

    assert [c[0]["text"] for c in contents(backend)] == [
        prompt("NULL", "NULL"), prompt("NULL", "NULL"), prompt("NULL", "d"),
    ]


def test_the_served_models_instruction_from_yaml_is_rendered(tmp_path, backend):
    def edit(raw):
        raw["served_models"][MODEL]["instructions"] = {"rerank": "Find the product the shopper describes."}

    with gateway_with(tmp_path, backend, edit) as gateway:
        r = rerank(gateway, query="red shoes", documents=["Crimson sneakers"])

    assert r.status_code == 200, r.text
    assert contents(backend) == [[text(prompt("red shoes", "Crimson sneakers", "Find the product the shopper describes."))]]


def test_unknown_instruction_key_is_a_config_error(tmp_path):
    import yaml
    from conftest import CONFIG

    from gateway.config import load_registry

    raw = yaml.safe_load(CONFIG.read_text())
    raw["served_models"][MODEL]["instructions"] = {"query": "x"}
    config = tmp_path / "gateway.yaml"
    config.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="instructions"):
        load_registry(config)


def test_vendor_knobs_are_accepted_and_not_forwarded(gateway, backend):
    r = rerank(gateway, query="q", documents=["d"], truncate_prompt_tokens=511, truncate="END")

    assert r.status_code == 200, r.text
    assert set(backend.embedding_bodies[0]) == {"model", "input", "encoding_format", "embd_normalize"}


def test_an_embedding_model_on_rerank_is_400(gateway, backend):
    r = gateway.post("/v1/rerank", headers=AUTH, json={"model": "Qwen3-VL-Embedding-2B", "query": "q", "documents": ["d"]})

    assert_400(r, "Qwen3-VL-Embedding-2B", "/v1/embeddings")
    assert backend.requests == []


def test_a_reranking_model_on_embeddings_is_400(gateway, backend):
    r = gateway.post("/v1/embeddings", headers=AUTH, json={"model": MODEL, "input": "x"})

    assert_400(r, MODEL, "/v1/rerank")
    assert backend.requests == []


def test_unknown_rerank_model_is_404(gateway, backend):
    r = gateway.post("/v1/rerank", headers=AUTH, json={"model": "nope", "query": "q", "documents": ["d"]})

    assert r.status_code == 404, r.text
    assert MODEL in r.json()["error"]["message"]


def test_100_documents_are_accepted_and_101_is_400(gateway, backend):
    assert rerank(gateway, query="q", documents=["d"] * 100).status_code == 200
    backend.requests.clear()

    assert_400(rerank(gateway, query="q", documents=["d"] * 101), "documents", "101", "100")
    assert backend.requests == []


@pytest.mark.parametrize("body, fragments", [
    ({"documents": ["d"]}, ["query"]),
    ({"query": "q"}, ["documents"]),
    ({"query": None, "documents": ["d"]}, ["query"]),
    ({"query": 42, "documents": ["d"]}, ["query"]),
    ({"query": "q", "documents": "d"}, ["documents"]),
    ({"query": "q", "documents": []}, ["documents", "empty"]),
    ({"query": "q", "documents": ["d", 7]}, ["documents[1]"]),
    ({"query": "q", "documents": ["d", {"txt": "typo"}]}, ["documents[1]", "txt"]),
    ({"query": "q", "documents": ["d", {"image": "https://example.com/a.png"}]}, ["documents[1].image", "URL"]),
    ({"query": "q", "documents": [{"video": "data:video/mp4;base64,AAAA"}]}, ["documents[0]", "video"]),
    ({"query": "q", "documents": [{"image": "data:image/png;base64,bm90IGFuIGltYWdl"}]}, ["documents[0].image"]),
    ({"query": {"image": "file:///etc/passwd"}, "documents": ["d"]}, ["query.image"]),
    ({"query": {"text": 1}, "documents": ["d"]}, ["query.text"]),
    ({"query": "q", "documents": ["d"], "top_n": 0}, ["top_n"]),
    ({"query": "q", "documents": ["d"], "top_n": "all"}, ["top_n"]),
])
def test_bad_requests_are_400_naming_the_field(gateway, backend, body, fragments):
    assert_400(rerank(gateway, **body), *fragments)
    assert backend.requests == []


def test_an_image_for_a_text_only_reranker_is_400_naming_the_item(tmp_path, backend):
    def edit(raw):
        raw["served_models"][MODEL]["modalities"] = ["text"]

    with gateway_with(tmp_path, backend, edit) as gateway:
        assert_400(rerank(gateway, query="q", documents=["d", {"image": URI}]), "documents[1]", "image")
        assert_400(rerank(gateway, query={"image": URI}, documents=["d"]), "query", "image")
    assert backend.requests == []


def test_an_over_long_document_is_400_naming_it(gateway, backend):
    backend.reject = lambda input: "input (9000 tokens) is too large" if "long" in input["content"][0]["text"] else None
    r = rerank(gateway, query="q", documents=["short", "long"])

    assert_400(r, "documents[1]", "9000 tokens")


@pytest.mark.parametrize("parallel", [2, 4])
def test_image_query_fanout_is_bounded_across_requests_and_recovers_after_errors(backend, parallel):
    from conftest import CONFIG

    from gateway.app import create_app
    from gateway.backend import BackendClient
    from gateway.config import load_registry

    async def run():
        active = peak = 0

        async def handle(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.001)
                return backend.handle(request)
            finally:
                active -= 1

        client = BackendClient("http://backend", "internal-key", transport=httpx.MockTransport(handle))
        registry = load_registry(CONFIG)
        registry.checkpoints[CHECKPOINT].args["parallel"] = parallel
        app = create_app(registry, client, {"test-key"})
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as gateway:
                body = {"model": MODEL, "query": {"image": URI}, "documents": ["d"] * 100}
                responses = await asyncio.gather(*(
                    gateway.post("/v1/rerank", headers=AUTH, json=body) for _ in range(2)
                ))
                assert all(r.status_code == 200 and len(r.json()["results"]) == 100 for r in responses)
                assert len(backend.embedding_bodies) == 200
                assert 1 < peak <= parallel

                backend.embeddings_response = httpx.Response(500, json={"error": {"message": "boom"}})
                failed = await gateway.post("/v1/rerank", headers=AUTH, json=body)
                assert failed.status_code == 502
                backend.embeddings_response = None
                recovered = await asyncio.wait_for(
                    gateway.post("/v1/rerank", headers=AUTH, json=body), timeout=5
                )
                assert recovered.status_code == 200
                assert active == 0
                assert peak <= parallel
        finally:
            await client.aclose()

    asyncio.run(run())


def test_backend_failure_is_502(gateway, backend):
    backend.embeddings_response = httpx.Response(500, json={"error": {"code": 500, "message": "boom", "type": "server_error"}})

    assert rerank(gateway, query="q", documents=["d"]).status_code == 502
