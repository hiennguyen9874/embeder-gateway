"""Prompt Formats: turn a client's Item into the exact Backend input a Checkpoint was trained on."""

import unicodedata
from dataclasses import dataclass
from typing import Callable

from gateway.items import Item


@dataclass(frozen=True)
class PromptFormat:
    kind: str  # "embedding" or "reranking"
    # Default Instruction per key: the Input Roles for embedding, "rerank" for reranking.
    instructions: dict[str, str]
    # One Backend /v1/embeddings input. Embedding: (Item, Instruction); reranking: (query, document, Instruction).
    render: Callable[..., dict]


def _qwen3_vl_embedding(item: Item, instruction: str) -> dict:
    # Mirrors format_model_input() + chat_template.jinja (add_generation_prompt=True) from
    # Qwen3-VL-Embedding-2B/scripts/qwen3_vl_embedding.py: images come before the text in the user
    # turn. The Backend wraps each image_url part in <|vision_start|>...<|vision_end|> itself.
    instruction = instruction.strip()
    if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
        instruction += "."
    head = f"<|im_start|>system\n{instruction or "Represent the user's input."}<|im_end|>\n<|im_start|>user\n"
    tail = "<|im_end|>\n<|im_start|>assistant\n"
    if not item.images:
        return {"content": [{"type": "text", "text": f"{head}{item.text or 'NULL'}{tail}"}]}
    return {"content": [
        {"type": "text", "text": head},
        *({"type": "image_url", "image_url": {"url": url}} for url in item.images),
        {"type": "text", "text": item.text + tail},
    ]}


def _qwen3_vl_reranker(query: Item, document: Item, instruction: str) -> dict:
    # Mirrors format_mm_instruction() + format_mm_content() + chat_template.jinja (add_generation_prompt=True)
    # from Qwen3-VL-Reranker-2B/scripts/qwen3_vl_reranker.py. The Instruction is used verbatim, and in both the
    # Query and the Document images come before the text; one with neither is "NULL".
    parts = [
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
        'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
        f"<|im_start|>user\n<Instruct>: {instruction}"
    ]
    for prefix, item in (("<Query>:", query), ("\n<Document>:", document)):
        parts.append(prefix)
        parts += ({"type": "image_url", "image_url": {"url": url}} for url in item.images)
        parts.append(item.text or ("" if item.images else "NULL"))
    parts.append("<|im_end|>\n<|im_start|>assistant\n")
    # Adjacent text is merged: the Backend concatenates text parts anyway.
    content = []
    for part in parts:
        if isinstance(part, dict):
            content.append(part)
        elif content and content[-1]["type"] == "text":
            content[-1]["text"] += part
        else:
            content.append({"type": "text", "text": part})
    return {"content": content}


FORMATS = {
    "qwen3-vl-embedding": PromptFormat(
        kind="embedding",
        instructions={
            "document": "Represent the user's input.",
            "query": "Retrieve relevant documents for the query.",
        },
        render=_qwen3_vl_embedding,
    ),
    "qwen3-vl-reranker": PromptFormat(
        kind="reranking",
        instructions={"rerank": "Given a search query, retrieve relevant candidates that answer the query."},
        render=_qwen3_vl_reranker,
    ),
}
