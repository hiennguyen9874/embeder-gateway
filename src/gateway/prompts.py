"""Prompt Formats: turn a client's Item into the exact Backend input a Checkpoint was trained on."""

import unicodedata
from dataclasses import dataclass
from typing import Callable

from gateway.items import Item


@dataclass(frozen=True)
class PromptFormat:
    kind: str  # "embedding" or "reranking"
    instructions: dict[str, str]  # default Instruction per Input Role
    render: Callable[[Item, str], dict]  # (Item, Instruction) -> one Backend /v1/embeddings input


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


FORMATS = {
    "qwen3-vl-embedding": PromptFormat(
        kind="embedding",
        instructions={
            "document": "Represent the user's input.",
            "query": "Retrieve relevant documents for the query.",
        },
        render=_qwen3_vl_embedding,
    ),
}
