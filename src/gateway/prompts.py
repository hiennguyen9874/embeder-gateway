"""Prompt Formats: turn a client's Item into the exact Backend input a Checkpoint was trained on."""

import unicodedata
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PromptFormat:
    kind: str  # "embedding" or "reranking"
    instructions: dict[str, str]  # default Instruction per Input Role
    render: Callable[[str, str], dict]  # (text, instruction) -> one Backend /v1/embeddings input


def _qwen3_vl_embedding(text: str, instruction: str) -> dict:
    # Mirrors format_model_input() + chat_template.jinja (add_generation_prompt=True) from
    # Qwen3-VL-Embedding-2B/scripts/qwen3_vl_embedding.py.
    instruction = instruction.strip()
    if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
        instruction += "."
    prompt = (
        f"<|im_start|>system\n{instruction or "Represent the user's input."}<|im_end|>\n"
        f"<|im_start|>user\n{text or 'NULL'}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return {"content": [{"type": "text", "text": prompt}]}


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
