"""Request Adapters: turn every accepted wire shape into a neutral list of Items."""

import base64
import binascii
import re
import warnings
from dataclasses import dataclass
from io import BytesIO

from PIL import Image


@dataclass(frozen=True)
class Item:
    text: str = ""
    images: tuple[str, ...] = ()  # data URIs or raw base64, forwarded to the Backend as sent
    instruction: str | None = None  # from a `system` message; overrides the Served Model's Instruction


class InvalidItem(ValueError):
    pass


_DATA_URI = re.compile(r"data:([^;,]*)((?:;[^;,]*)*),(.*)", re.DOTALL)
_VIDEO_PARTS = {"video", "video_url", "input_video"}


def image(value, where: str, max_bytes: int) -> str:
    """Validates one image (data URI or raw base64) and returns it unchanged."""
    if not isinstance(value, str) or not value:
        raise InvalidItem(f"{where}: must be a data URI or a base64 string")
    if value.startswith(("http://", "https://")):
        raise InvalidItem(f"{where}: image URLs are not supported; send a data URI or raw base64")
    payload = value
    if value.startswith("data:"):
        m = _DATA_URI.fullmatch(value)
        if not m:
            raise InvalidItem(f"{where}: malformed data URI")
        mime, params, payload = m.groups()
        if mime.startswith("video/"):
            raise InvalidItem(f"{where}: video is not supported")
        if not mime.startswith("image/"):
            raise InvalidItem(f"{where}: unsupported media type {mime!r}; expected image/*")
        if not params.endswith(";base64"):
            raise InvalidItem(f"{where}: data URI must be base64-encoded")
    try:
        decoded = base64.b64decode(payload, validate=True)
        size = len(decoded)
    except (binascii.Error, ValueError):
        raise InvalidItem(f"{where}: not a data URI or valid base64") from None
    if not size:
        raise InvalidItem(f"{where}: image is empty")
    if size > max_bytes:
        raise InvalidItem(f"{where}: image is {size} bytes, over the limit of {max_bytes}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            # Restrict Pillow to formats the Backend can decode as images, not its video fallback.
            formats = ("PNG", "JPEG", "GIF", "BMP", "WEBP", "PPM", "TGA", "PSD")
            with Image.open(BytesIO(decoded), formats=formats) as picture:
                picture.verify()
            with Image.open(BytesIO(decoded), formats=formats) as picture:
                picture.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise InvalidItem(f"{where}: must contain a valid supported image; video is not supported") from None
    return value


def input_item(value, where: str, max_image_bytes: int) -> Item:
    """A string or a Jina-style Item {"text"?, "image"?}."""
    if isinstance(value, str):
        return Item(text=value)
    if not isinstance(value, dict):
        raise InvalidItem(f'{where}: must be a string or an Item {{"text"?, "image"?}}')
    if "video" in value:
        raise InvalidItem(f"{where}: video is not supported")
    if unknown := sorted(set(value) - {"text", "image"}):
        raise InvalidItem(f'{where}: unknown field(s) {unknown}; an Item has "text" and/or "image"')
    text = value.get("text")
    if text is not None and not isinstance(text, str):
        raise InvalidItem(f"{where}.text: must be a string")
    img = value.get("image")
    return Item(text=text or "", images=() if img is None else (image(img, f"{where}.image", max_image_bytes),))


def _content(message: dict, where: str, max_image_bytes: int, allow_images: bool) -> tuple[str, tuple[str, ...]]:
    content = message.get("content")
    if isinstance(content, str):
        return content, ()
    if not isinstance(content, list):
        raise InvalidItem(f"{where}.content: must be a string or an array of content parts")
    texts, images = [], []
    for j, part in enumerate(content):
        w = f"{where}.content[{j}]"
        kind = part.get("type") if isinstance(part, dict) else None
        if not isinstance(kind, str):
            raise InvalidItem(f"{w}.type: must be a string")
        if kind == "text":
            if not isinstance(part.get("text"), str):
                raise InvalidItem(f"{w}.text: must be a string")
            texts.append(part["text"])
        elif kind == "image_url" and allow_images:
            url = part.get("image_url")
            images.append(image(url.get("url") if isinstance(url, dict) else None, f"{w}.image_url.url", max_image_bytes))
        elif kind in _VIDEO_PARTS:
            raise InvalidItem(f"{w}: video is not supported")
        elif kind == "image_url":
            raise InvalidItem(f"{w}: a system message may only contain text")
        else:
            raise InvalidItem(f"{w}: unsupported content part type {kind!r}; expected text or image_url")
    # Text parts are concatenated, as the chat template does.
    return "".join(texts), tuple(images)


def conversation(value, where: str, max_image_bytes: int) -> Item:
    """A vLLM `messages` conversation: an optional system message, then one user message."""
    if not isinstance(value, list) or not all(isinstance(m, dict) for m in value):
        raise InvalidItem(f"{where}: must be an array of messages")
    messages, instruction = value, None
    if messages and messages[0].get("role") == "system":
        instruction, _ = _content(messages[0], f"{where}[0]", max_image_bytes, allow_images=False)
        messages = messages[1:]
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise InvalidItem(f"{where}: expected one user message, optionally preceded by a system message")
    text, images = _content(messages[0], f"{where}[{len(value) - 1}]", max_image_bytes, allow_images=True)
    return Item(text=text, images=images, instruction=instruction)


def embedding_items(input, messages, max_items: int, max_image_bytes: int) -> list[tuple[str, Item]]:
    """`input` or `messages` of an embeddings request -> (location for error messages, Item) per input."""
    if (input is None) == (messages is None):
        raise InvalidItem("exactly one of input or messages is required")
    if input is not None:
        if isinstance(input, str):
            return [("input", Item(text=input))]
        if not isinstance(input, list):
            raise InvalidItem("input: must be a string, an array of strings, or an array of Items")
        field, values, parse = "input", input, input_item
    else:
        if not isinstance(messages, list):
            raise InvalidItem("messages: must be a conversation or an array of conversations")
        if messages and all(isinstance(c, dict) for c in messages):
            return [("messages", conversation(messages, "messages", max_image_bytes))]
        field, values, parse = "messages", messages, conversation
    if not values:
        raise InvalidItem(f"{field}: must not be empty")
    if len(values) > max_items:
        raise InvalidItem(f"{field}: {len(values)} inputs is over the limit of {max_items} per request")
    return [(f"{field}[{i}]", parse(v, f"{field}[{i}]", max_image_bytes)) for i, v in enumerate(values)]
