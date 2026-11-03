# Embedder Gateway

A gateway that exposes embedding and reranking models served by llama.cpp behind one stable, provider-compatible API, primarily for WeKnora.

## Language

**Served Model**:
A name a client puts in the `model` field of a request, bound to one Checkpoint plus a Prompt Format and its settings. One Checkpoint may back several Served Models.
_Avoid_: alias, model name, deployment

**Checkpoint**:
The concrete weights files of a model (GGUF, plus mmproj for vision), at a specific quantization.
_Avoid_: model file, weights

**Backend**:
The llama.cpp server instance that loads a Checkpoint and performs inference.
_Avoid_: worker, engine, upstream

**Prompt Format**:
A reusable rule for turning a client's input into the exact model input a Checkpoint was trained on (instruction, chat wrapping, media placement).
_Avoid_: template, adapter

**Instruction**:
The task description given to an instruction-aware model alongside the input, e.g. "Represent the user's input."
_Avoid_: system prompt, task prompt

**Input Role**:
Whether an embedding input is a **Query** (searched with) or a **Document** (indexed); it selects the Instruction. Defaults to Document.
_Avoid_: input type, task

**Modality**:
A kind of content an Item can carry: text or image. A Served Model declares which Modalities it accepts.
_Avoid_: media type, input format

**Item**:
One unit of content in a request (an embedding input, a rerank query, or a rerank document), made of text, images, or both.
_Avoid_: part, content, payload
