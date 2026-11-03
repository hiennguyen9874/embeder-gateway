"""Preset Generator: Registry -> the llama-server router's models-preset.ini."""

from gateway.config import Registry


def _value(v) -> str:
    return str(v).lower() if isinstance(v, bool) else str(v)


def render_preset(registry: Registry) -> str:
    lines = []
    for c in registry.checkpoints.values():
        options = {"model": c.dir / c.model}
        if c.mmproj:
            options["mmproj"] = c.dir / c.mmproj
        # Loaded at startup and, with the router's --models-max >= presets, never evicted.
        options["load-on-startup"] = True
        options.update(c.args)
        lines += [f"[{c.name}]", *(f"{k} = {_value(v)}" for k, v in options.items()), ""]
    return "\n".join(lines)
