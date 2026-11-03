#!/usr/bin/env python3
"""Remove soft wrapping from Markdown prose (stdlib only; Python 3.10+).

Default: check docs/agents/{voice-lab,worker}. Use --write to apply changes.
Conservative source formatter, not a complete CommonMark parser: quotes,
tables, HTML, reference definitions and code are left untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = [ROOT / "docs/agents/voice-lab", ROOT / "docs/agents/worker"]
LIST = re.compile(r"^( *(?:[-+*]|\d+[.)])\s+)(.*)$")
FENCE = re.compile(r"^\s*(\x60{3,}|~{3,})")
SPECIAL = re.compile(r"^\s*(?:#{1,6}(?:\s|$)|>)")
RULE = re.compile(r"^ {0,3}(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,}|=+\s*$|-+\s*$)$")


def hard_break(line: str) -> bool:
    trailing = len(line) - len(line.rstrip("\\"))
    return line.endswith("  ") or trailing % 2 == 1


def unwrap(text: str) -> str:
    """Join adjacent prose in the same paragraph/list item.

    Preserve blank lines, explicit hard breaks, newline convention and final EOF.
    List continuation indentation is removed only at a join boundary.
    """
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(newline)
    out: list[str] = []
    pending: str | None = None
    pending_indent = 0
    list_indent: int | None = None
    fence: tuple[str, int] | None = None
    opaque = False
    frontmatter = bool(lines and lines[0] == "---")

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            out.append(pending)
            pending = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        match = FENCE.match(line)
        if frontmatter:
            flush()
            out.append(line)
            if index > 0 and stripped in ("---", "..."):
                frontmatter = False
            continue
        if fence:
            out.append(line)
            if re.fullmatch(r" *" + re.escape(fence[0]) + "{" + str(fence[1]) + r",}\s*", line):
                fence = None
            continue
        if match:
            flush()
            fence = (match[1][0], len(match[1]))
            out.append(line)
            continue
        if not stripped:
            flush()
            out.append(line)
            opaque = False
            continue
        if opaque:
            out.append(line)
            continue
        # Leave ambiguous blocks unchanged rather than risk altering semantics.
        if stripped.startswith("<") or re.match(r"\[[^\]]+\]:", stripped):
            flush()
            out.append(line)
            opaque = True
            continue
        next_is_setext = index + 1 < len(lines) and re.fullmatch(r" {0,3}(?:=+|-+)\s*", lines[index + 1])
        if SPECIAL.match(line) or RULE.fullmatch(line) or "|" in line or next_is_setext or "\t" in line[:indent + 1]:
            flush()
            out.append(line)
            list_indent = None
            continue
        if indent >= (list_indent or 0) + 4:
            flush()
            out.append(line)
            continue
        item = LIST.match(line)
        if item:
            flush()
            list_indent = len(item[1])
            pending_indent = list_indent
            pending = line
            continue
        if list_indent is not None and indent < list_indent:
            list_indent = None
        if indent >= (list_indent or 0) + 4:
            flush()
            out.append(line)
            continue
        expected_indent = list_indent or 0
        if pending is not None and pending_indent == expected_indent and not hard_break(pending):
            pending = pending.rstrip() + " " + line.lstrip()
        else:
            flush()
            pending = line
            pending_indent = expected_indent
    flush()
    return newline.join(out)


def markdown_files(paths: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        if path.is_symlink():
            raise ValueError(f"Refusing symlink: {path}")
        if path.is_dir():
            files.update(p for p in path.rglob("*.md") if p.is_file() and not p.is_symlink())
        elif path.suffix.lower() == ".md":
            files.add(path)
        else:
            raise ValueError(f"Expected a Markdown file or directory: {path}")
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="rewrite files in place")
    mode.add_argument("--check", action="store_true", help="check only (default); exit 1 if changes are needed")
    parser.add_argument("paths", nargs="*", type=Path, help="Markdown files/directories; defaults to the two agent guide directories")
    args = parser.parse_args()
    try:
        files = markdown_files(args.paths or DEFAULT_PATHS)
        changed = 0
        for path in files:
            # Bytes avoid universal-newline conversion during read/write.
            original = path.read_bytes().decode("utf-8")
            updated = unwrap(original)
            if updated != original:
                changed += 1
                print(f"{'Fixed' if args.write else 'Would fix'}: {path}")
                if args.write:
                    path.write_bytes(updated.encode("utf-8"))
        print(f"{len(files)} Markdown files checked; {changed} {'fixed' if args.write else 'need fixing'}.")
        return int(changed > 0 and not args.write)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())