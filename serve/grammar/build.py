"""Build one request's grammar from the checked-in skeleton.

`skeleton.lark` carries the reply structure and nothing about a client; a request supplies two things, and each
replaces the text strictly between a pair of marker comments:

    SPLICE_JSON_SCHEMA   the client's response schema, pasted verbatim after `%json`
    SPLICE_TOOLS         the declared tools, compiled into per-tool rules by tools_fragment.py

The third substitution is the entry rule: `start:` names the stub `entry_select`, and `select_entry` replaces that
one token with the entry the prompt tail calls for, so the same file serves every request shape.

Both splices are exact.  With no tools and no schema the output is the skeleton byte for byte, which is what makes
`skeleton.lark` reviewable on its own and makes an unintended edit show up as a test failure rather than as a
changed mask.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SKELETON = Path(__file__).with_name("skeleton.lark")

JSON_BEGIN = "// >>> SPLICE_JSON_SCHEMA_BEGIN"
JSON_END = "// <<< SPLICE_JSON_SCHEMA_END"
TOOLS_BEGIN = "// >>> SPLICE_TOOLS_BEGIN"
TOOLS_END = "// <<< SPLICE_TOOLS_END"
START = "start: entry_select"

# What the skeleton already holds in each block.  Building with nothing declared must reproduce it exactly.
DEFAULT_JSON_RULE = 'json_response: %json {"type": "object"}'
DEFAULT_TOOLS = "tool_dispatch: NEVER_MATCH\nNEVER_MATCH: /[^\\x00-\\xff]/"


class GrammarError(ValueError):
    """The skeleton is not in the shape the builder expects.  A startup failure, never a request-level one."""


def skeleton() -> str:
    return SKELETON.read_text(encoding="utf-8")


def _splice(text: str, begin: str, end: str, replacement: str) -> str:
    """Replace the lines strictly between the two markers, keeping the markers themselves.

    Both markers must appear exactly once: a duplicated marker means the file grew a second splice point, which
    would silently make the replacement ambiguous.
    """
    if text.count(begin) != 1 or text.count(end) != 1:
        raise GrammarError(f"{SKELETON.name}: expected each of {begin!r} / {end!r} exactly once")
    i = text.index(begin) + len(begin)
    j = text.index(end)
    if j < i:
        raise GrammarError(f"{SKELETON.name}: {end!r} precedes {begin!r}")
    return text[:i] + "\n" + replacement.rstrip("\n") + "\n" + text[j:]


def build(tools: list[dict[str, Any]] | None = None, schema: dict[str, Any] | None = None) -> str:
    """The grammar for one request, still starting at `entry_select`.

    `schema` is used only when no tools are declared: the template lets a reply be tool calls or one JSON object,
    never both, so with tools present the schema is not applied (`HANDOVER.md` D10).
    """
    from .tools_fragment import compile_tools_fragment

    text = skeleton()
    text = _splice(text, TOOLS_BEGIN, TOOLS_END, compile_tools_fragment(tools or []))
    if tools:
        schema = None             # tool calls or one JSON object, never both: the tools win
    rule = DEFAULT_JSON_RULE if schema is None else "json_response: %json " + json.dumps(schema)
    return _splice(text, JSON_BEGIN, JSON_END, rule)


def select_entry(grammar: str, entry: str) -> str:
    """Point `start:` at `entry`.  The marker line appears exactly once, by construction."""
    if grammar.count(START) != 1:
        raise GrammarError(f"expected {START!r} exactly once, found {grammar.count(START)}")
    return grammar.replace(START, "start: " + entry)


def for_entry(entry: str, tools: list[dict[str, Any]] | None = None,
              schema: dict[str, Any] | None = None) -> str:
    """The grammar a decode of `entry` starts from, which is what the matcher is built from."""
    return select_entry(build(tools, schema), entry)
