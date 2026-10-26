"""The compile ladder: what to do when the grammar compiler refuses a client's JSON schema.

The spec is `tmp/impl-plan/ladder-design.md`; the messages and the offset arithmetic are measured in
`tmp/research/ladder-probe.md`, and the classes they belong to are tabulated in
`tmp/research/llg-error-taxonomy.md`.  This module is data in and data out: it never loads the library, opens
a file, imports `matcher`, or logs.  The caller hands it a compiler as `compile_text(payload) -> str | None`
that returns `None` when the text compiles and the library's message when it does not, so the whole loop can be
tested without a `.so`, a pack or a server.

Why a ladder and not one repair: a schema can carry several independent faults and the compiler reports the
first one it reaches, so a repair has to be re-compiled, and the loop needs three bounds (design §5, D-055).
Why "relaxing" and not "fixing": a repair may widen what the client asked for, and the reply is still checked
against the **original** schema after it is generated, so a widened schema turns "cannot compile" into
"compiles, reply checked afterwards" - never into a wrong answer.

The four moving parts, each a pure function:

    classify   the library's message -> a class plus the names it mentions (design §2)
    locate     a class + the payload   -> a JSON pointer, `ambiguous`, or no pointer (design §3)
    repair     a schema + a finding    -> one relaxed copy of the schema, or None (design §4)
    run        the loop with the three bounds (design §5)

Offsets are only meaningful against one exact string, and that string is `payload(schema)` - `json.dumps` with
default separators and `ensure_ascii=True`, which is what `serve/grammar/build.py` splices.  Nothing here maps
an offset onto the client's own JSON text.

Two things in the design's table are worth reading twice.  First, `regex parse error` arrives in **two**
envelopes - inside `failed to compile JSON schema:` for a schema `pattern`, and inside `invalid regex "..."
(in regex):` for a tool parameter that is compiled as a Lark regex - and this module classifies both as
`unsupported`, carrying the echoed pattern and the caret column, because the repair (drop the schema's
`pattern`) is the same either way.  Second, the `at <line>(<column>)` in envelope A is the column of the
rule's first token and carries **no** positional information about the schema (probe §B): only envelope B's
`at line 1 column N` and the regex echo are exact channels.

Small extensions to the design: a few prefixes the design's table omits but the taxonomy measures
(`oneOf constraints ...`, `circular references with sibling keys ...`, `min/maxProperties only supported ...`,
`patternProperty regexes ...`) classify as `unsupported` instead of `unknown`, and `Finding` carries `channel`,
`pattern`, `caret`, `named` and `candidates` because the echoed pattern, the caret column and a multi-name
message (`Unimplemented keys: ["contains", "if", "then"]`) have nowhere to live in the fields the design named.
The one ordering change is the `config` prefix, tested before the `invalid` row: `unknown field `bogus`,
expected one of ...` contains the word `expected`, which the design's `invalid` row matches.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator

# Design §5: the message bound and the rung ceiling.  A 40 000-branch schema's message carries the whole
# ~690 KB payload as its source excerpt (probe §1 case 8), so a message is cut before it is stored or raised.
MESSAGE_LIMIT = 2000
MAX_RUNGS = 8

#: Every class the classifier can return: the design's six plus `ok` for a clean compile, `parse` for envelope B
#: and `unknown` for a message that could not be attributed at all.
CLASSES = ("ok", "unsupported", "resource", "reference", "invalid", "unsatisfiable", "parse", "config", "unknown")

#: Design §6 / D-065: a schema that cannot be made to compile is the client's 400; a bad `x-guidance` object is
#: ours (the server writes it, not the client), so it is a 500.  Raising is `serve/pipeline.py`'s job.
TERMINAL_STATUS = 400
STATUS = {"config": 500}

#: Channels `locate` knows (design §3).  `none` means the message names no place at all.
CHANNELS = ("offset", "regex", "named", "none")


def payload(schema: Any) -> str:
    """The exact string the grammar builder splices after `%json`, and the only string offsets mean anything on.

    `json.dumps` defaults: `", "` / `": "` separators and `ensure_ascii=True`, so the payload is pure ASCII and
    a character offset is a byte offset (probe §B2).  Deliberately no `allow_nan=False`: the payload must be the
    string the compiler was handed, and `Infinity` is one of the failures the ladder has to explain (§7).
    """
    return json.dumps(schema)


@dataclass(frozen=True)
class Finding:
    """One compile message, classified and (after `locate`) attributed.

    `core` is the message with its envelope and source excerpt stripped, truncated to `MESSAGE_LIMIT` characters
    plus `... (N bytes)`.  `raw` is the whole message, truncated the same way, for an operator's log.

    `keyword` is the schema keyword the message names verbatim, and exists for rung 3 (delete it); it is None
    when the message names no keyword this module knows (then an `invalid` schema is terminal, design §4).
    `named` is every token the message quotes - a property name, a `$ref` pointer, a format value, a list of
    unimplemented keys - and is what the named channel searches for.
    `pattern` and `caret` are the regex echo: the whole pattern as the library printed it and the 0-based offset
    of its caret line within that pattern (probe §B1).
    `candidates` is every pointer `locate` considered; `pointer` is set only when there was exactly one.
    """

    ok: bool
    cls: str
    core: str
    pointer: str | None = None
    ambiguous: bool = False
    keyword: str | None = None
    raw: str = ""
    channel: str | None = None
    pattern: str | None = None
    caret: int | None = None
    named: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class Step:
    """One rung that was applied, so an operator can see the client's schema was not honoured exactly."""

    rung: int
    cls: str
    pointer: str | None
    note: str


@dataclass(frozen=True)
class Outcome:
    """The schema to compile, or `None` plus the terminal finding when no repair was left."""

    schema: Any | None
    steps: tuple[Step, ...] = ()
    terminal: Finding | None = None

    @property
    def ok(self) -> bool:
        return self.terminal is None


def status(finding: Finding) -> int:
    """The HTTP status a terminal finding is raised as (design §6, D-065).  Data only: nothing here raises."""
    return STATUS.get(finding.cls, TERMINAL_STATUS)


# ---------------------------------------------------------------------------------------------------------
# Envelopes and the message core (design §2, probe §0)
# ---------------------------------------------------------------------------------------------------------

# Envelope A: `at 173(16): failed to compile JSON schema: <inner>` followed by a numbered source excerpt.
# Envelope C: `at 189(22): invalid regex "(?:(?=a))" (in regex): <regex parse error>` plus its own excerpt.
# Envelope B is not stripped here: its prefix is the classifier for the only class that has a real offset.
_ENVELOPE_JSON = re.compile(r"\Aat \d+\(\d+\): failed to compile JSON schema: ")
_ENVELOPE_REGEX = re.compile(r'\Aat \d+\(\d+\): invalid regex ".*?" \(in regex\): ')
_ENVELOPE_PARSE = 'failed to parse "%json": '
_EXCERPT_LINE = re.compile(r"\A\s*\d+ \|")

# The design's `invalid` row is a substring match on two phrases; everything else is a prefix.
_INVALID_PHRASES = (" must be ", "expected ")

# Resource phrases (design §2's third row).  `exceeds` is added beside the design's `exceeded`: llguidance's
# `minLength (4294967296) exceeds the supported range` is the same limit, spelled in the present tense.
_RESOURCE_PHRASES = ("schema too large", "exceeded", "exceeds", "too many")

# Envelope B's one repairable message (probe §9): serde_json's nesting limit, which reports an exact column and
# so can be collapsed at the offending node.  Every other parse failure is our own serialization going wrong and
# stays terminal.
_RECURSION_LIMIT = "recursion limit exceeded"

# Unsupported messages from `tmp/research/llg-error-taxonomy.md` §1.1 that the design's table does not prefix.
_EXTRA_UNSUPPORTED = (
    "oneOf constraints are not supported",
    "circular references with sibling keys are not supported",
    "min/maxProperties only supported",
    "patternProperty regexes ",
    "can't determine if patternProperty regexes",
)

# What rung 1 may fix: exactly these two messages (design §4: `unsupported/keys`, `unsupported/format`).
_RUNG1_PREFIXES = ("Unimplemented keys: ", "Unknown format: ")

# A rung-3 keyword must be a JSON Schema keyword the message names, never an arbitrary token: `schema must be an
# object or boolean` names "schema", and deleting a key called "schema" would repair nothing.
_KEYWORDS = frozenset({
    "$ref", "$schema", "additionalProperties", "allOf", "anyOf", "const", "contains", "dependentRequired",
    "dependentSchemas", "else", "enum", "exclusiveMaximum", "exclusiveMinimum", "format", "if", "items",
    "maxItems", "maxLength", "maxProperties", "maximum", "minItems", "minLength", "minProperties", "minimum",
    "multipleOf", "not", "oneOf", "pattern", "patternProperties", "prefixItems", "properties", "propertyNames",
    "required", "then", "type", "unevaluatedItems", "unevaluatedProperties", "uniqueItems",
})


def _truncate(text: str) -> str:
    """Design §5: 2000 characters, then `... (N bytes)` with N the size of what was there."""
    if len(text) <= MESSAGE_LIMIT:
        return text
    return text[:MESSAGE_LIMIT] + "... (%d bytes)" % len(text.encode("utf-8"))


def _head(message: str) -> str:
    """The message with envelope A or C's prefix removed; envelope B is left whole (its prefix is a class)."""
    for pattern in (_ENVELOPE_JSON, _ENVELOPE_REGEX):
        match = pattern.match(message)
        if match is not None:
            return message[match.end():]
    return message


def _strip_excerpt(text: str) -> str:
    """Drop the numbered source excerpt envelope A appends (and C appends, in the same shape).

    The excerpt echoes the payload line in full, which is where a 40 000-branch schema's ~690 KB error string
    comes from; it carries no information the inner message does not.
    """
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if _EXCERPT_LINE.match(line):
            return "\n".join(lines[:index]).rstrip()
    return text.rstrip()


def _regex_echo(text: str) -> tuple[str | None, int | None]:
    """The echoed pattern and the caret column inside it (probe §B1: the excerpt line is `len(pattern) + 4`)."""
    lines = text.split("\n")
    for index in range(1, len(lines)):
        line = lines[index]
        if not line.startswith("    "):
            continue
        pattern = line[4:]
        caret = None
        if index + 1 < len(lines):
            following = lines[index + 1]
            stripped = following.lstrip(" ")
            if stripped.startswith("^"):
                caret = len(following) - len(stripped) - 4
        return pattern, caret
    return None, None


def _named_keyword(core: str) -> str | None:
    """`prefixItems must be an array` -> `prefixItems`; `Expected f64 for 'minimum'` -> `minimum`."""
    match = re.search(r"([A-Za-z$][\w$]*) must be ", core)
    if match is None:
        match = re.search(r"for '([A-Za-z$][\w$]*)'", core)
    if match is not None and match.group(1) in _KEYWORDS:
        return match.group(1)
    return None


def classify(message: str | None) -> Finding:
    """Sort one compile message into a class, in the design's table order, specific before general.

    The one reordering is the `config` prefix ahead of the `invalid` row (see the module docstring).
    `None` or an empty message is the `ok` finding, so a caller can `classify(compile_text(...))` without
    special-casing success.
    """
    if not message or not message.strip():
        return Finding(ok=True, cls="ok", core="", raw="")

    raw = _truncate(message)
    head = _head(message)
    core = _strip_excerpt(head)
    if head.startswith(_ENVELOPE_PARSE):
        core = core[len(_ENVELOPE_PARSE):]
    truncated = _truncate(core)

    def finding(cls: str, channel: str, **extra: Any) -> Finding:
        return Finding(ok=False, cls=cls, core=truncated, raw=raw, channel=channel, **extra)

    # Row 1: envelope B, the only exact positional channel.  Non-finite numbers and the parse-time depth limit.
    if head.startswith(_ENVELOPE_PARSE):
        return finding("parse", "offset")

    # Rows 2-3: a named key list or a named format value; rung 1 relaxes these at the root, so no pointer.
    if core.startswith("Unimplemented keys: "):
        return finding("unsupported", "none", named=tuple(re.findall(r'"([^"]*)"', core)))
    if core.startswith("Unknown format: "):
        value = core[len("Unknown format: "):].splitlines()[0].strip()
        return finding("unsupported", "none", named=(value,))

    # Row 4: the regex echo, in either envelope.  The repair is a schema-side `pattern` deletion.
    if core.startswith("regex parse error:"):
        pattern, caret = _regex_echo(core)
        return finding("unsupported", "regex", pattern=pattern, caret=caret, keyword="pattern")

    # Row 5: a `$ref` that does not resolve, and the resource/fetch failure taxonomy §1.3 also calls reference.
    match = re.match(r"Pointer '(.*?)' does not exist", core)
    if match is not None:
        return finding("reference", "named", named=(match.group(1),))
    match = re.match(r"Resource '(.*?)' is not present in a registry", core)
    if match is not None:
        return finding("reference", "named", named=(match.group(1),))

    lowered = core.lower()

    # Row 9 (ahead of row 6, which is the only reordering here): our own `x-guidance` object is terminal and a
    # 500 because the server wrote it.  `unknown field \`bogus\`, expected one of ...` contains the substring
    # `expected`, so it would otherwise be claimed by the invalid row.  Specific before general, as §2 asks.
    if core.startswith("unknown field `") or "invalid escape character in allowed_escapes" in core:
        return finding("config", "none")

    # Row 6: a value that is not a legal JSON Schema.  Repairable only when the message names the keyword.
    if any(phrase in lowered for phrase in _INVALID_PHRASES):
        return finding("invalid", "none", keyword=_named_keyword(core))

    # Row 7: a size / fuel / count limit.
    if any(phrase in lowered for phrase in _RESOURCE_PHRASES):
        return finding("resource", "none")

    # Row 8: an empty language llguidance already decided on.  Only `required property 'X'` names a place.
    if "unsatisfiable" in lowered:
        match = re.search(r"required property '(.*?)' is unsatisfiable", core)
        return finding("unsatisfiable", "named", named=(match.group(1),) if match else ())

    # Taxonomy §1.1 messages the design's table does not list; all unsupported constructs, none repairable here.
    if any(phrase in core for phrase in _EXTRA_UNSUPPORTED):
        return finding("unsupported", "none")

    # Row 10: could not be attributed.  Terminal; the message says so (`Outcome.terminal.cls == "unknown"`).
    return finding("unknown", "none")


# ---------------------------------------------------------------------------------------------------------
# Locate: the three channels (design §3)
# ---------------------------------------------------------------------------------------------------------

_MISSING = object()


class _ScanError(ValueError):
    """The payload is not JSON this scanner can walk.  Only ever raised internally."""


def _child(pointer: str, token: Any) -> str:
    escaped = str(token).replace("~", "~0").replace("/", "~1")
    return (pointer + "/" + escaped) if pointer else "/" + escaped


def _tokens(pointer: str) -> list[str]:
    if not pointer:
        return []
    return [token.replace("~1", "/").replace("~0", "~") for token in pointer.split("/")[1:]]


def _string_end(text: str, index: int) -> int:
    """The index just past the string token that starts at `index` (a `"`)."""
    cursor = index + 1
    while cursor < len(text):
        char = text[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == '"':
            return cursor + 1
        cursor += 1
    raise _ScanError("unterminated string")


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\n\r":
        index += 1
    return index


def _spans(text: str) -> list[tuple[int, int, str, str]]:
    """Every key token, scalar and container in `text`, as `(start, end, pointer, kind)`.

    A hand-written scanner rather than `json.loads` for two measured reasons: the payload for the `Infinity`
    case is not valid JSON for any parser (probe §B2), and the pointer needs the *character offsets*, which
    `json.loads` throws away.  Strings are unescaped with `json.loads` only for the key *token*, which is always
    well formed because the payload came from `json.dumps`.
    """
    spans: list[tuple[int, int, str, str]] = []

    def value(index: int, pointer: str) -> int:
        index = _skip_ws(text, index)
        if index >= len(text):
            raise _ScanError("missing value")
        start = index
        char = text[index]
        if char == "{":
            index = _skip_ws(text, index + 1)
            if index < len(text) and text[index] == "}":
                spans.append((start, index + 1, pointer, "value"))
                return index + 1
            while True:
                index = _skip_ws(text, index)
                if index >= len(text) or text[index] != '"':
                    raise _ScanError("expected a key")
                key_end = _string_end(text, index)
                key = json.loads(text[index:key_end])
                child = _child(pointer, key)
                spans.append((index, key_end, child, "key"))
                index = _skip_ws(text, key_end)
                if index >= len(text) or text[index] != ":":
                    raise _ScanError("expected ':'")
                index = _skip_ws(text, value(index + 1, child))
                if index < len(text) and text[index] == ",":
                    index += 1
                    continue
                if index < len(text) and text[index] == "}":
                    spans.append((start, index + 1, pointer, "value"))
                    return index + 1
                raise _ScanError("expected ',' or '}'")
        if char == "[":
            index = _skip_ws(text, index + 1)
            if index < len(text) and text[index] == "]":
                spans.append((start, index + 1, pointer, "value"))
                return index + 1
            position = 0
            while True:
                index = _skip_ws(text, value(index, _child(pointer, position)))
                position += 1
                if index < len(text) and text[index] == ",":
                    index += 1
                    continue
                if index < len(text) and text[index] == "]":
                    spans.append((start, index + 1, pointer, "value"))
                    return index + 1
                raise _ScanError("expected ',' or ']'")
        if char == '"':
            end = _string_end(text, index)
            spans.append((index, end, pointer, "value"))
            return end
        end = index
        while end < len(text) and text[end] not in ",]} \t\n\r":
            end += 1
        if end == index:
            raise _ScanError("empty literal")
        spans.append((index, end, pointer, "value"))    # number, true, false, null, Infinity, -Infinity, NaN
        return end

    value(0, "")
    return spans


def pointer_at(text: str, offset: int) -> str | None:
    """The JSON pointer for a 0-based offset into `payload(schema)` (design §3.1, probe §B2).

    A key token wins over every container that contains it, which is what makes the depth case report
    `/properties/p` x 64: the parse fails at the *key* of the 64th level, whose value is the container that
    would have been one nesting level too deep (probe §B2 table row 2).
    """
    if offset < 0 or offset >= len(text):
        return None
    try:
        spans = _spans(text)
    except (_ScanError, ValueError, RecursionError):
        return None
    best: tuple[int, int, str] | None = None
    for start, end, pointer, kind in spans:
        if not (start <= offset < end):
            continue
        if kind == "key":
            return pointer
        if best is None or start > best[0] or (start == best[0] and end < best[1]):
            best = (start, end, pointer)
    if best is not None:
        return best[2]
    earlier = [span for span in spans if span[0] <= offset]
    return max(earlier, key=lambda span: span[0])[2] if earlier else None


def _follows_pattern_key(text: str, index: int) -> bool:
    """True when the text at `index` is a `pattern` value, not the same text inside a `description` (probe §B1).

    `index` is the first character of the escaped value, so immediately before it is the value's opening quote.
    Walking back over that quote, the whitespace and the `:` must land on the key token `"pattern"`; in
    `"description": "avoid (?=a)"` the character before the text is `d`, so the decoy is rejected.
    """
    cursor = index - 1
    while cursor >= 0 and text[cursor] in " \t\n\r":
        cursor -= 1
    if cursor >= 0 and text[cursor] == '"':
        cursor -= 1
    while cursor >= 0 and text[cursor] in " \t\n\r":
        cursor -= 1
    if cursor < 0 or text[cursor] != ":":
        return False
    cursor -= 1
    while cursor >= 0 and text[cursor] in " \t\n\r":
        cursor -= 1
    key = '"pattern"'
    return cursor - len(key) + 1 >= 0 and text[cursor - len(key) + 1:cursor + 1] == key


def _pattern_hits(text: str, pattern: str | None) -> list[int]:
    """Offsets in `text` where the *escaped* form of `pattern` occurs as a `pattern` value (probe §B1).

    The message echoes the decoded pattern; the payload holds the JSON-escaped one - `(a)\\1` in the message is
    `(a)\\\\1` in the payload - so the search form is `json.dumps(pattern)[1:-1]`, which is exactly that.
    """
    if not pattern:
        return []
    needle = json.dumps(pattern)[1:-1]
    if not needle:
        return []
    hits: list[int] = []
    start = 0
    while True:
        found = text.find(needle, start)
        if found < 0:
            return hits
        start = found + 1
        if _follows_pattern_key(text, found):
            hits.append(found)


def _fragment(ref: str) -> str:
    hash_index = ref.find("#")
    return ref[hash_index + 1:] if hash_index >= 0 else ref


def _walk(node: Any, pointer: str = "") -> Iterator[tuple[str, Any]]:
    yield pointer, node
    if isinstance(node, dict):
        for key, item in node.items():
            yield from _walk(item, _child(pointer, key))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _walk(item, _child(pointer, index))


def _named_hits(schema: Any, finding: Finding) -> list[str]:
    """Every node the message's named thing lives in (design §3.3).  Order is document order."""
    hits: list[str] = []
    if not finding.named:
        return hits
    name = finding.named[0]
    if finding.cls == "reference":
        for pointer, node in _walk(schema):
            if not isinstance(node, dict):
                continue
            ref = node.get("$ref")
            if isinstance(ref, str) and (_fragment(ref) == name or ref == name):
                hits.append(pointer)
            elif node.get("$schema") == name:
                hits.append(pointer)
    elif finding.cls == "unsatisfiable":
        for pointer, node in _walk(schema):
            if isinstance(node, dict) and isinstance(node.get("required"), list) and name in node["required"]:
                hits.append(pointer)
    return hits


def locate(finding: Finding, text: str) -> Finding:
    """Fill in the pointer from whichever channel the message offers (design §3).

    `text` must be `payload(schema)` for the schema that produced the message: an offset or a pattern search
    against anything else is meaningless (probe §C).  No pointer, one pointer and several pointers are all
    legitimate answers; only the last one sets `ambiguous`.
    """
    if finding.ok:
        return finding

    if finding.channel == "offset":
        match = re.search(r"at line (\d+) column (\d+)", finding.core)
        if match is None:
            return finding
        return replace(finding, pointer=pointer_at(text, int(match.group(2)) - 1))

    if finding.channel == "regex":
        hits = _pattern_hits(text, finding.pattern)
        pointers = tuple(p for p in (pointer_at(text, hit) for hit in hits) if p is not None)
        if len(hits) == 1:
            return replace(finding, pointer=pointers[0] if pointers else None, candidates=pointers)
        if len(hits) > 1:
            return replace(finding, pointer=None, ambiguous=True, candidates=pointers)
        return finding

    if finding.channel == "named":
        try:
            schema = json.loads(text)
        except (TypeError, ValueError, RecursionError):
            return finding
        hits = _named_hits(schema, finding)
        if len(hits) == 1:
            return replace(finding, pointer=hits[0], candidates=tuple(hits))
        if len(hits) > 1:
            return replace(finding, pointer=None, ambiguous=True, candidates=tuple(hits))
        return finding

    return finding


# ---------------------------------------------------------------------------------------------------------
# Repair: one rung, one relaxation, a deep copy (design §4)
# ---------------------------------------------------------------------------------------------------------


def _node_at(schema: Any, pointer: str) -> Any:
    node = schema
    for token in _tokens(pointer):
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        else:
            return _MISSING
    return node


def _parent_of(schema: Any, pointer: str) -> tuple[Any, str] | tuple[None, None]:
    """The container holding `pointer`'s last token, and that token.  No re-encoding: walk the tokens."""
    tokens = _tokens(pointer)
    if not tokens:
        return None, None
    node = schema
    for token in tokens[:-1]:
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        else:
            return None, None
    return node, tokens[-1]


def _set(container: Any, token: str, value: Any) -> bool:
    if isinstance(container, dict) and token in container:
        container[token] = value
        return True
    if isinstance(container, list) and token.isdigit() and int(token) < len(container):
        container[int(token)] = value
        return True
    return False


def _drop(container: Any, token: str) -> bool:
    if isinstance(container, dict) and token in container:
        del container[token]
        return True
    if isinstance(container, list) and token.isdigit() and int(token) < len(container):
        del container[int(token)]
        return True
    return False


def _candidates(schema: Any, finding: Finding) -> list[str]:
    """The pointers this finding points at: located ones first, else a structural search of the schema.

    The search is what makes `repair` usable on a finding that `locate` could not attribute - for a `$ref` the
    message names the pointer string itself, and for an unsatisfiable property it names the key.
    """
    if finding.candidates:
        return list(finding.candidates)
    if finding.pointer is not None:
        return [finding.pointer]
    return _named_hits(schema, finding)


def _rung1(schema: Any) -> tuple[int, Any, str] | None:
    """Root `x-guidance.lenient` for `Unimplemented keys` / `Unknown format`.

    Root only, and that is measured, not assumed: the same object next to the offending keyword changes nothing,
    because `JsonCompileOptions::json_to_llg_with_overrides` reads `x-guidance` once from the root object
    (probe §D).
    """
    if not isinstance(schema, dict):
        return None
    fresh = copy.deepcopy(schema)
    guidance = fresh.get("x-guidance")
    if guidance is None:
        guidance = {}
    elif isinstance(guidance, dict):
        guidance = dict(guidance)
    else:
        return None
    if guidance.get("lenient") is True:
        return None                     # already relaxed: no repair is left, so the ladder must stop
    guidance["lenient"] = True
    fresh["x-guidance"] = guidance
    return 1, fresh, "added x-guidance.lenient at the root (the option is read only from the root; measured)"


def _drop_patterns(schema: Any, finding: Finding) -> tuple[int, Any, str] | None:
    """Rung 2 for a regex echo: delete the offending `pattern` at the located node(s)."""
    pointers = [p for p in _candidates(schema, finding) if p.endswith("/pattern")]
    if not pointers:
        return None
    fresh = copy.deepcopy(schema)
    removed: list[str] = []
    for pointer in pointers:
        parent, token = _parent_of(fresh, pointer)
        if parent is not None and _drop(parent, token):
            removed.append(pointer)
    if not removed:
        return None
    if len(removed) == 1:
        note = "deleted the pattern at %s" % removed[0]
    elif finding.ambiguous:
        note = ("deleted the pattern at %d nodes (%s): the message echoed one pattern and the schema has it in "
                "more than one place" % (len(removed), ", ".join(removed)))
    else:
        note = "deleted the pattern at %d nodes (%s)" % (len(removed), ", ".join(removed))
    return 2, fresh, note


def _drop_refs(schema: Any, finding: Finding) -> tuple[int, Any, str] | None:
    """Rung 2 for a `$ref` that does not resolve: the node holding it becomes `{}`."""
    pointers = _candidates(schema, finding)
    if not pointers:
        return None
    name = finding.named[0] if finding.named else None
    fresh = copy.deepcopy(schema)
    replaced: list[str] = []
    dropped: list[str] = []
    for pointer in pointers:
        node = _node_at(fresh, pointer)
        if node is _MISSING or not isinstance(node, dict):
            continue
        if "$ref" in node:
            if pointer == "":
                fresh = {}
            else:
                parent, token = _parent_of(fresh, pointer)
                if parent is None or not _set(parent, token, {}):
                    continue
            replaced.append(pointer or "/")
        elif name is not None and node.get("$schema") == name:
            del node["$schema"]             # an unknown dialect: dropping the key is the relaxation, not the node
            dropped.append(_child(pointer, "$schema"))
    if not replaced and not dropped:
        return None
    parts = []
    if replaced:
        parts.append("replaced %s with {} (the reference does not resolve)" % ", ".join(replaced))
    if dropped:
        parts.append("dropped %s (the dialect is not one the compiler can fetch)" % ", ".join(dropped))
    return 2, fresh, "; ".join(parts)


def _relax_required(schema: Any, finding: Finding) -> tuple[int, Any, str] | None:
    """Rung 2 for `required property 'X' is unsatisfiable`: that property's schema becomes `{}`.

    The root cause (`minLength (5) is greater than maxLength (2)`) is not in the message - anyhow prints the
    outermost context only (taxonomy §1.5) - so the property is widened rather than "corrected".
    """
    if not finding.named:
        return None
    name = finding.named[0]
    pointers = _candidates(schema, finding)
    if not pointers:
        return None
    fresh = copy.deepcopy(schema)
    relaxed: list[str] = []
    for pointer in pointers:
        node = _node_at(fresh, pointer)
        if node is _MISSING or not isinstance(node, dict):
            continue
        properties = node.get("properties")
        if isinstance(properties, dict) and name in properties:
            properties[name] = {}
            relaxed.append(_child(_child(pointer, "properties"), name))
    if not relaxed:
        return None
    return 2, fresh, "relaxed %s to {} (required property %r was unsatisfiable)" % (
        ", ".join(relaxed), name)


def _truncate_at(schema: Any, pointer: str, why: str = "a resource limit") -> tuple[int, Any, str] | None:
    """Rung 2 for a limit that did name an offset: the located node becomes `true`.

    `true` is a legal JSON Schema (it accepts every value) and it is **not a container**.  Collapsing to `true`
    rather than `{}` is what makes the whole subtree's depth disappear.
    """
    if pointer == "":
        return 2, True, "replaced the whole schema with true (%s fired)" % why
    fresh = copy.deepcopy(schema)
    parent, token = _parent_of(fresh, pointer)
    if parent is None or not _set(parent, token, True):
        return None
    return 2, fresh, "collapsed %s to true (%s fired)" % (pointer, why)


#: Where a schema has to stop nesting, counted the way serde_json counts: containers, so a `properties` level is
#: two (the object and its map).  Measured on this build with `build.for_entry` and llguidance 1.9.1: a 63-level
#: chain - 126 containers - compiles and 64 levels fails.  The cap sits just under that ceiling, and it only ever
#: runs *after* the library has already refused with the nesting limit, so a schema that compiles is never cut.
DEPTH_LIMIT = 100

#: The keywords whose value is a schema, and how many containers each costs towards the nesting limit.
_SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
_SCHEMA_LISTS = ("prefixItems", "allOf", "anyOf", "oneOf")
_SCHEMA_ONE = ("items", "additionalItems", "additionalProperties", "unevaluatedItems", "unevaluatedProperties",
               "contains", "propertyNames", "not", "if", "then", "else", "contentSchema")


def _cap_depth(schema: Any, limit: int = DEPTH_LIMIT) -> tuple[int, Any, str] | None:
    """Rung 2 for the nesting limit: every *schema* below `limit` levels becomes `true`.

    Collapsing the one node the message's column names was measured **not** to work: the column stays 2492 after
    the deepest node shrinks to `true`, so the next round repeats the same repair and the ladder (rightly) stops.
    The reason is that the limit is on a chain's depth, not on one node's size - the chain above the leaf is what
    is too deep.  So the repair caps the chain: schema nodes past `limit` become `true`, which keeps every
    constraint above them and relaxes only what could not be represented.

    A `properties` value is a map of schemas, not a schema, so it is descended into rather than replaced: replacing
    it would emit `"properties": true`, which the library then rejects as `properties must be ...`.
    """
    def walk(node: Any, level: int) -> tuple[Any, list[str]]:
        if not isinstance(node, dict):
            return node, []
        if level >= limit:
            return True, [f"depth {level}"]
        out: dict[str, Any] = {}
        cut: list[str] = []
        for key, value in node.items():
            if key in _SCHEMA_MAPS and isinstance(value, dict):
                children = {}
                for name, sub in value.items():
                    fresh, below = walk(sub, level + 2)      # the object and its map
                    children[name] = fresh
                    cut.extend(below)
                out[key] = children
            elif key in _SCHEMA_LISTS and isinstance(value, list):
                children = []
                for sub in value:
                    fresh, below = walk(sub, level + 2)      # the array and the element
                    children.append(fresh)
                    cut.extend(below)
                out[key] = children
            elif key in _SCHEMA_ONE and isinstance(value, dict):
                fresh, below = walk(value, level + 1)
                out[key] = fresh
                cut.extend(below)
            else:
                out[key] = value
        return out, cut

    fresh, cut = walk(schema, 0)
    if not cut:
        return None
    return 2, fresh, "capped %d schema node(s) below %d levels to true (the nesting limit fired)" % (
        len(cut), limit)


def _drop_keyword(schema: Any, keyword: str) -> tuple[int, Any, str] | None:
    """Rung 3: delete the keyword the message named, at every node that carries it (design §4).

    The pointers are collected first and the deletions happen afterwards: deleting while `_walk` is iterating
    the same dict raises `RuntimeError: dictionary changed size during iteration`.
    """
    fresh = copy.deepcopy(schema)
    targets = [p for p, node in _walk(fresh) if isinstance(node, dict) and keyword in node]
    removed: list[str] = []
    for pointer in targets:
        node = _node_at(fresh, pointer)
        if isinstance(node, dict) and keyword in node:
            del node[keyword]
            removed.append(_child(pointer, keyword))
    if not removed:
        return None
    return 3, fresh, "deleted %r at %d node(s): %s" % (keyword, len(removed), ", ".join(removed))


def _apply(schema: Any, finding: Finding) -> tuple[int, Any, str] | None:
    """The single dispatch: which rung this finding is repaired at, or `None` for a terminal finding."""
    if finding.cls == "unsupported":
        if finding.channel == "regex":
            return _drop_patterns(schema, finding)
        if finding.core.startswith(_RUNG1_PREFIXES):
            return _rung1(schema)
        return None
    if finding.cls == "reference":
        return _drop_refs(schema, finding)
    if finding.cls == "unsatisfiable":
        return _relax_required(schema, finding)
    if finding.cls == "parse" and _RECURSION_LIMIT in finding.core:
        # serde_json's nesting limit.  Measured: one node collapse does not fix it (the column does not move when
        # the leaf shrinks), a depth cap does.  p7.md item 5 asks for a limit to be "flattened, capped or
        # collapsed": this is the cap.
        return _cap_depth(schema)
    if finding.cls == "resource" and finding.pointer is not None:
        return _truncate_at(schema, finding.pointer)
    if finding.cls == "invalid" and finding.keyword:
        return _drop_keyword(schema, finding.keyword)
    return None


def repair(schema: Any, finding: Finding) -> tuple[Any, str] | None:
    """One relaxation of `schema` for `finding`, or `None` when this finding has none.

    The input is never mutated: the returned schema is a deep copy with exactly one repair applied.  A `None`
    here is what makes the ladder terminate - it says "no rung is left", not "try again".
    """
    applied = _apply(schema, finding)
    if applied is None:
        return None
    _, fresh, note = applied
    return fresh, note


# ---------------------------------------------------------------------------------------------------------
# The loop and its three bounds (design §5, D-055)
# ---------------------------------------------------------------------------------------------------------


def _canonical(schema: Any) -> str:
    return json.dumps(schema, sort_keys=True)


class Ladder:
    """Compile, classify, locate, repair - bounded three ways so a repair that removes nothing cannot loop.

    The bounds (design §5): a set of canonical shapes (a shape that repeats is terminal), a set of
    `(pointer, class)` sites already repaired (the same site coming back means the outstanding violations did not
    shrink), and a hard ceiling of `limit` rungs (8 by default).  The design also names the initial violation
    count as a fourth cap, but only one violation is ever visible per compile, so the site set stands in for it.
    """

    def __init__(self, compile_text: Callable[[str], str | None], *, limit: int = MAX_RUNGS) -> None:
        self.compile_text = compile_text
        self.limit = limit

    def run(self, schema: Any) -> Outcome:
        current = schema
        steps: list[Step] = []
        seen = {_canonical(current)}
        sites: set[tuple[str | None, str]] = set()
        while True:
            text = payload(current)
            finding = locate(classify(self.compile_text(text)), text)
            if finding.ok:
                return Outcome(current, tuple(steps), None)
            site = (finding.pointer, finding.cls)
            if site in sites or len(steps) >= self.limit:
                return Outcome(None, tuple(steps), finding)
            applied = _apply(current, finding)
            if applied is None:
                return Outcome(None, tuple(steps), finding)
            rung, candidate, note = applied
            key = _canonical(candidate)
            if key in seen:
                return Outcome(None, tuple(steps), finding)
            seen.add(key)
            sites.add(site)
            steps.append(Step(rung, finding.cls, finding.pointer, note))
            current = candidate


def run(schema: Any, compile_text: Callable[[str], str | None], *, limit: int = MAX_RUNGS) -> Outcome:
    """`Ladder(compile_text).run(schema)`, for callers that do not keep a ladder around."""
    return Ladder(compile_text, limit=limit).run(schema)
