"""One request's grammar: the prompt's own tail decides the entry, the request decides the rest.

This is the layer between a request and the text the engine compiles.  It answers three questions and caches the
answer to the expensive one:

    the entry rule     from the bytes the prompt ends with (`entries.resolve_entry`, never a `thinking` flag)
    the text           the skeleton with the request's tools and schema spliced in, `start:` pointed at the entry
    the identity       the first 16 hex of the SHA-256 of that text, which is also the grammar file's basename

Two requests with the same entry, tools and schema produce the same text, so they share one splice and one file on
disk (the name is the text's first 16 hex).  The cache is keyed by shape, not by text, because the text is what a miss
computes.  The key is the shape the **request** declared, never the repaired one: a repair is a function of the
shape, so two requests that differ only in what had to be relaxed still share the splice of the original.

A `Pipeline` given a `compiler` (`ladder-design.md` §1) compiles before it caches: the compiler's message names a
tool parameter whose schema the grammar cannot express, that parameter is widened while retaining its string-vs-JSON
value encoding; then the ladder
(`serve/grammar/ladder.py`) relaxes the response schema until it compiles.  A compiler that is `None` is the pure
splice, unchanged, which is what the tests of the splice itself build.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from typing import Any

from serve.grammar import build as bg
from serve.grammar import entries as be
from serve.grammar import ladder as bl

CACHE_MAX = 64
# The tail is a handful of tokens, but a call's name and the two template tails together are longer than that, and
# only the end of what is decoded is read: decoding a generous window costs microseconds and removes a class of
# "fits in 16 tokens" bugs (a byte-level tokenizer needs one id per character).
TAIL_TOKENS = 256


class GrammarUnavailable(RuntimeError):
    """This request has no grammar, so it cannot be decoded at all (D-030).

    One class for every reason the server can have for refusing a reply it cannot constrain, because both API
    surfaces map it the same way: `status` alone decides the HTTP status and `kind` the error family, so a new
    reason is a new message, never a new branch at the boundary.  Below 500 the request asked for something no
    grammar can express; 500 is Strata's own fault (a skeleton that does not splice, an unusable file name); 503 is
    a grammar the server cannot supply or write at all (no pack tokenizer, an engine built without the grammar
    decoder, an unwritable grammar directory).  Not a ValueError: the generic handlers turn one of those into a 400,
    which would call the server's own condition the client's.
    """

    code = "grammar_unavailable"

    def __init__(self, message: str, status: int = 500):
        super().__init__(message)
        self.status = status

    @property
    def kind(self) -> str:
        """The error family both APIs spell: `invalid_request_error` for a 4xx, `server_error` otherwise."""
        return "invalid_request_error" if self.status < 500 else "server_error"


@dataclass(frozen=True)
class Grammar:
    """What one request decodes with.  `id` is the 16-hex basename of the file the engine reads.

    `repairs` is what the ladder changed to make the text compile: its `Step`s, one per rung at the response
    schema, plus one per tool parameter that became `{}`.  Empty for a schema that compiled as written, which is
    also every `Grammar` built without a compiler.
    """

    entry: str
    text: str
    id: str
    repairs: tuple = ()
    # Tool schemas after compiler-directed widening. The reply reader must use the same schemas as the grammar.
    _tools: tuple | None = None

    def __post_init__(self):
        object.__setattr__(self, "_tools", None if self._tools is None else tuple(copy.deepcopy(self._tools)))

    @property
    def tools(self) -> tuple | None:
        """An isolated copy of the effective schemas, so callers cannot mutate the cached grammar."""
        return None if self._tools is None else tuple(copy.deepcopy(self._tools))


def identity(text: str) -> str:
    """The grammar's name: 16 hex digits of the SHA-256 of its bytes, which is what the file is called."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------- a compile message -> a parameter
# llguidance prints a numbered excerpt around the line a compile error belongs to and puts a caret under that line
# (`third_party/llguidance/parser/src/lark/lexer.rs::highlight_location`), which is the only place the message names
# the rule that failed.  Inside a `%json` payload the message carries no position at all (probe §B), so for a tool
# this excerpt is the whole mapping: rule -> tool index, the rule's own literal -> the parameter's name.
_SOURCE_LINE = re.compile(r"\A\s*\d+ \| (.*)\Z")
# What `tools_fragment.compile_param_value_rule` writes: `param_t<tool>_<slug>_<bit>: "<parameter=NAME>\n" ...`.
# The tool index comes from the rule's own name and the parameter from the template literal, because the bit is an
# order (required first) and would have to be undone to find the name.
_PARAM_RULE = re.compile(r'\Aparam_t(\d+)_[A-Za-z0-9_]+: "<parameter=(.*?)>\\n"')


def failing_line(message: str) -> str | None:
    """The grammar source line a compile message points at, or None when it points at none.

    The line to read is the one above the caret line.  A caret under a regex (envelope C's excerpt of the failing
    pattern) is skipped because the line it follows is not a numbered source line.
    """
    lines = message.splitlines()
    for index in range(1, len(lines)):
        if not lines[index].lstrip(" ").startswith("^") or not lines[index].startswith(" "):
            continue
        match = _SOURCE_LINE.match(lines[index - 1])
        if match is not None:
            return match.group(1)
    return None


def parameter_of(message: str) -> tuple[int, str] | None:
    """`(tool index, parameter name)` when the compile message points at a tool parameter's rule, else None.

    None means "this message is not about one parameter", and the caller must stop rather than guess: a wrong guess
    would replace a parameter the compiler accepted.
    """
    line = failing_line(message)
    match = _PARAM_RULE.match(line or "")
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def _function(tool: dict[str, Any]) -> dict[str, Any]:
    """A tool's function object: OpenAI nests it, MCP does not."""
    return tool.get("function", tool)


def _properties(tool: dict[str, Any]) -> dict[str, Any]:
    """The parameters `compile_tools_fragment` turns into rules, read the same way it reads them."""
    return (_function(tool).get("parameters") or {}).get("properties") or {}


def _summary(core: str) -> str:
    """One line's worth of a message, without the lines llguidance indents (the regex echo and its caret)."""
    return " ".join(line.strip() for line in core.splitlines()
                    if line.strip() and not line.startswith("    "))


def _terminal_message(finding: bl.Finding) -> str:
    """The client-facing reason a response schema has no grammar (D-065)."""
    return ("the response schema cannot be compiled, and it was not simply refused: this server relaxed it as far "
            f"as it can and the grammar compiler still refuses it. The compiler said: {finding.core}")


def _tools_message(finding: bl.Finding) -> str:
    """The same, for a request that declared tools and no schema: the tools are what could not be compiled.

    A parameter whose schema the compiler refuses is already widened by the time this can happen (design §6), so a
    refusal here is about the tools as a whole - a count, a name the grammar cannot spell - and the message says
    that rather than blaming a schema the request never sent.
    """
    return ("the tools in this request cannot be compiled, and it was not simply refused: each parameter the "
            "compiler named had its schema widened to accept anything, and the grammar still does not build. "
            f"The compiler said: {finding.core}")


class Pipeline:
    """Shape -> grammar, with the splice cached.  Holds a tokenizer only to read a prompt's tail.

    `compiler` is anything with `.compile_error(text) -> str | None` (`matcher.Vocabulary`), and a `Service` always
    hands one over, so the ladder is mandatory wherever a request reaches a grammar.  `None` keeps the pure splice:
    no compile, no repair, byte-identical text, which is what the splice's own tests build.
    """

    def __init__(self, tok, max_entries: int = CACHE_MAX, compiler=None):
        self.tok = tok
        self.max = max_entries
        self.compiler = compiler
        self._lock = threading.Lock()
        self._shapes: collections.OrderedDict[tuple, Grammar] = collections.OrderedDict()
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------ the request's shape
    def tail(self, ids) -> str:
        """The bytes the prompt ends with: the state the reply starts from."""
        return self.tok.decode(list(ids)[-TAIL_TOKENS:])

    def for_request(self, ids, tools: list[dict[str, Any]] | None = None,
                    schema: dict[str, Any] | None = None, required: bool = False) -> Grammar:
        """The grammar this request decodes with.

        Raises `entries.UnresolvedTail` when the prompt does not end in a state the grammar has an entry for (a
        prompt that was not rendered by this template), and `build.GrammarError` when the skeleton no longer splices.
        """
        entry = be.resolve_entry(self.tail(ids), tools=tools, schema=schema, required=required)
        return self.for_entry(entry, tools, schema)

    def for_entry(self, entry: str, tools: list[dict[str, Any]] | None = None,
                  schema: dict[str, Any] | None = None) -> Grammar:
        """The grammar for one entry of one declared shape, cached by shape.

        Tool order is part of the shape: the fragment's rule names carry the tool's index, so two orders are two
        grammars.  A schema's own key order is part of it too, because the schema is pasted into the text verbatim.

        Raises `GrammarUnavailable` when the compiler refuses a response schema the ladder cannot relax (D-065).
        """
        key = (entry, json.dumps(tools, sort_keys=True), json.dumps(schema, sort_keys=True))
        with self._lock:
            hit = self._shapes.get(key)
            if hit is not None:
                self._shapes.move_to_end(key)
                self.hits += 1
                return hit
            self.misses += 1
        # Built outside the lock: a splice is a few milliseconds of string work and holding the lock for it would
        # serialise every request behind the slowest one.  A racing duplicate is harmless - the same text.
        text, repairs, effective_tools = self._text(entry, tools, schema)
        grammar = Grammar(entry=entry, text=text, id=identity(text), repairs=repairs,
                          _tools=None if effective_tools is None else tuple(effective_tools))
        with self._lock:
            self._shapes[key] = grammar
            while len(self._shapes) > self.max:
                self._shapes.popitem(last=False)
        return grammar

    # ------------------------------------------------------------------ the ladder, before the splice is cached
    def _text(self, entry: str, tools: list[dict[str, Any]] | None,
              schema: dict[str, Any] | None) -> tuple[str, tuple, list[dict[str, Any]] | None]:
        """`(text, repairs, tools)`: compiled text, repairs, and tool schemas after parameter widening.

        Without a compiler this is exactly `build.for_entry` and no repairs, so the pure-splice callers and their
        tests are untouched.  With one, the tool parameters come first (a parameter the grammar cannot express must
        not take the request down) and the response schema last, because only the schema's own ladder can decide
        whether the request has a grammar at all.
        """
        if self.compiler is None:
            return bg.for_entry(entry, tools, schema), (), tools
        tools, steps = self._repair_parameters(entry, tools)

        def compile_text(payload_text: str) -> str | None:
            # The ladder hands back the exact string the builder splices (`json.dumps` defaults, `Infinity` and
            # all), so parsing it is how a repair is re-compiled; `json.loads` accepts the non-finite numbers the
            # payload may hold.
            return self.compiler.compile_error(bg.for_entry(entry, tools, json.loads(payload_text)))

        outcome = bl.run(schema, compile_text)
        if outcome.terminal is not None:
            # With no response schema the only thing left that can fail is the tools themselves, and saying
            # "the response schema" then would name something the request does not have.
            message = _tools_message(outcome.terminal) if schema is None else _terminal_message(outcome.terminal)
            raise GrammarUnavailable(message, status=bl.status(outcome.terminal))
        return bg.for_entry(entry, tools, outcome.schema), tuple(steps) + outcome.steps, tools

    def _repair_parameters(self, entry: str, tools: list[dict[str, Any]] | None
                           ) -> tuple[list[dict[str, Any]] | None, list[bl.Step]]:
        """Every tool parameter whose schema the grammar refuses is widened to its value encoding (design §6, slice 5e).

        The loop is bounded by the parameter count plus one and stops as soon as the message no longer names a
        parameter: a message that points elsewhere (`json_response`, a `params_` rule) is not something guessing can
        improve.  The copy is deep because the caller's request shape is the cache key and must stay as it came in.
        """
        if not tools or self.compiler is None:
            return tools, []
        repaired = copy.deepcopy(tools)
        steps: list[bl.Step] = []
        budget = sum(len(_properties(tool)) for tool in repaired) + 1
        for _ in range(budget):
            message = self.compiler.compile_error(bg.for_entry(entry, repaired, None))
            if message is None:
                break
            found = parameter_of(message)
            if found is None:
                break
            index, name = found
            if index >= len(repaired) or name not in _properties(repaired[index]):
                break
            properties = _function(repaired[index]).setdefault("parameters", {}) \
                .setdefault("properties", {})
            if properties[name] == {}:          # already relaxed: a second round would change nothing
                break
            # A widened string remains raw in the template. Preserve that encoding fact for StructureParser:
            # an untyped schema would make it JSON-decode the unquoted bytes (and stream them with JSON quotes).
            original = properties[name]
            params = _function(repaired[index]).get("parameters") or {}
            defs = params.get("$defs") or params.get("definitions") or {}
            properties[name] = {"type": "string"} if _declares_string(original, defs) else {}
            finding = bl.classify(message)
            tool_name = _function(repaired[index]).get("name", "?")
            note = ("tool %r parameter %r was widened (the compiler refused its schema: %s)"
                    % (tool_name, name, _summary(finding.core)[:200]))
            print(f"[strata] grammar: {note}", flush=True)
            steps.append(bl.Step(2, finding.cls, None, note))
        return repaired, steps

    # ------------------------------------------------------------------ counters, for the exit deliverable
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "shapes": len(self._shapes), "max": self.max}


def _declares_string(schema: Any, defs: dict[str, Any] | None = None) -> bool:
    """Whether the template's declared value type is a raw string (or the singleton string spelling)."""
    if not isinstance(schema, dict):
        return False
    kind = schema.get("type")
    if kind == "string" or kind == ["string"]:
        return True
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return _declares_string((defs or {}).get(ref.split("/")[-1], {}), {})
    return False
