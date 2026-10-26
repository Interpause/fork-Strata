"""One request's grammar: the prompt's own tail decides the entry, the request decides the rest.

This is the layer between a request and the text the engine compiles.  It answers three questions and caches the
answer to the expensive one:

    the entry rule     from the bytes the prompt ends with (`entries.resolve_entry`, never a `thinking` flag)
    the text           the skeleton with the request's tools and schema spliced in, `start:` pointed at the entry
    the identity       the first 16 hex of the SHA-256 of that text, which is also the grammar file's basename

Two requests with the same entry, tools and schema produce the same text, so they share one splice and one file on
disk (the name is the text's first 16 hex).  The cache is keyed by shape, not by text, because the text is what a miss
computes.
"""

from __future__ import annotations

import collections
import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any

from serve.grammar import build as bg
from serve.grammar import entries as be

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
    """What one request decodes with.  `id` is the 16-hex basename of the file the engine reads."""

    entry: str
    text: str
    id: str


def identity(text: str) -> str:
    """The grammar's name: 16 hex digits of the SHA-256 of its bytes, which is what the file is called."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class Pipeline:
    """Shape -> grammar, with the splice cached.  Holds a tokenizer only to read a prompt's tail."""

    def __init__(self, tok, max_entries: int = CACHE_MAX):
        self.tok = tok
        self.max = max_entries
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
        text = bg.for_entry(entry, tools, schema)
        grammar = Grammar(entry=entry, text=text, id=identity(text))
        with self._lock:
            self._shapes[key] = grammar
            while len(self._shapes) > self.max:
                self._shapes.popitem(last=False)
        return grammar

    # ------------------------------------------------------------------ counters, for the exit deliverable
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "shapes": len(self._shapes), "max": self.max}
