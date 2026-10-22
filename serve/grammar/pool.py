"""Compiled matchers, one prototype per grammar, cloned for every request.

Compiling a grammar costs a lexer over 248,320 tokens; advancing one costs microseconds.  So the prototype is kept
and never handed to a request — a request gets a `clone`, which is the same compiled state with its own position
(`contracts/c2-matcher-lifecycle.md`).  A clone is freed when its request ends; it is never put back, because a
matcher that failed a `consume` is permanently in error and must not be pooled.

The pool is bounded: a long-running server sees unbounded distinct grammars (every tool set a client declares is a
different one), and a prototype holds a compiled lexer, not a few bytes.
"""

from __future__ import annotations

import collections
import threading
from pathlib import Path
from typing import Any, Callable

POOL_MAX = 32


class Pool:
    """Grammar id -> prototype matcher.  Disabled (and honest about it) when the library is not built."""

    def __init__(self, tokenizer_dir: Path | str | None = None, tok: Any = None, max_entries: int = POOL_MAX,
                 log: Callable[[str], None] = print):
        self.dir = Path(tokenizer_dir) if tokenizer_dir is not None else None
        self.tok = tok
        self.max = max_entries
        self.log = log
        self._lock = threading.Lock()
        self._prototypes: collections.OrderedDict[str, Any] = collections.OrderedDict()
        self._vocab = None
        self._off = ""
        self.hits = self.misses = self.evictions = 0

    # ------------------------------------------------------------------ state
    def _vocabulary(self):
        """The pack's vocabulary, built once.  None (and the pool is off) when the library is missing."""
        if self._vocab is not None:
            return self._vocab
        if self._off:
            return None
        from serve.grammar import matcher as M
        try:
            if self.dir is None:
                raise M.LibraryMissing("no tokenizer directory for the grammar pool")
            self._vocab = M.vocabulary(self.dir, self.tok)
        except Exception as e:                           # noqa: BLE001 - a missing library is a normal state here
            self._off = str(e) if isinstance(e, M.LibraryMissing) else f"{type(e).__name__}: {e}"
            self.log(f"[strata] grammar: no matcher pool ({self._off}). Nothing is constrained until "
                     f"tools/build_llguidance.sh has been run")
            return None
        return self._vocab

    @property
    def disabled(self) -> str:
        """Why the pool cannot compile anything ("" while it can).  Asks the vocabulary, so it costs nothing."""
        if not self._off:
            self._vocabulary()
        return self._off

    # ------------------------------------------------------------------ requests
    def acquire(self, grammar_id: str, text: str):
        """A matcher for one request: a clone of the prototype, compiled on first use.  None when unavailable."""
        vocab = self._vocabulary()
        if vocab is None:
            return None
        with self._lock:                                 # compiling and cloning under one lock: both are short,
            prototype = self._prototypes.get(grammar_id)  # and a duplicate compile is the only thing racing buys
            if prototype is not None:
                self._prototypes.move_to_end(grammar_id)
                self.hits += 1
            else:
                try:
                    prototype = vocab.new_matcher(text)
                except Exception as e:                   # noqa: BLE001 - one bad grammar must not stop the server
                    self.log(f"[strata] grammar {grammar_id}: cannot compile ({type(e).__name__}: {e}); this "
                             f"request decodes unconstrained")
                    return None
                self.misses += 1
                self._prototypes[grammar_id] = prototype
                while len(self._prototypes) > self.max:
                    _, gone = self._prototypes.popitem(last=False)
                    gone.close()
                    self.evictions += 1
            try:
                return prototype.clone()
            except Exception as e:                       # noqa: BLE001
                self.log(f"[strata] grammar {grammar_id}: cannot clone ({type(e).__name__}: {e}); this request "
                         f"decodes unconstrained")
                return None

    @staticmethod
    def release(matcher) -> None:
        """Free a request's matcher.  A clone is never returned to the pool: its position is spent, and a failed
        `consume` leaves it in a permanent error state."""
        if matcher is not None:
            matcher.close()

    def close(self) -> None:
        with self._lock:
            while self._prototypes:
                _, prototype = self._prototypes.popitem(last=False)
                prototype.close()

    # ------------------------------------------------------------------ counters, for the exit deliverable
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                    "prototypes": len(self._prototypes), "max": self.max, "off": self._off}

    def __enter__(self) -> "Pool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
