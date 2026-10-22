"""The library that turns a grammar into a token mask, and the vocabulary it needs.

llguidance parses the grammar and answers one question per decoding step: which tokens may come next.  It does not
tokenize: the vocabulary is handed over once, as byte strings, and a token whose bytes start with 0xFF is a special
token the grammar may name (`<|im_end|>` in the Lark file, 0xFF 0x3C ... in the vocabulary).  That split is what
makes the masks exact, because the pack's own BPE is the only thing that decides which ids a piece of text becomes.

The vocabulary is built from the pack's `tokenizer/vocab.json` and `token_type.json`, with the GPT-2
byte-to-unicode alphabet inverted so every token becomes its true byte string.  It is not cheap (248,320 ids,
2,839,470 bytes) and it is built once per tokenizer directory, not per request.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any

# Where the library is, in order: the environment, then the checkout tools/build_llguidance.sh builds.
LIB_ENV = "STRATA_LLGUIDANCE_LIB"
LIB_DEFAULT = Path(__file__).resolve().parents[2] / "third_party" / "llguidance" / "target" / "release" / "libllguidance.so"


class LibraryMissing(RuntimeError):
    """libllguidance has not been built or cannot be loaded."""


def library_path() -> Path:
    """The shared object to load.  A path that does not exist is LibraryMissing, not a silent fallback."""
    named = os.environ.get(LIB_ENV, "").strip()
    path = Path(named).expanduser() if named else LIB_DEFAULT
    if not path.exists():
        raise LibraryMissing(f"{path} does not exist: run tools/build_llguidance.sh "
                             f"(or set {LIB_ENV} to a built libllguidance.so)")
    return path


# --------------------------------------------------------------------------- structs, from parser/llguidance.h
class LlgParserLimits(ctypes.Structure):
    _fields_ = [
        ("max_items_in_row", ctypes.c_size_t),
        ("initial_lexer_fuel", ctypes.c_uint64),
        ("step_lexer_fuel", ctypes.c_uint64),
        ("step_max_items", ctypes.c_size_t),
        ("max_lexer_states", ctypes.c_size_t),
        ("max_grammar_size", ctypes.c_size_t),
        ("precompute_large_lexemes", ctypes.c_bool),
        ("verbose_errors", ctypes.c_bool),
    ]


class LlgConstraintInit(ctypes.Structure):
    _fields_ = [
        ("tokenizer", ctypes.c_void_p),
        ("log_buffer_level", ctypes.c_uint32),
        ("log_stderr_level", ctypes.c_uint32),
        ("ff_tokens_ok", ctypes.c_bool),
        ("backtrack_ok", ctypes.c_bool),
        ("limits", LlgParserLimits),
    ]


TOKENIZE_FN = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
                               ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t)


class LlgTokenizerInit(ctypes.Structure):
    _fields_ = [
        ("vocab_size", ctypes.c_uint32),
        ("tok_eos", ctypes.c_uint32),
        ("token_lens", ctypes.POINTER(ctypes.c_uint32)),
        ("token_bytes", ctypes.POINTER(ctypes.c_uint8)),
        ("tokenizer_json", ctypes.c_char_p),
        ("tokenize_assumes_string", ctypes.c_bool),
        ("tokenize_fn", TOKENIZE_FN),
        ("use_approximate_greedy_tokenize_fn", ctypes.c_bool),
        ("tokenize_user_data", ctypes.c_void_p),
        ("slices", ctypes.POINTER(ctypes.c_char_p)),
    ]


# --------------------------------------------------------------------------- byte-level vocabulary
def _byte_to_unicode() -> dict[int, str]:
    """The GPT-2 byte encoder: 256 bytes -> 256 printable characters, reversibly."""
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


UNICODE_TO_BYTE = {v: k for k, v in _byte_to_unicode().items()}

# Token types a pack uses for a piece the grammar may name as one token: 3 = CONTROL, 4 = USER_DEFINED.
SPECIAL_TYPES = (3, 4)
EOS_TEXT = "<|im_end|>"


class Library:
    """The shared library, with every entry point this module uses given an explicit signature."""

    def __init__(self, path: Path | None = None):
        self.path = path or library_path()
        try:
            self.lib = ctypes.CDLL(str(self.path))
        except OSError as e:
            raise LibraryMissing(f"cannot load {self.path}: {e}") from e
        L = self.lib
        P = ctypes.POINTER
        L.llg_constraint_init_set_defaults.argtypes = [P(LlgConstraintInit), ctypes.c_void_p]
        L.llg_constraint_init_set_defaults.restype = None
        L.llg_new_matcher.argtypes = [P(LlgConstraintInit), ctypes.c_char_p, ctypes.c_char_p]
        L.llg_new_matcher.restype = ctypes.c_void_p
        L.llg_clone_matcher.argtypes = [ctypes.c_void_p]
        L.llg_clone_matcher.restype = ctypes.c_void_p
        L.llg_free_matcher.argtypes = [ctypes.c_void_p]
        L.llg_free_matcher.restype = None
        L.llg_new_tokenizer.argtypes = [P(LlgTokenizerInit), ctypes.c_char_p, ctypes.c_size_t]
        L.llg_new_tokenizer.restype = ctypes.c_void_p
        L.llg_free_tokenizer.argtypes = [ctypes.c_void_p]
        L.llg_free_tokenizer.restype = None
        L.llg_matcher_is_error.argtypes = [ctypes.c_void_p]
        L.llg_matcher_is_error.restype = ctypes.c_bool
        L.llg_matcher_get_error.argtypes = [ctypes.c_void_p]
        L.llg_matcher_get_error.restype = ctypes.c_char_p
        L.llg_matcher_compute_mask.argtypes = [ctypes.c_void_p]
        L.llg_matcher_compute_mask.restype = ctypes.c_int32
        L.llg_matcher_get_mask.argtypes = [ctypes.c_void_p]
        L.llg_matcher_get_mask.restype = P(ctypes.c_uint32)
        L.llg_matcher_get_mask_byte_size.argtypes = [ctypes.c_void_p]
        L.llg_matcher_get_mask_byte_size.restype = ctypes.c_size_t
        L.llg_matcher_consume_token.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        L.llg_matcher_consume_token.restype = ctypes.c_int32
        L.llg_matcher_validate_tokens.argtypes = [ctypes.c_void_p, P(ctypes.c_uint32), ctypes.c_size_t]
        L.llg_matcher_validate_tokens.restype = ctypes.c_int32
        L.llg_matcher_is_accepting.argtypes = [ctypes.c_void_p]
        L.llg_matcher_is_accepting.restype = ctypes.c_bool
        L.llg_matcher_is_stopped.argtypes = [ctypes.c_void_p]
        L.llg_matcher_is_stopped.restype = ctypes.c_bool
        L.llg_matcher_reset.argtypes = [ctypes.c_void_p]
        L.llg_matcher_reset.restype = ctypes.c_int32
        # Counts TOKENS, not bytes: it forwards to the byte-based rollback through the token lengths.
        L.llg_matcher_rollback.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        L.llg_matcher_rollback.restype = ctypes.c_int32
        L.llg_get_version.argtypes = []
        L.llg_get_version.restype = ctypes.c_char_p

    def version(self) -> str:
        return (self.lib.llg_get_version() or b"").decode()


class Vocabulary:
    """One pack tokenizer's ids as llguidance wants them: lengths, bytes, and the EOS id."""

    def __init__(self, library: Library, tokenizer_dir: Path, tok: Any = None):
        self.lib = library
        self.dir = Path(tokenizer_dir)
        vocab = json.loads((self.dir / "vocab.json").read_text(encoding="utf-8"))
        types = json.loads((self.dir / "token_type.json").read_text(encoding="utf-8"))
        if len(vocab) != len(types):
            raise ValueError(f"{self.dir}: vocab.json has {len(vocab)} entries, token_type.json {len(types)}")
        self.n = len(vocab)
        self.text = [None] * self.n
        self.type = types
        self.special: dict[str, int] = {}
        lens = [0] * self.n
        flat = bytearray()
        for text, i in vocab.items():
            if types[i] in SPECIAL_TYPES:
                b = b"\xff" + text.encode("utf-8")     # 0xFF is how a grammar names a special token
                self.special[text] = i
            else:
                b = bytes(UNICODE_TO_BYTE[ch] for ch in text)
            self.text[i] = text
            lens[i] = len(b)
            flat += b
        if EOS_TEXT not in self.special:
            raise ValueError(f"{self.dir}: no {EOS_TEXT} token, so there is no end of turn to constrain")
        self.eos = self.special[EOS_TEXT]
        self._lens = (ctypes.c_uint32 * self.n)(*lens)
        self._flat = (ctypes.c_uint8 * len(flat)).from_buffer_copy(bytes(flat))

        # Exact encode/decode.  The pack's own BPE is the only thing that may decide an id; llguidance never does.
        if tok is None:
            import sys
            tools = str(Path(__file__).resolve().parents[2] / "tools")
            if tools not in sys.path:
                sys.path.insert(0, tools)
            import strata_tokenizer
            merges = (self.dir / "merges.txt").read_text(encoding="utf-8").split("\n")[1:]   # first line is a header
            tok = strata_tokenizer.Tokenizer(list(vocab.keys()), merges, types)
        self.tok = tok
        self._tok_ptr = self._new_tokenizer()

    def _new_tokenizer(self) -> int:
        ti = LlgTokenizerInit()
        ti.vocab_size = self.n
        ti.tok_eos = self.eos
        ti.token_lens = self._lens
        ti.token_bytes = self._flat
        # No fast-forward yet, so no tokenizer callback: the C API accepts a null one only together with the
        # approximate greedy flag (HANDOVER.md section 6).
        ti.tokenize_fn = TOKENIZE_FN()
        ti.use_approximate_greedy_tokenize_fn = True
        err = ctypes.create_string_buffer(4096)
        ptr = self.lib.lib.llg_new_tokenizer(ctypes.byref(ti), err, len(err))
        if not ptr:
            raise RuntimeError(f"llg_new_tokenizer failed: {err.value.decode()}")
        return ptr

    # ----------------------------------------------------------------- encoding, for tests and for prompts
    def encode(self, text: str, parse_special: bool = False, plain: tuple = ()) -> list[int]:
        return self.tok.encode(text, parse_special=parse_special, plain=plain)

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids))

    def ids(self, text: str, parse_special: bool = True) -> list[int]:
        """Ids for grammar-shaped text, where a delimiter must be its one special id."""
        return self.encode(text, parse_special=parse_special)

    def new_matcher(self, grammar: str, kind: str = "lark") -> "Matcher":
        ci = LlgConstraintInit()
        self.lib.lib.llg_constraint_init_set_defaults(ctypes.byref(ci), self._tok_ptr)
        ci.log_stderr_level = 0        # a refused draft token is normal, not something to print
        ci.log_buffer_level = 1        # ...but keep the message, so error() can say what failed
        ci.limits.verbose_errors = False   # without the 10 KB grammar and the token history in every message
        ptr = self.lib.lib.llg_new_matcher(ctypes.byref(ci), kind.encode(), grammar.encode())
        if not ptr:
            raise RuntimeError("llg_new_matcher returned NULL")
        return Matcher(self, ptr)

    def close(self) -> None:
        if getattr(self, "_tok_ptr", None):
            self.lib.lib.llg_free_tokenizer(self._tok_ptr)
            self._tok_ptr = None


class Matcher:
    """One decoding state.  A request owns one; every speculative draft validates on a clone of it."""

    def __init__(self, vocab: Vocabulary, ptr: int):
        self.vocab = vocab
        self.lib = vocab.lib.lib
        self.ptr = ptr

    # ------------------------------------------------------------------ questions
    def error(self) -> str | None:
        e = self.lib.llg_matcher_get_error(self.ptr)
        return e.decode() if e else None

    def is_error(self) -> bool:
        return bool(self.lib.llg_matcher_is_error(self.ptr))

    def is_accepting(self) -> bool:
        return bool(self.lib.llg_matcher_is_accepting(self.ptr))

    def is_stopped(self) -> bool:
        return bool(self.lib.llg_matcher_is_stopped(self.ptr))

    def mask(self) -> tuple[list[int], int]:
        """The ids this state allows, and the library's return code (negative means the grammar failed)."""
        rc = self.lib.llg_matcher_compute_mask(self.ptr)
        if rc != 0:
            return [], rc
        nbytes = self.lib.llg_matcher_get_mask_byte_size(self.ptr)
        words = (ctypes.c_uint32 * (nbytes // 4)).from_address(ctypes.addressof(self.lib.llg_matcher_get_mask(self.ptr).contents))
        return [i for i in range(self.vocab.n) if words[i // 32] & (1 << (i % 32))], 0

    def allowed(self, token: int) -> bool:
        """Whether `token` is in the mask.  Same computation as mask(), without building the list."""
        rc = self.lib.llg_matcher_compute_mask(self.ptr)
        if rc != 0:
            return False
        nbytes = self.lib.llg_matcher_get_mask_byte_size(self.ptr)
        words = (ctypes.c_uint32 * (nbytes // 4)).from_address(ctypes.addressof(self.lib.llg_matcher_get_mask(self.ptr).contents))
        return bool(words[token // 32] & (1 << (token % 32)))

    def valid_prefix(self, ids) -> int:
        """How many of `ids` the grammar accepts, from this state.  Mutates nothing (the caller keeps the state)."""
        arr = (ctypes.c_uint32 * len(ids))(*ids)
        return int(self.lib.llg_matcher_validate_tokens(self.ptr, arr, len(ids)))

    # ------------------------------------------------------------------ transitions
    def consume(self, ids) -> bool:
        for i in ids:
            if self.lib.llg_matcher_consume_token(self.ptr, int(i)) != 0:
                return False
        return True

    def clone(self) -> "Matcher":
        ptr = self.lib.llg_clone_matcher(self.ptr)
        if not ptr:
            raise RuntimeError("llg_clone_matcher returned NULL")
        return Matcher(self.vocab, ptr)

    def rollback(self, n_tokens: int) -> bool:
        """Undo `n_tokens` tokens.  Counts tokens; a failed rollback leaves the matcher permanently in error."""
        return self.lib.llg_matcher_rollback(self.ptr, n_tokens) == 0

    def reset(self) -> bool:
        return self.lib.llg_matcher_reset(self.ptr) == 0

    def close(self) -> None:
        if self.ptr:
            self.lib.llg_free_matcher(self.ptr)
            self.ptr = None

    def __enter__(self) -> "Matcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_VOCABULARIES: dict[tuple[str, int], Vocabulary] = {}


def vocabulary(tokenizer_dir, tok: Any = None) -> Vocabulary:
    """The vocabulary for a tokenizer directory, built once.  Pass `tok` when the server already has one."""
    path = Path(tokenizer_dir).resolve()
    key = (str(path), id(tok) if tok is not None else 0)
    hit = _VOCABULARIES.get(key)
    if hit is None:
        hit = _VOCABULARIES[key] = Vocabulary(Library(), path, tok)
    return hit
