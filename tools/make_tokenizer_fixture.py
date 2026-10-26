"""tools/make_tokenizer_fixture.py - the stand-in tokenizer the grammar's compile tests use.

WHY A FIXTURE AT ALL.  The compile ladder (D-064) compiles the grammar in-process, which needs a
`serve.grammar.matcher.Vocabulary`: `vocab.json`, `token_type.json` and `merges.txt` in one directory.
The tests used to run against a real pack, which a checkout does not have, and a ladder that silently
skips is a ladder that does not exist.  So this script generates the three files into
`serve/fixtures/tokenizer/` and they are committed: reviewable, deterministic, and small.

WHAT IT IS NOT.  Not the pack's tokenizer, and not trying to be.  The ids differ from any pack's, and
nothing here reproduces a real BPE.  That is sound because the ladder's facts are schema-shape facts -
an unknown format, an unparsable regex, a missing `$ref`, a depth limit, an unsatisfiable property -
which depend on llguidance reading the schema, not on which ids the model's vocabulary happens to have.
Mask SIZES do differ from a real pack, so no test may assert a count (D-064).

WHAT IS IN IT (all of it, so a diff can be read):

  * 256 single-byte tokens, bytes 0x00..0xff in the inverted GPT-2 byte alphabet.  `Vocabulary` maps a
    token's characters back through that alphabet, so these cover every byte of any grammar text and of
    any UTF-8 JSON string a reply can produce.
  * a few dozen fragments (`{"`, `":`, `": "`, `true`, `answer`, `\n\n`, ...).  They are not there for
    `encode`: they are there so a mask is small, because a mask counts the ids whose bytes the grammar
    allows, and one id that spells `{"` is one id where the bytes allow two.
  * the delimiters the server names, as special tokens: the seven in `serve/grammar/guard.py::
    DELIMITER_IDS` plus the two the frontend marks as tool literals (`<tool_response>`,
    `</tool_response>`).  Type 3 is CONTROL (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`) and type 4
    is USER_DEFINED, so `<|im_end|>` is only matched with `parse_special=True` - which is exactly how
    `Vocabulary`'s EOS and the real pack behave.

ORDER MATTERS, AND IT IS ASSERTED BELOW.  `Vocabulary` builds `token_lens` indexed by id but the
`token_bytes` buffer by iterating `vocab.json`; the two only agree if the file lists ids in ascending
order.  Writing a dict from a set or sorted by text scrambles every token's bytes and the failure looks
like a bad grammar, not a bad fixture.

NO MERGES, DELIBERATELY.  `merges.txt` is a header and nothing else, which `Vocabulary` reduces to an
empty merge list.  With no rank a pre-tokenized piece stays single bytes, so `encode` is one id per
byte, and `decode(encode(s)) == s` for every `s` with no BPE rank order to get wrong.  (The header must
also be the last line: `Vocabulary` splits on "\\n" and drops only the first line, so a trailing newline
would hand it an empty merge, which `Tokenizer` rejects.  `tools/strata_tokenizer.py::extract` writes
`merges.txt` the same way, without a trailing newline.)

    python tools/make_tokenizer_fixture.py            # rewrite the three files
    python tools/make_tokenizer_fixture.py --check    # fail if they are not byte-identical
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "serve" / "fixtures" / "tokenizer"

sys.path.insert(0, str(ROOT / "tools"))
from strata_tokenizer import BYTE_TO_UNICODE      # noqa: E402 - the pack's own alphabet, not a copy

# Fragments, as the RAW BYTES the token must spell.  `_token` maps them through the byte alphabet, so a
# space here is byte 0x20 and becomes 'Ġ' in the file, exactly as the pack writes it.  Kept short and
# hand-readable on purpose: each line is a choice, and none is a single byte (those are already there).
# A doubled backslash below is a LITERAL backslash, which is what a JSON escape (`\n`, `\u`) is made of.
FRAGMENTS: list[bytes] = [
    # JSON punctuation and escapes: the shapes a schema reply is made of.
    b'{"', b'"}', b'":', b'": ', b'":{', b'",', b'", ', b'": "', b'"\n',
    b"{}", b"[]", b'["', b'"]', b": ", b", ", b",\\n", b"\n  ",
    b'\\"', b"\\n", b"\\u", b"\\\\",
    # JSON literals and the values the test schemas use.
    b"true", b"false", b"null", b"answer", b"celsius", b"fahrenheit", b"city", b"unit",
    # Function-call shapes (the skeletons' tool calls).
    b"function", b"parameters", b"arguments", b"name", b"tool", b"get_weather", b"list_files",
    # Ordinary prose, long enough to be worth one id.
    b"Hello", b"world", b"hello", b"the", b"and", b"that", b"with", b"this",
    b"from", b"have", b"ing", b"tion", b" a", b" I", b" is", b" of", b" to", b"yes", b"no",
    b"ok", b"the ",
    # Whitespace runs and line endings: reasoning text and formatted JSON.
    b"  ", b"    ", b"\n\n", b"\n\n\n", b".\n", b".\n\n", b"!\n", b"?\n",
    # Markdown and delimiters written as plain text (never as their special tokens).
    b"- ", b"**", b"()", b"\n<", b"</", b"/>",
]

# The delimiters, as (literal text, GGUF token type).  Text is NOT byte-mapped: `Vocabulary` prefixes a
# special token with 0xFF and passes this string through as written.
SPECIALS: list[tuple[str, int]] = [
    ("<|im_start|>", 3),
    ("<|im_end|>", 3),          # the EOS `Vocabulary` requires: `EOS_TEXT`
    ("<|endoftext|>", 3),
    ("<think>", 4),
    ("</think>", 4),
    ("<tool_call>", 4),
    ("</tool_call>", 4),
    ("<tool_response>", 4),
    ("</tool_response>", 4),
]

MERGE_HEADER = "#version: 0.2"


def _token(raw: bytes) -> str:
    """The token text a byte string gets in the pack's byte-level alphabet."""
    return "".join(BYTE_TO_UNICODE[b] for b in raw)


def tokens() -> list[str]:
    """Every token, in id order: 256 bytes, then the fragments, then the specials."""
    out = [_token(bytes([b])) for b in range(256)]
    seen = set(out)
    assert len(seen) == 256, "the byte alphabet is not injective"
    for raw in FRAGMENTS:
        if len(raw) < 2:
            raise AssertionError(f"fragment {raw!r} is a single byte, already covered")
        text = _token(raw)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    for text, _type in SPECIALS:
        if text in seen:
            raise AssertionError(f"special token {text!r} collides with an ordinary token")
        seen.add(text)
        out.append(text)
    return out


def build() -> dict[str, bytes]:
    """The three files' exact bytes.  `--check` compares these; `main` writes them."""
    ordered = tokens()
    vocab = {text: i for i, text in enumerate(ordered)}         # ascending ids, by construction
    types = [0] * (len(ordered) - len(SPECIALS)) + [ty for _text, ty in SPECIALS]
    assert [vocab[t] for t in ordered] == list(range(len(ordered)))
    assert len(vocab) == len(types) == len(ordered)
    # `json.dumps(indent=1)` puts one `"<token>": <id>` per line: a diff shows exactly which token moved.
    return {
        "vocab.json": (json.dumps(vocab, indent=1, ensure_ascii=False) + "\n").encode("utf-8"),
        "token_type.json": (json.dumps(types) + "\n").encode("utf-8"),
        "merges.txt": MERGE_HEADER.encode("utf-8"),             # no trailing newline: see the docstring
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="do not write; fail if the committed files differ from what this script generates")
    args = ap.parse_args(argv)
    files = build()
    OUT.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, data in files.items():
        path = OUT / name
        if args.check:
            on_disk = path.read_bytes() if path.exists() else None
            if on_disk != data:
                print(f"{path}: differs from what this script generates ({len(data)} bytes expected, "
                      f"{'missing' if on_disk is None else str(len(on_disk)) + ' bytes on disk'})",
                      file=sys.stderr)
                bad += 1
            else:
                print(f"{path}: identical ({len(data)} bytes)")
        else:
            path.write_bytes(data)
            print(f"{path}: {len(data)} bytes")
    if args.check and bad:
        return 1
    vocab = json.loads(files["vocab.json"])
    specials = {t: vocab[t] for t, _ty in SPECIALS}
    print(f"vocab {len(vocab)} ids, {len(SPECIALS)} specials {specials}, merges 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
