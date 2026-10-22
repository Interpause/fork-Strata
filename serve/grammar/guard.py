"""What has to still be true before the server may use the grammar, checked once at startup.

`skeleton.lark` bakes in two facts that live outside it: the pack tokenizer's seven delimiter ids, and the bytes the
chat template writes around a reply.  Nothing about a normal request would reveal that either had moved — the parser
would simply start refusing, or accepting, the wrong bytes — so both are asserted at startup and a mismatch stops the
server (D-013: for every pack, with no flag to skip it).

The remedy is the same for both failures, and the messages say it: the grammar is generated from the template and
the tokenizer, so it has to be re-derived when either changes.
"""

from __future__ import annotations

from typing import Callable

from serve.grammar.build import GrammarError
from serve.grammar.entries import THINKING_OFF_TAIL, THINKING_ON_TAIL

# The one place these numbers live in serve/.  The grammar names the delimiters, not their ids, so this table is the
# only thing tying the file to the gyro_l pack; a second pack with the same ids needs no change, a re-tokenized one
# does.
DELIMITER_IDS: dict[str, int] = {
    "<|im_start|>": 248045,
    "<|im_end|>": 248046,       # the pack's EOS, and the only token that ends a turn
    "<|endoftext|>": 248044,    # never generated: a turn ends at <|im_end|>
    "<think>": 248068,
    "</think>": 248069,
    "<tool_call>": 248058,
    "</tool_call>": 248059,
}

PROMPT_HEAD = "<|im_start|>assistant\n"


class GuardError(GrammarError):
    """The pack or the template no longer matches the grammar."""


def check_ids(encode: Callable[..., list[int]]) -> None:
    """Each delimiter must be exactly one token, with the id the grammar assumes.

    `encode` is the pack's own encoder, called the way the server calls it for text that may contain a delimiter.
    A delimiter that splits into several pieces is as wrong as one with a different id: the grammar names the
    token, so it could never be generated.
    """
    for text, want in DELIMITER_IDS.items():
        try:
            got = list(encode(text, parse_special=True))
        except Exception as e:                       # noqa: BLE001 - any failure here is the same failure
            raise GuardError(f"the pack's tokenizer cannot encode {text!r}: {e}") from e
        if got != [want]:
            raise GuardError(f"the pack's tokenizer encodes {text!r} as {got}, but the grammar assumes [{want}]: the "
                             f"pack is not the one serve/grammar/skeleton.lark was measured on, so pin the pack or "
                             f"re-export the grammar for this tokenizer")


def _probes() -> tuple[dict, list[dict], dict]:
    user = {"role": "user", "content": "guard_probe_question"}
    tool = {"type": "function", "function": {
        "name": "guard_probe_tool", "description": "guard_probe_description",
        "parameters": {"type": "object", "properties": {"guard_probe_arg": {"type": "string"}}}}}
    call = {"function": {"name": "guard_probe_tool", "arguments": {"guard_probe_arg": "guard_probe_value"}}}
    return user, [tool], call


def check_template(template) -> None:
    """The prompt tails the entry resolver keys on, and the three call shapes the grammar encodes.

    `template` is a `serve.frontend.ChatTemplate`.  Its own capability probes are separate and only hints; these
    renderings are assertions.
    """
    user, tools, call = _probes()
    for thinking, tail in ((True, THINKING_ON_TAIL), (False, THINKING_OFF_TAIL)):
        for named, toolset in ((False, None), (True, tools)):
            prompt = template.render([user], tools=toolset, enable_thinking=thinking)
            want = PROMPT_HEAD + tail
            if not prompt.endswith(want):
                raise GuardError(f"thinking {'on' if thinking else 'off'} with "
                                 f"{'tools' if named else 'no tools'}: the chat template no longer ends the prompt "
                                 f"with {want!r} (it ends with {prompt[-len(want) - 16:]!r}): the entry resolver "
                                 f"keys on those bytes, so re-export serve/grammar/skeleton.lark for this template")

    def answer(content: str, calls: int) -> str:
        turns = [user, {"role": "assistant", "content": content, "tool_calls": [call] * calls},
                 {"role": "tool", "content": "guard_probe_result"}, user]
        return template.render(turns, tools=tools, enable_thinking=False)

    # A call at byte 0 of the answer (content was empty), and a call after non-empty prose: two newlines, exactly.
    call_bytes = "<tool_call>\n<function=guard_probe_tool>\n<parameter=guard_probe_arg>\nguard_probe_value\n</parameter>\n</function>\n</tool_call>"
    for content, want in (("", THINKING_OFF_TAIL + call_bytes),
                          ("guard_probe_prose", "\n\n" + call_bytes)):
        prompt = answer(content, 1)
        if want not in prompt:
            raise GuardError(f"a tool call after {content!r} is no longer written as {want!r}: the grammar's call "
                             f"shape (and the separator before it) has to be re-derived from the template")
    # A second call in the same turn: one newline, no prose.
    double = call_bytes + "\n" + call_bytes
    if double not in answer("guard_probe_prose", 2):
        raise GuardError(f"two calls in one turn are no longer written as {double!r}: the grammar allows exactly "
                         f"one newline between calls, so re-export it for this template")


def check(encode: Callable[..., list[int]], template) -> None:
    """Both assertions.  Either one raises GuardError; the server turns that into a failed start."""
    check_ids(encode)
    check_template(template)
