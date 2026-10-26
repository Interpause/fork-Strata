"""Which entry rule a request decodes from, decided by the bytes the prompt ends with.

The template has already written the state before the reply begins, so the tail *is* the state:

    ...<|im_start|>assistant\\n<think>\\n                    thinking on, nothing written yet
    ...<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n     thinking off: the block is prompt text, not output
    ...</think>\\n\\n                                       a continuation (the budget wrap-up): the answer
    ...<function=NAME>\\n                                   inside a call, the name already written
    ...<function                                          inside a call, the name still to write

**Key on the bytes, never on a `thinking` flag.**  Both thinking tails contain `<think>`; they differ only in what
follows it.  A boolean that is missing or defaulted would pick a thinking entry for a request that asked for no
thinking, and that entry's first mask would then permit exactly the reasoning phase the caller suppressed.

This is the one entry resolver.  The exploration carried three copies of this table; nothing else in the tree should
grow a fourth.
"""

from __future__ import annotations

import re
from typing import Any

from .tools_fragment import slugify

THINKING_ON_TAIL = "<think>\n"
THINKING_OFF_TAIL = "<think>\n\n</think>\n\n"
# D-045: the server can continue a reply in a second pass (the thinking budget's wrap-up appends `REASONING_WRAP_UP` -
# text, then the `</think>` token, then the template's two newlines - and continues).  Those are exactly `thinking`'s
# own last two terminals, so the reply then stands where the thinking-off prompt leaves it: at the answer.  Passing
# the first pass's grammar to that pass instead is D-045's measured failure (the answer reads as reasoning).
CONTINUATION_TAIL = "</think>\n\n"

_NAMED = re.compile(r"<function=([^>\n]+)>?\n?$", re.S)
_UNNAMED = re.compile(r"<function=?$")


class UnresolvedTail(ValueError):
    """The prompt does not end in a state this grammar has an entry for.

    `reason` says whose condition the unresolved tail is, which is what decides the HTTP status the refusal is
    reported with: `undeclared_call` and `required_without_tools` come from what the client sent, and `unknown_tail`
    is the server's own - every prompt a chat route decodes was rendered by this template, so a tail with no entry
    is template drift rather than a field the client can correct.
    """

    def __init__(self, message: str, reason: str = "unknown_tail"):
        super().__init__(message)
        self.reason = reason


def resolve_entry(tail: str, tools: list[dict[str, Any]] | None = None,
                  schema: dict[str, Any] | None = None, required: bool = False) -> str:
    """The entry rule name for a prompt ending at `tail`.

    `tail` may be the whole prompt or just its end; only the last bytes are read.

    `required` is `tool_choice: required`: the reply must contain a call, so the prose-only branch is dropped.  When
    the request also names one function, the caller builds the tool fragment from that function alone, which is what
    narrows the name position to it; this resolver needs no special entry for that (`HANDOVER.md` G15).
    """
    tools = tools or []

    named = _NAMED.search(tail)
    if named:
        name = named.group(1)
        for idx, tool in enumerate(tools):
            fn = tool.get("function", tool)
            if fn.get("name") == name:
                return f"entry_call_named_{slugify(name, idx)}"
        raise UnresolvedTail(f"the prompt is inside a call to {name!r}, which this request did not declare",
                             reason="undeclared_call")
    if _UNNAMED.search(tail):
        return "entry_call_name"

    if tail.endswith(THINKING_OFF_TAIL):
        phase = "answer"          # the empty reasoning block is in the prompt: the reply starts at the answer
    elif tail.endswith(THINKING_ON_TAIL):
        phase = "think"
    elif tail.endswith(CONTINUATION_TAIL):
        # D-045: `THINKING_OFF_TAIL` ends in this too, so it is checked first; the thinking tail stops at `<think>` and
        # a newline, so only a prompt the server continued past its own thinking can end here.
        phase = "answer"
    else:
        raise UnresolvedTail("the prompt ends in no state the grammar has an entry for")

    if tools:
        return f"entry_{phase}_tools_required" if required else f"entry_{phase}_tools"
    if required:
        raise UnresolvedTail("tool_choice requires a call but the request declared no tools",
                             reason="required_without_tools")
    if schema is not None:
        return f"entry_{phase}_json"
    return f"entry_{phase}_plain"
