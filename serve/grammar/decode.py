"""Structure from the grammar, not from the bytes.

The matcher decides **legality**: only it may approve a token (`serve/grammar/matcher.py`, the engine's mask).  This
module decides **meaning**: which token is a delimiter and which is payload.  It is built from the grammar the server
authors - `tools_fragment.compile_tools_fragment` returns the text the engine compiles together with `Labels`, the
literals it wrote and what each one is - so it cannot describe a grammar other than the one being decoded (D-042).

Why the labels can be trusted, and why the bytes alone cannot be:

* A delimiter's ordinary-text pieces are legal *inside* a value and inside prose (`</`, `parameter`, `>`, spelled as
  separate BPE pieces, are just text), so nothing about a byte says "structure".
* A **special token id** cannot appear in a text terminal.  Measured with the real pack vocabulary: a free-text
  regex offers 0 of the 33 special ids at the start mask and only EOS afterwards
  (`tmp/grammar-explore/toy/freetext/report.md` §V2, `v2-run.txt`, `v4-run.txt` V4.1; reason in
  `tmp/grammar-explore/HANDOVER.md` §11 A10).  So a special id is structure, exactly and only.
* A parameter value closes only at a standalone `"\\n</parameter>\\n"` line (`tmp/grammar-explore/HANDOVER.md` §2 D5,
  §11 A7/X1), which is what `skeleton.lark`'s `raw_param_end` encodes and what the template writes.

The decoder works on **tokens**, so a delimiter split across two tokens is not a case it can get wrong, and there is
no "streamed versus whole" question to test: one code path, fed the same ids.  `feed` takes the id and the token's
text; a delimiter id is recognised by id and its text is ignored.

What it does not do: it does not build the `tool_args` JSON, and it does not decide what prose looks like for
display (fences, inline code).  It reports structure - `call_open`, `param`, `param_end`, `call_end` - and
`serve/frontend.py` turns that into the API's events.
"""

from __future__ import annotations

from dataclasses import dataclass

from serve.grammar.guard import DELIMITER_IDS
from serve.grammar.tools_fragment import CALL_END, PARAM_CLOSE, Labels

# The skeleton's own literals (skeleton.lark, section 2 "REPLY STRUCTURE").  They are checked against the file by
# test_grammar_decoder.py, so an edit to the skeleton that the decoder does not follow fails a test rather than
# silently mislabelling a reply.
EQUALS = "="                      # entry_call_name: "=" tool_dispatch ...
FUNC_OPEN = "\n<function="        # single_tool_call: <tool_call> "\n<function=" tool_dispatch
CALL_SEP = "\n"                   # more_tool_calls: ( "\n" single_tool_call )*
BLANK_LINE = "\n\n"              # thinking: THINK_BODY </think> "\n\n"

# For the drift tests: exactly the literals `skeleton.lark` writes (a stale constant here, or a new literal there,
# fails them), plus the fragment's own, which `tools_fragment.py` writes and labels.
SKELETON_LITERALS: tuple[str, ...] = (EQUALS, FUNC_OPEN, CALL_SEP, BLANK_LINE, CALL_END)
FRAGMENT_LITERALS: tuple[str, ...] = (PARAM_CLOSE,)

# The special ids, by name.  `guard.DELIMITER_IDS` is the one table, checked against the pack at startup.
_NAME_OF_ID: dict[int, str] = {tid: name for name, tid in DELIMITER_IDS.items()}

THINK_CLOSE = "</think>"
CALL_START = "<tool_call>"
CALL_STOP = "</tool_call>"
TURN_END = "<|im_end|>"

# Two newlines are the separator the template writes before a call; a third can only be prose (ANSWER_TEXT is greedy
# and PROSE_BEFORE_TOOLS' character before its two newlines is not a newline), so no more than two are ever held.
HOLD = 2


class StructureError(ValueError):
    """A byte the grammar says cannot be here.  The mask allowed the token, so this is a disagreement between this
    module and the grammar the engine compiled - a bug here, or a grammar the labels do not describe.  It is never a
    request-level refusal: `serve/frontend.py` ends the turn at the last structural position it agreed with."""


@dataclass(frozen=True)
class Step:
    """One piece of structure or payload, in reply order.  `kind` is the API's vocabulary one level down:

        reasoning   `text` of the reasoning block
        content     `text` of prose
        call_open   a call's name is known (`name`); its parameters follow
        param       `text` of the parameter `name`'s value (may arrive in pieces)
        param_end   the parameter `name`'s value is complete
        call_end    the call `name`'s body is complete
        turn_end    the end-of-turn token; the reply is over
    """

    kind: str
    name: str = ""
    text: str = ""


THINK = "think"
AFTER_THINKING = "after_thinking"
EQUAL = "equal"
NAME = "name"
PARAMS = "params"
VALUE = "value"
CLOSE = "close"
AFTER_CALL = "after_call"
OPENING = "opening"
ANSWER = "answer"
DONE = "done"


def _longest_prefix_suffix(text: str, word: str) -> int:
    """How many trailing bytes of `text` could still become `word`.  0 when the suffix cannot start `word`."""
    for n in range(min(len(text), len(word) - 1), 0, -1):
        if text.endswith(word[:n]):
            return n
    return 0


def _start_state(entry: str, labels: Labels) -> tuple[str, str]:
    """Where a decode of `entry` begins, and the tool name a mid-call entry already knows.

    The entry is `serve/grammar/entries.py`'s; the prompt's own tail chose it, so the reply starts mid-structure
    exactly where the prompt left it.
    """
    if entry.startswith("entry_think"):
        return THINK, ""
    if entry == "entry_call_name":
        return EQUAL, ""          # the prompt ended at "<function": "=" and the name are still to be written
    if entry.startswith("entry_call_named_"):
        slug = entry[len("entry_call_named_"):]
        for tool in labels.tools:
            if tool.slug == slug:
                return PARAMS, tool.name
        raise StructureError(f"{entry} names no tool this fragment declares")
    return ANSWER, ""


class Decoder:
    """One reply, fed token by token.

    `feed` returns the steps that token completed; `finish` flushes what the end of the reply makes unambiguous (a
    held newline in prose, a partial closer in a parameter value).  Nothing is held beyond 13 bytes of a possible
    `"\\n</parameter>\\n"` and 2 newlines of a possible call separator, so a stream is reported as it arrives.
    """

    def __init__(self, entry: str, labels: Labels | None = None):
        self.labels = labels or Labels()
        self.state, self.tool = _start_state(entry, self.labels)
        self.param = ""                      # the parameter whose value is being read
        self.buf = ""                        # payload bytes no transition has consumed
        self.held = ""                       # trailing newlines of prose: held, they may be the call separator
        self.tools = sorted(self.labels.tools, key=lambda t: len(t.literal), reverse=True)
        self.params = sorted(self.labels.params, key=lambda p: len(p.literal), reverse=True)

    # ------------------------------------------------------------------ input
    def feed(self, token_id: int, text: str = "") -> list[Step]:
        """One token: a delimiter by id (its text is ignored), or payload bytes."""
        out: list[Step] = []
        name = _NAME_OF_ID.get(token_id)
        if name is None:
            self.buf += text
        else:
            self._delimiter(name, out)
        self._pump(out)
        return out

    def finish(self) -> list[Step]:
        """The end of the reply: whatever is held is payload, because no delimiter can follow it.

        A reply cut mid-structure (at `max_tokens`, so the grammar never got its closing token) leaves the unread
        bytes here; they are returned as `content`, which is what `serve/frontend.py` does with an unterminated call
        today (#211).  The caller decides what to do with them.
        """
        out: list[Step] = []
        if self.state == VALUE and self.buf:
            out.append(Step("param", self.param, self.buf))
        elif self.state == THINK and self.buf:
            out.append(Step("reasoning", text=self.buf))
        elif self.state == ANSWER:
            if self.held:
                out.append(Step("content", text=self.held))
            if self.buf:
                out.append(Step("content", text=self.buf))
        elif self.buf:
            out.append(Step("content", text=self.buf))
        self.buf = ""
        self.held = ""
        return out

    # ------------------------------------------------------------------ transitions
    def _delimiter(self, name: str, out: list[Step]) -> None:
        if self.buf:
            # A delimiter token can only follow payload; the payload's own state decides what it means.
            self._pump(out)
            if self.buf:
                raise StructureError(f"{self.buf!r} is still unread when {name} arrived in state {self.state!r}")
        if name == THINK_CLOSE:
            if self.state != THINK:
                raise StructureError(f"</think> in state {self.state!r}")
            self.state = AFTER_THINKING
        elif name == CALL_START:
            if self.state == ANSWER:
                self.held = ""            # the two template newlines: the separator, never content
            elif self.state not in (AFTER_CALL, OPENING):
                raise StructureError(f"<tool_call> in state {self.state!r}")
            self.state = OPENING
        elif name == CALL_STOP:
            if self.state != CLOSE:
                raise StructureError(f"</tool_call> in state {self.state!r}")
            out.append(Step("call_end", self.tool))
            self.tool = ""
            self.state = AFTER_CALL
        elif name == TURN_END:
            if self.state not in (ANSWER, AFTER_CALL):
                raise StructureError(f"<|im_end|> in state {self.state!r}")
            if self.held:
                out.append(Step("content", text=self.held))
                self.held = ""
            out.append(Step("turn_end"))
            self.state = DONE
        else:                             # <think>, <|im_start|>, <|endoftext|>: the grammar never offers them
            raise StructureError(f"{name} is never generated, yet it arrived in state {self.state!r}")

    def _pump(self, out: list[Step]) -> None:
        while self.buf:
            if self.state == THINK:
                out.append(Step("reasoning", text=self.buf))
                self.buf = ""
                return
            if self.state == ANSWER:
                run = len(self.buf) - len(self.buf.rstrip("\n"))
                body, tail = self.buf[:len(self.buf) - run], self.buf[len(self.buf) - run:]
                if body:
                    if self.held:
                        out.append(Step("content", text=self.held))
                        self.held = ""
                    out.append(Step("content", text=body))
                self.buf = ""
                self.held += tail
                if len(self.held) > HOLD:
                    out.append(Step("content", text=self.held[:-HOLD]))
                    self.held = self.held[-HOLD:]
                return
            if self.state == VALUE:
                at = self.buf.find(PARAM_CLOSE)
                if at >= 0:
                    if at:
                        out.append(Step("param", self.param, self.buf[:at]))
                    out.append(Step("param_end", self.param))
                    self.buf = self.buf[at + len(PARAM_CLOSE):]
                    self.param = ""
                    self.state = PARAMS
                    continue
                keep = _longest_prefix_suffix(self.buf, PARAM_CLOSE)
                if len(self.buf) > keep:
                    out.append(Step("param", self.param, self.buf[:len(self.buf) - keep]))
                    self.buf = self.buf[len(self.buf) - keep:]
                return
            if not self._literal_step(out):
                return

    def _literal_step(self, out: list[Step]) -> bool:
        """Consume one structural literal from `buf`.  False: it is a proper prefix, so wait for more bytes."""
        state = self.state
        if state == AFTER_THINKING:
            return self._expect(BLANK_LINE, lambda: setattr(self, "state", ANSWER))
        if state == EQUAL:
            return self._expect(EQUALS, lambda: setattr(self, "state", NAME))
        if state == OPENING:
            return self._expect(FUNC_OPEN, lambda: setattr(self, "state", NAME))
        if state == AFTER_CALL:
            # One newline separates two calls; the turn's own end is a special token and arrives with no bytes.
            return self._expect(CALL_SEP, lambda: setattr(self, "state", OPENING))
        if state == NAME:
            for tool in self.tools:
                if self.buf.startswith(tool.literal):
                    self.buf = self.buf[len(tool.literal):]
                    self.tool = tool.name
                    out.append(Step("call_open", name=tool.name))
                    self.state = PARAMS
                    return True
            if any(t.literal.startswith(self.buf) for t in self.tools):
                return False
            raise StructureError(f"{self.buf!r} is not a declared tool name (state name)")
        if state == PARAMS:
            if self.buf.startswith(CALL_END):
                self.buf = self.buf[len(CALL_END):]
                self.state = CLOSE
                return True
            for param in self.params:
                if self.buf.startswith(param.literal):
                    self.buf = self.buf[len(param.literal):]
                    self.param = param.name
                    self.state = VALUE
                    return True
            if CALL_END.startswith(self.buf) or any(p.literal.startswith(self.buf) for p in self.params):
                return False
            raise StructureError(f"{self.buf!r} is neither a parameter nor the end of the call (state params)")
        if state == CLOSE:
            raise StructureError(f"{self.buf!r} arrived before </tool_call>, which is a token, not text")
        raise StructureError(f"{self.buf!r} arrived in state {state!r}, which expects no payload")

    def _expect(self, literal: str, then) -> bool:
        if self.buf.startswith(literal):
            self.buf = self.buf[len(literal):]
            then()
            return True
        if literal.startswith(self.buf):
            return False
        raise StructureError(f"{self.buf!r} does not start {literal!r} (state {self.state!r})")
