"""The grammar's structure reported as the API's events, against the text parser that reports them today.

    python -m unittest serve.test_structure_parser -v

No library and no pack: `StructureParser` takes (token id, text) pairs and decides structure from the grammar's
labels, which are built from the request's tools.  The two things checked here are:

* **Parity with `OutputParser`.**  A reply the grammar can produce is reported the same way by both, event for event
  and argument for argument, so swapping the reader changes nothing a client sees.  This is the check that lets the
  decoder take over the constrained path.
* **The differences that are the point.**  A reply the grammar cannot produce is where the two disagree, and each
  disagreement is a case the text parser has to guess at: a spelled delimiter, a fence, a value holding a
  ``</parameter>``, a call that never closed.
"""

import json
import unittest

from serve.frontend import Event, OutputParser, StructureParser, ToolCall
from serve.grammar import guard

WEATHER = {"name": "get_weather", "parameters": {"type": "object", "required": ["city"], "properties": {
    "city": {"type": "string"},
    "days": {"type": "integer"},
    "unit": {"type": "string"},
}}}
LISTING = {"name": "list_files", "parameters": {"type": "object"}}
TOOLS = [WEATHER, LISTING]

PAYLOAD_ID = 1_000_000


class Special:
    """A delimiter token: structure by id, whichever bytes it would be if spelled."""

    def __init__(self, name: str):
        self.name = name


def text_of(parts) -> str:
    """What the text parser is fed.  The end-of-turn token is not in it: it is a stop id, so the server breaks
    before feeding it (see Service._run), and the text parser never sees it either."""
    return "".join(p.name if isinstance(p, Special) else p for p in parts
                   if not (isinstance(p, Special) and p.name == "<|im_end|>"))


def tokens(parts, size: int = 0):
    """(id, text) pairs: specials by id, payload in `size`-byte pieces (0: one piece per part)."""
    out = []
    for part in parts:
        if isinstance(part, Special):
            out.append((guard.DELIMITER_IDS[part.name], ""))
        elif size:
            for i in range(0, len(part), size):
                out.append((PAYLOAD_ID + len(out), part[i:i + size]))
        else:
            out.append((PAYLOAD_ID + len(out), part))
    return out


def merge(events):
    """Adjacent text of one kind is one event: how the reply was split is not what a client is told."""
    out = []
    for ev in events:
        if out and out[-1].kind == ev.kind and ev.text and out[-1].call is None and ev.call is None:
            out[-1] = Event(ev.kind, out[-1].text + ev.text)
        else:
            out.append(Event(ev.kind, ev.text, ev.call))
    return out


def shape(events):
    """What a client can tell: the kinds and text, each call's name and arguments, and where its JSON streamed.
    A call's id is renamed in order of appearance, so the two parsers' random ids can be compared."""
    out, args, calls, index = [], {}, {}, {}

    def cid(call):
        return index.setdefault(call.id, len(index))

    for ev in merge(events):
        if ev.kind == "tool_call":
            calls[cid(ev.call)] = (ev.call.name, ev.call.arguments)
            out.append(("tool_call", ev.call.name, json.dumps(ev.call.arguments, sort_keys=True)))
        elif ev.kind in ("tool_start", "tool_args"):
            args[cid(ev.call)] = args.get(cid(ev.call), "") + (ev.text or "")
            if ev.kind == "tool_start":
                out.append(("tool_start", ev.call.name))
        else:
            out.append((ev.kind, ev.text))
    return out, args, calls


def both(parts, entry, thinking=True, size=0):
    """The same reply through both parsers: (text parser's shape, structure parser's shape)."""
    want = OutputParser(thinking=thinking, tools=TOOLS, stream_tools=True)
    text = text_of(parts)
    want_events = merge(want.feed(text) + want.finish())
    got = StructureParser(entry, TOOLS)
    got_events = []
    for tid, piece in tokens(parts, size):
        got_events += got.feed(tid, piece)
    got_events += got.finish()
    return shape(want_events), shape(got_events)


def call(name="get_weather", params=(("city", "Paris"), ("days", "3")), close=True):
    parts = [Special("<tool_call>"), "\n<function=", f"{name}>\n"]
    for pname, value in params:
        parts += [f"<parameter={pname}>\n", value, "\n</parameter>\n"]
    parts += ["</function>\n"]
    if close:
        parts += [Special("</tool_call>")]
    return parts


class Parity(unittest.TestCase):
    """A reply the grammar can produce: both parsers must report it identically."""

    def assertSame(self, parts, entry="entry_answer_tools", thinking=True):
        want, got = both(parts, entry, thinking)
        self.assertEqual(got[0], want[0], "the events differ")
        self.assertEqual(got[2], want[2], "the calls differ")
        for cid, streamed in got[1].items():
            self.assertEqual(json.loads(streamed), got[2][cid][1], "the streamed JSON is not the call's arguments")
        return got

    def test_prose_then_a_call(self):
        self.assertSame(["Checking ", "the weather.", "\n\n"] + call() + [Special("<|im_end|>")], thinking=False)

    def test_a_call_at_byte_zero(self):
        self.assertSame(call() + [Special("<|im_end|>")], entry="entry_answer_tools", thinking=False)

    def test_two_calls_in_one_turn(self):
        self.assertSame(call() + ["\n"] + call(params=(("city", "Oslo"), ("days", "1")))
                        + [Special("<|im_end|>")], thinking=False)

    def test_a_thinking_reply(self):
        self.assertSame(["\nLet me think.\n", Special("</think>"), "\n\n", "Answer.", Special("<|im_end|>")],
                        entry="entry_think_tools")

    def test_a_call_missing_only_its_closing_token_is_still_whole(self):
        # The body ended; only the wrapper's closing token did not arrive.  Its arguments are complete, so the call
        # is delivered - the shape the text parser has always accepted (`UnfinishedToolCall`).
        self.assertSame(["", Special("<tool_call>"), "\n<function=", "list_files>\n", "</function>\n"],
                        thinking=False)

    def test_a_call_with_no_parameters(self):
        self.assertSame(["", Special("<tool_call>"), "\n<function=", "list_files>\n", "</function>\n",
                         Special("</tool_call>"), Special("<|im_end|>")], thinking=False)

    def test_a_plain_answer_with_no_tools(self):
        self.assertSame(["just prose", Special("<|im_end|>")], entry="entry_answer_plain", thinking=False)

    def test_a_value_the_schema_does_not_call_a_string(self):
        # `days` is an integer: both parsers hand the client the JSON number, not the bytes.
        got = self.assertSame(call(params=(("city", "Paris"), ("days", "3"))) + [Special("<|im_end|>")],
                              thinking=False)
        self.assertEqual(got[2][next(iter(got[2]))][1], {"city": "Paris", "days": 3})

    def test_the_split_of_the_tokens_does_not_change_anything(self):
        parts = ["Checking the weather.", "\n\n"] + call() + [Special("<|im_end|>")]
        want = both(parts, "entry_answer_tools")[1]
        for size in (1, 2, 3, 5, 7, 13):
            with self.subTest(size=size):
                self.assertEqual(both(parts, "entry_answer_tools", size=size)[1], want)


class Differences(unittest.TestCase):
    """Where the two disagree, and why the grammar's answer is the one to trust."""

    def structure(self, parts, entry="entry_answer_tools", thinking=True):
        got = StructureParser(entry, TOOLS)
        out = []
        for tid, piece in tokens(parts):
            out += got.feed(tid, piece)
        return shape(merge(out + got.finish()))

    def test_a_parameter_value_may_hold_a_parameter_tag(self):
        value = "a </parameter> that is not alone on its line"
        got = self.structure(call(params=(("city", value),)) + [Special("<|im_end|>")], thinking=False)
        self.assertEqual(got[2][next(iter(got[2]))][1], {"city": value})

    def test_a_fenced_call_is_prose_because_its_delimiters_are_not_tokens(self):
        # The fence tracker exists in the text parser to make this guess; here the delimiters are simply not ids.
        fenced = "```\n<tool_call>\n<function=get_weather>\n</function>\n</tool_call>\n```"
        got = self.structure([fenced, Special("<|im_end|>")], thinking=False)
        self.assertEqual([k for k, *_ in got[0]], ["content"])
        self.assertEqual(got[2], {})

    def test_a_spelled_call_in_prose_is_prose(self):
        got = self.structure(["I will use <tool_call> blocks.", Special("<|im_end|>")], thinking=False)
        self.assertEqual(got[2], {})

    def test_a_call_cut_before_its_body_ends_is_not_a_call(self):
        # #211: it streamed, but the body never ended, so it is not announced as a call and its JSON is not closed
        # on its behalf.
        got = self.structure([Special("<tool_call>"), "\n<function=", "get_weather>\n",
                              "<parameter=city>\n", "Paris", "\n</parameter>\n"], thinking=False)
        self.assertEqual([k for k, *_ in got[0]], ["tool_start"])
        self.assertEqual(got[2], {})
        self.assertTrue(got[1][next(iter(got[1]))].startswith('{"city":"Paris"'))
        self.assertFalse(got[1][next(iter(got[1]))].endswith("}"))



if __name__ == "__main__":
    unittest.main()
