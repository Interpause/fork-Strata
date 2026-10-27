"""The grammar's structure reported as the API's events, from the structured reader alone.

    python -m unittest serve.test_structure_parser -v

No library and no pack: `StructureParser` takes (token id, text) pairs and decides structure from the grammar's
labels, which are built from the request's tools.  The things checked here are:

* **The structured reader's own report.**  The kinds, the text, each call's name and arguments, and the JSON
  streamed for those arguments - what a client is told, stated as literal expectations.
* **The split-invariance property.**  How the reply was cut into tokens is not something a client can tell: every
  piece boundary gives one shape (`Split`).
* **The replies the grammar cannot produce.**  A spelled delimiter, a fence, a value holding a
  ``</parameter>``, a call that never closed: each is a case the text parser had to guess at, and the structured
  reader reports what the grammar left instead (`Impossible`).

P7 deleted the text reader that was the reference for an event-by-event parity comparison up to P6: an
unconstrained reply is impossible now, so there is no reply left for it to read.
"""

import json
import unittest

from serve.frontend import Event, StructureParser
from serve.grammar import decode as dec
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
    """The reply as bytes, every delimiter spelled: what the same reply would be with no id to mark it.  The
    end-of-turn token is left out because it is a stop id, so neither reader is ever fed it (see Service._run)."""
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
    A call's id is renamed in order of appearance, so shapes from two runs can be compared without sharing ids."""
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


def call(name="get_weather", params=(("city", "Paris"), ("days", "3")), close=True):
    parts = [Special("<tool_call>"), "\n<function=", f"{name}>\n"]
    for pname, value in params:
        parts += [f"<parameter={pname}>\n", value, "\n</parameter>\n"]
    parts += ["</function>\n"]
    if close:
        parts += [Special("</tool_call>")]
    return parts


class Split(unittest.TestCase):
    """How the reply was cut into tokens is not something a client can tell: every piece boundary gives one shape."""

    def test_the_split_of_the_tokens_does_not_change_anything(self):
        parts = (["Checking the weather.", "\n\n"] + call()
                 + ["\n"] + call(params=(("city", "Oslo"), ("days", "1"))) + [Special("<|im_end|>")])
        want = None
        for size in (0, 1, 2, 3, 5, 8, 13):
            with self.subTest(size=size):
                parser = StructureParser("entry_answer_tools", TOOLS)
                events = []
                for tid, piece in tokens(parts, size):
                    events += parser.feed(tid, piece)
                events += parser.finish()
                if want is None:
                    want = shape(events)          # size=0: one token per part, the reference
                self.assertEqual(shape(events), want)


class Differences(unittest.TestCase):
    """What the structured reader reports where the text parser had to guess, and why the grammar's answer is the
    one to trust."""

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
        # There is no fence tracker left to make this guess: the delimiters are simply not ids.
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



def events_of(parts, entry, tools=None, size=0):
    """One reply through the structure parser alone: (its events, the parser)."""
    parser = StructureParser(entry, TOOLS if tools is None else tools)
    out = []
    for tid, piece in tokens(parts, size):
        out += parser.feed(tid, piece)
    return out + parser.finish(), parser


def feeds(parts, entry, tools=None):
    """The same, for a reply that ends in an exception: what the decoder had reported before it."""
    parser = StructureParser(entry, TOOLS if tools is None else tools)
    out = []
    for tid, piece in tokens(parts):
        out += parser.feed(tid, piece)
    return out


class Impossible(unittest.TestCase):
    """A reply the grammar cannot produce, and what the parser does with it (p5.md C2's failures).

    Each test is one situation the text parser had to decide by looking at the bytes and the decoder cannot reach at
    all, so what matters is not that it behaves *well* but that the decision has no counterpart to disagree with: the
    bytes are prose, or the delimiter is in a state that does not take it, and the parser says so instead of
    guessing.  The mask makes the impossible ones unsamplable (`test_grammar_matcher`); these are the same cases one
    layer up, where a client would see the difference.
    """

    def test_a_prompt_opened_call_is_announced_once(self):
        """Failure 6: with thinking off, `tool_choice` has the server write the call's opening into the prompt.  Those
        tokens are not output, so the entry says the reply starts at the parameters and the call is announced from
        the entry - once, under one id, with the name the grammar chose."""
        parser = StructureParser("entry_call_named_t0_get_weather", TOOLS)
        started = parser.opening()
        self.assertEqual([e.kind for e in started], ["tool_start", "tool_args"])
        self.assertEqual(started[0].call.name, "get_weather")
        rest = []
        for tid, piece in tokens(["<parameter=city>\nParis\n</parameter>\n</function>\n",
                                  Special("</tool_call>")]):
            rest += parser.feed(tid, piece)
        rest += parser.finish()
        calls = [e.call for e in started + rest if e.call]
        self.assertEqual(len({c.id for c in calls}), 1, "the call is announced once, not re-opened by the body")
        final = [e for e in rest if e.kind == "tool_call"]
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0].call.name, "get_weather")
        self.assertEqual(final[0].call.arguments, {"city": "Paris"})
        self.assertEqual(final[0].call.id, started[0].call.id)

    def test_an_entry_that_still_owes_the_name_announces_nothing(self):
        """#7's other half: `entry_call_name` starts at `=`, so the name is output.  Nothing may be announced from
        the prompt there, or the call would be opened twice."""
        self.assertEqual(StructureParser("entry_call_name", TOOLS).opening(), [])

    def test_a_delimiter_inside_the_thinking_is_refused_not_rescued(self):
        """Rows 13, 22 and 50: a `<tool_call>` token cannot arrive in the thinking (THINK_BODY is a text terminal),
        so there is no call to hold, stream early, release after N bytes, or count.  Nothing is rescued on the text
        route either: it reports no calls at all (P6)."""
        with self.assertRaises(dec.StructureError) as caught:
            feeds(["I could call ", Special("<tool_call>")], "entry_think_tools")
        self.assertIn("'think'", str(caught.exception))

    def test_a_thinking_reply_has_nothing_to_rescue(self):
        """Row 50: the grammar's thinking body is a text terminal, so the only call in this reply is the real one -
        nothing was held, streamed early, or rescued from the reasoning."""
        events, _ = events_of(["Thinking.", Special("</think>"), "\n\n"] + call(), "entry_think_tools")
        self.assertEqual([e.kind for e in events if e.kind == "tool_call"], ["tool_call"])

    def test_a_json_body_inside_the_wrapper_is_not_a_production(self):
        """Rows 16 and 29: a JSON answer is one `%json` value with no wrapper.  `<tool_call>` is followed by `\n` and
        `<function=`, never by an object, so there is no fenced-JSON-call state to fall into."""
        with self.assertRaises(dec.StructureError):
            feeds([Special("<tool_call>"), '\n{"name": "get_weather"}', Special("</tool_call>")],
                  "entry_answer_tools")

    def test_two_calls_in_one_wrapper_are_not_a_production(self):
        """Row 30: `more_tool_calls` puts every later call in its own `<tool_call>` wrapper.  The merge the text
        parser does (and the `batch` axis behind it) describes a reply the grammar refuses."""
        body = call()[1:-1]                                     # one whole call body, without its wrapper
        with self.assertRaises(dec.StructureError):
            feeds([Special("<tool_call>"), *body, *body, Special("</tool_call>")], "entry_answer_tools")

    def test_a_bare_function_is_prose(self):
        """Rows 25 and 30: a `<function=` written without the wrapper is not a call, because the wrapper is one
        token and the call production starts with it."""
        events, _ = events_of(['<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n'],
                              "entry_answer_tools")
        self.assertEqual([e.kind for e in merge(events)], ["content"])
        self.assertIn("get_weather", events[0].text)

    def test_an_opener_in_fences_or_inline_code_is_prose(self):
        """Failure 1 and rows 10, 27: a fence or a backtick has no structural meaning, and the delimiters were never
        tokens, so the reply is reasoning text and nothing else - there is no fence tracker here to keep in step."""
        for text in ("```\n" + text_of(call()) + "\n```\n",
                     "Use `<tool_call>` then `<function=get_weather>` in your note.",
                     "I'll open a <tool_call> block, it is how calls look."):
            with self.subTest(text=text):
                events, _ = events_of([text], "entry_think_tools")
                self.assertEqual([e.kind for e in merge(events) if e.kind != "reasoning"], [])
                self.assertEqual("".join(e.text for e in events if e.kind == "reasoning"), text,
                                 "the bytes are the reply, in the reasoning")

    def test_no_text_event_carries_the_call(self):
        """Failure 7: the call's bytes are its arguments, never also content or reasoning."""
        events, _ = events_of(["Let me look.\n\n"] + call(), "entry_answer_tools")
        kinds = [e.kind for e in events]
        self.assertEqual(kinds[0], "content")
        self.assertEqual(kinds[-1], "tool_call")
        self.assertNotIn("content", kinds[1:], "the call's bytes are not re-emitted as content")
        spoken = "".join(e.text for e in events if e.kind in ("content", "reasoning"))
        for delimiter in ("<tool_call>", "<function=", "<parameter=", "</parameter>", "</function>"):
            self.assertNotIn(delimiter, spoken)

    def test_a_reply_with_no_structure_keeps_its_bytes(self):
        """Failure 9: for a reply with no structure, the events are the reply - the same bytes, in the same kinds,
        which is why the constrained path cannot change what a client reads.  A literal expectation, not a
        comparison with another reader, so nothing can drift along with it."""
        events, _ = events_of(["Paris is the capital of France."], "entry_answer_plain", tools=[])
        self.assertEqual([(e.kind, e.text) for e in events],
                         [("content", "Paris is the capital of France.")])
        events, _ = events_of(["Two plus two.", Special("</think>"), "\n\n", "Four."], "entry_think_tools")
        self.assertEqual([(e.kind, e.text) for e in events],
                         [("reasoning", "Two plus two."), ("content", "Four.")])

    def test_the_reason_the_turn_ended_changes_no_event(self):
        """Failure 8: `finish(reason)` takes the reason and ignores it.  A finish reason derived from the text
        (a held-call byte count) has no place here: a reply cut short is reported as the grammar left it."""
        cut = [Special("<tool_call>"),
               "\n<function=get_weather>\n<parameter=city>\nParis\n</para"]     # the closer is still arriving
        for reason in (None, "stop", "length", "cancel"):
            with self.subTest(reason=reason):
                events, _ = events_of(cut, "entry_answer_tools")
                self.assertEqual([e.kind for e in events if e.kind != "tool_args"], ["tool_start"])
                args = "".join(e.text for e in events if e.kind == "tool_args")
                self.assertIn(json.dumps("\n</para")[1:-1], args, "a held closer becomes value text, not a close")
                self.assertNotIn("}", args, "a cut call's JSON is never closed for the client")


if __name__ == "__main__":
    unittest.main()
