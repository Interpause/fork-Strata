"""The `rcall_specimens.json` corpus replayed through the decoder: every spelled call is prose.

    python -m unittest serve.test_grammar_corpus -v

The corpus was recorded for #804/#1058, where calls written *inside* the thinking block had to be recovered from the
text: 16 specimens are calls a text rule delivered, 21 are quotes that a text rule had to refuse.  The grammar makes
the 16 unrepresentable instead of recovered - the thinking body is a text terminal, so a `<tool_call>` written while
thinking is payload, and only the `<tool_call>` *token* opens a call - which is why the same 37 inputs now have one
expectation, not two.

So what this checks is not the old split but the property the split was standing in for: a delimiter spelled as
characters produces **no structure**, and none of the input is lost on the way.  The text parser's own replay of the
same file is `serve/test_reasoning_rescue.py`'s `Corpus`, which is what these 37 expectations used to be.
"""

import json
import unittest
from pathlib import Path

from serve.frontend import StructureParser

ROOT = Path(__file__).resolve().parents[1]
SPECIMENS = json.loads((ROOT / "serve/fixtures/rcall_specimens.json").read_text(encoding="utf-8"))["specimens"]

WIDTHS = (1, 2, 3, 7, 13)          # payload bytes per token: a delimiter must not depend on how it was split
PAYLOAD_ID = 1_000_000


def tools_of(specimen) -> list:
    """The specimen's tools, as the frontend passes them (the corpus records names only)."""
    return [{"name": name, "parameters": {"properties": {}}} for name in specimen["tools"]]


def replay(text: str, tools: list, width: int, stream_tools: bool = True):
    """The whole input as payload tokens, in `width`-byte pieces: what the model would emit if it spelled it."""
    parser = StructureParser("entry_think_tools", tools, stream_tools=stream_tools)
    events = []
    for i in range(0, len(text), width):
        events += parser.feed(PAYLOAD_ID + i, text[i:i + width])
    return events + parser.finish(), parser


class Corpus(unittest.TestCase):
    def test_no_specimen_produces_structure(self):
        """All 37, including the 16 the text parser delivered as calls: the bytes are spelled, so they are prose."""
        for specimen in SPECIMENS:
            tools = tools_of(specimen)
            seen = set()
            for width in WIDTHS:
                for stream in (True, False):
                    with self.subTest(name=specimen["name"], width=width, stream_tools=stream):
                        events, parser = replay(specimen["input"], tools, width, stream)
                        self.assertEqual([e.kind for e in events if e.kind in ("tool_call", "tool_start")], [])
                        # nothing is lost and nothing turns into an answer: the whole input is the thinking
                        self.assertEqual("".join(e.text for e in events if e.kind == "reasoning"),
                                         specimen["input"])
                        self.assertEqual([e.kind for e in events if e.kind != "reasoning"], [])
                        self.assertEqual((parser.rescued, parser.refused, parser.pending), (0, 0, []))
                        seen.add("".join(e.text for e in events))
            self.assertEqual(len(seen), 1, f"{specimen['name']}: the split changed what a client reads")

    def test_the_corpus_is_the_reported_one(self):
        """If the fixture changes, the replay above is a different check and should be looked at again."""
        self.assertEqual((len(SPECIMENS), sum(bool(s["calls"]) for s in SPECIMENS)), (37, 16))

    def test_a_specimen_read_as_an_answer_is_content_instead(self):
        """The same bytes in the answer phase are content: which channel a spelled call lands in is the entry's
        decision, and neither channel makes a call of it."""
        specimen = next(s for s in SPECIMENS if s["calls"])
        parser = StructureParser("entry_answer_tools", tools_of(specimen))
        events = []
        text = specimen["input"]
        for i in range(0, len(text), 7):
            events += parser.feed(PAYLOAD_ID + i, text[i:i + 7])
        events += parser.finish()
        self.assertEqual({e.kind for e in events}, {"content"}, "the pieces are content, and nothing else")
        self.assertEqual("".join(e.text for e in events), text)


if __name__ == "__main__":
    unittest.main()
