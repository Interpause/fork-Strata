"""The one tag the text reader still holds back: a `</think>` split across deltas.

    python -m unittest serve.test_parser_hold -v
"""
import random
import unittest

from serve.frontend import OutputParser

CLOSER = "</think>"
REPLIES = [
    "Reasoning.\n" + CLOSER + "\nAnswer café 🙂日本語\n",
    "Quoted `</think>` in prose, and a <tool_call> spelled in the thinking.",
    CLOSER + "\nThe closer came first.",
    "Thinking only, and then the end." + CLOSER,
    "<tool_call><function=lookup><parameter=q>abc</parameter></function></tool_call>\nDone",
    "An answer that never closed its thinking: café 🙂日本語 and a bare <tool_",
]


def joined(events, kind):
    return "".join(e.text for e in events if e.kind == kind)


def text_events(events):
    """What a client can tell: adjacent text of one kind is one event, so a chunk boundary is not shown."""
    out = []
    for ev in events:
        if out and out[-1][0] == ev.kind:
            out[-1] = (ev.kind, out[-1][1] + ev.text)
        else:
            out.append((ev.kind, ev.text))
    return out


class HoldTests(unittest.TestCase):
    def test_the_closer_is_held_until_it_is_whole(self):
        """Every proper prefix of `</think>` that ends the buffer is held, so a client never reads `</thi`."""
        for n in range(len(CLOSER)):                      # "", "<", "</", ... "</thin"
            with self.subTest(prefix=CLOSER[:n]):
                parser = OutputParser(thinking=True)
                events = parser.feed("reason") + parser.feed(CLOSER[:n])
                self.assertEqual(text_events(events), [("reasoning", "reason")],
                                 "the prefix is never text: nothing more goes out while it is held")
                self.assertEqual(parser.buf, CLOSER[:n],
                                 "nothing is held for the empty prefix; a partial closer waits in the buffer")
                events = parser.feed(CLOSER[n:] + "Answer") + parser.finish()
                self.assertEqual(text_events(events), [("content", "Answer")],
                                 "once the closer is whole the reasoning ends and the answer is content")

    def test_nothing_else_is_held(self):
        """A partial `<tool_call>` is not a delimiter any more: the reasoning text goes out at once."""
        parser = OutputParser(thinking=True)
        self.assertEqual(joined(parser.feed("a <tool_c"), "reasoning"), "a <tool_c")
        parser = OutputParser(thinking=False)
        self.assertEqual(joined(parser.feed("and a <tool_c in the answer"), "content"),
                         "and a <tool_c in the answer")

    def test_any_split_gives_the_events_of_one_feed(self):
        """Chunk boundaries are not something a client can tell: the same bytes in random pieces are the same events."""
        rng = random.Random(1028)
        for text in REPLIES:
            with self.subTest(text=text):
                one = OutputParser(thinking=True)
                want = text_events(one.feed(text) + one.finish())
                for _ in range(5):
                    parser, events, pos = OutputParser(thinking=True), [], 0
                    while pos < len(text):
                        end = pos + rng.randint(1, 9)
                        events += parser.feed(text[pos:end])
                        pos = end
                    events += parser.finish()
                    self.assertEqual(text_events(events), want,
                                     "a random split reports what one feed reports")



if __name__ == "__main__":
    unittest.main()
