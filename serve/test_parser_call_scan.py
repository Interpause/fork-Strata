"""Wrapped-call precheck parity against the original structural walk."""
import itertools
import random
import unittest

from serve import frontend


class ReferenceParser(frontend.OutputParser):
    def _wrapped_call_end(self):
        return frontend.call_end(self.buf)


def events_without_ids(events):
    return [(e.kind, e.text, (e.call.name, e.call.arguments) if e.call else None) for e in events]


class CallScanTests(unittest.TestCase):
    def test_chunk_boundaries_literal_tags_multiple_calls_and_recovery(self):
        def call(value):
            return "<tool_call><function=write><parameter=text>" + value + "</parameter></function></tool_call>"
        bodies = [call("ordinary"), call("literal </tool_call> text </function> here"),
                  call("literal </parameter> prose"), call("a") + call("b"),
                  call("unfinished")[:-5], call("incomplete </tool_call> prose")[:-30],
                  "<tool_call><function=write></function></tool_call>",
                  "<function=write><parameter=text>bare</parameter></function>",
                  "<tool_call><function=write><parameter=text>a</parameter></function>"
                  "<parameter=write><parameter=text>b</parameter></function></tool_call>"]
        tools = [{"name": "write", "parameters": {"properties": {"text": {"type": "string"}}}}]
        rng = random.Random(1030)
        for text, stream in itertools.product(bodies, (False, True)):
            # Every fixed chunk width, plus random partitions; compare after EACH feed.
            for width in range(1, 16):
                a = frontend.OutputParser(thinking=False, tools=tools, stream_tools=stream)
                b = ReferenceParser(thinking=False, tools=tools, stream_tools=stream)
                pos = 0
                while pos < len(text):
                    end = pos + (rng.randint(1, 20) if width == 15 else width)
                    delta = text[pos:end]
                    self.assertEqual(events_without_ids(a.feed(delta)), events_without_ids(b.feed(delta)))
                    pos = end
                self.assertEqual(events_without_ids(a.finish()), events_without_ids(b.finish()))


if __name__ == "__main__":
    unittest.main()
