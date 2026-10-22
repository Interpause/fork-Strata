"""The request -> grammar layer: entry from the prompt's tail, splice cached by shape, id from the text.

    python -m unittest serve.test_grammar_pipeline -v

No library and no pack: this layer only needs a tokenizer, and a stub with the two methods it uses is enough.
"""
import hashlib
import json
import unittest

from serve.grammar import build as bg
from serve.grammar import entries as be
from serve.grammar import pipeline as pl

WEATHER = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}}}}}
LISTING = {"type": "function", "function": {"name": "list_files", "parameters": {"type": "object"}}}
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}

TAILS = {  # the prompts the template writes, as ids, for a stub tokenizer
    be.THINKING_ON_TAIL: [248045, 99, 248068, 11],
    be.THINKING_OFF_TAIL: [248045, 99, 248068, 11, 11, 248069, 11, 11],
}


class Stub:
    """A tokenizer that decodes the template's two tails and nothing else (the pipeline uses only `decode`)."""

    def __init__(self):
        self.encoded: dict[str, list[int]] = {}

    def encode(self, text, parse_special=False, plain=()):
        if text not in self.encoded:
            self.encoded[text] = [len(self.encoded) + 1] * max(1, len(text))
        return list(self.encoded[text])

    def decode(self, ids):
        for tail, tail_ids in TAILS.items():
            if list(ids)[-len(tail_ids):] == tail_ids:
                return "CONVERSATION TEXT" + tail
        return "SOMETHING ELSE"


def request(tail, tools=None, schema=None, required=False):
    ids = TAILS[tail] if tail in TAILS else [1, 2, 3]
    return ids, tools, schema, required


class Shapes(unittest.TestCase):
    def setUp(self):
        self.p = pl.Pipeline(Stub())

    def test_the_tail_decides_the_entry_family(self):
        on = self.p.for_request(*request(be.THINKING_ON_TAIL))
        off = self.p.for_request(*request(be.THINKING_OFF_TAIL))
        self.assertEqual(on.entry, "entry_think_plain")
        self.assertEqual(off.entry, "entry_answer_plain")
        self.assertIn("start: entry_think_plain", on.text)
        self.assertIn("start: entry_answer_plain", off.text)

    def test_tools_and_schema_pick_the_entry(self):
        for tail, phase in ((be.THINKING_ON_TAIL, "think"), (be.THINKING_OFF_TAIL, "answer")):
            with self.subTest(tail=tail):
                self.assertEqual(self.p.for_request(*request(tail, tools=[WEATHER])).entry,
                                 f"entry_{phase}_tools")
                self.assertEqual(self.p.for_request(*request(tail, tools=[WEATHER], required=True)).entry,
                                 f"entry_{phase}_tools_required")
                self.assertEqual(self.p.for_request(*request(tail, schema=SCHEMA)).entry, f"entry_{phase}_json")

    def test_a_prompt_that_ends_nowhere_is_an_error(self):
        with self.assertRaises(be.UnresolvedTail):
            self.p.for_request([1, 2, 3])

    def test_a_required_call_without_tools_is_an_error(self):
        with self.assertRaises(be.UnresolvedTail):
            self.p.for_request(*request(be.THINKING_OFF_TAIL, required=True))

    def test_the_identity_is_the_text_is_the_file_name(self):
        g = self.p.for_request(*request(be.THINKING_ON_TAIL, tools=[WEATHER]))
        self.assertEqual(g.id, hashlib.sha256(g.text.encode()).hexdigest()[:16])
        self.assertEqual(len(g.id), 16)
        self.assertTrue(all(c in "0123456789abcdef" for c in g.id))

    def test_the_text_is_the_one_the_builder_makes(self):
        # The pipeline must not grow a second splice.
        g = self.p.for_request(*request(be.THINKING_OFF_TAIL, tools=[WEATHER]))
        self.assertEqual(g.text, bg.for_entry("entry_answer_tools", [WEATHER], None))


class Cache(unittest.TestCase):
    def setUp(self):
        self.p = pl.Pipeline(Stub())

    def test_one_shape_is_one_splice(self):
        first = self.p.for_request(*request(be.THINKING_ON_TAIL, tools=[WEATHER]))
        second = self.p.for_request(*request(be.THINKING_ON_TAIL, tools=[WEATHER]))
        self.assertIs(first, second)
        self.assertEqual(self.p.stats(), {"hits": 1, "misses": 1, "shapes": 1, "max": pl.CACHE_MAX})

    def test_different_shapes_are_different_splices(self):
        self.p.for_request(*request(be.THINKING_ON_TAIL, tools=[WEATHER]))
        self.p.for_request(*request(be.THINKING_ON_TAIL, tools=[WEATHER, LISTING]))
        self.p.for_request(*request(be.THINKING_ON_TAIL, schema=SCHEMA))
        self.assertEqual(self.p.stats()["misses"], 3)

    def test_tool_order_is_part_of_the_shape(self):
        # The fragment's rule names carry the tool's index, so two orders are two grammars.
        a = self.p.for_request(*request(be.THINKING_ON_TAIL, tools=[WEATHER, LISTING]))
        b = self.p.for_request(*request(be.THINKING_ON_TAIL, tools=[LISTING, WEATHER]))
        self.assertNotEqual(a.id, b.id)
        self.assertEqual(self.p.stats()["misses"], 2)

    def test_the_bound_evicts_the_oldest(self):
        p = pl.Pipeline(Stub(), max_entries=2)
        for tools in ([WEATHER], [LISTING], [WEATHER, LISTING]):
            p.for_request(*request(be.THINKING_ON_TAIL, tools=tools))
        self.assertEqual(p.stats()["shapes"], 2)
        p.for_request(*request(be.THINKING_ON_TAIL, tools=[WEATHER]))     # the oldest shape is gone: a miss again
        self.assertEqual(p.stats()["misses"], 4)

    def test_the_schema_text_reaches_the_grammar(self):
        g = self.p.for_request(*request(be.THINKING_ON_TAIL, schema=SCHEMA))
        self.assertIn(json.dumps(SCHEMA), g.text)


if __name__ == "__main__":
    unittest.main()
