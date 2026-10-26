"""The request -> grammar layer: entry from the prompt's tail, splice cached by shape, id from the text.

    python -m unittest serve.test_grammar_pipeline -v

The splice cases need no library and no pack: this layer only needs a tokenizer, and a stub with the two methods it
uses is enough.  The ladder cases (`Ladder` below) are the exception: compiling a schema is what decides whether a
repair was needed, so they build the committed stand-in vocabulary (`serve/fixtures/tokenizer`, D-064) and skip,
with the reason, when `libllguidance.so` has not been built.
"""
import hashlib
import json
import re
import unittest
from pathlib import Path

from serve.grammar import build as bg
from serve.grammar import entries as be
from serve.grammar import matcher as M
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


def vocabulary():
    """The committed stand-in tokenizer, or a skip that says why (D-064; `test_grammar_matcher`'s shape)."""
    try:
        return M.vocabulary(Path(__file__).with_name("fixtures") / "tokenizer")
    except M.LibraryMissing as e:               # no libllguidance: nothing here can compile
        raise unittest.SkipTest(f"the compile cases need libllguidance ({e})")


def rule_names(text):
    """The rules a grammar text declares, in file order: what a repair must not change."""
    return re.findall(r"(?m)^([A-Za-z_][A-Za-z0-9_]*):", text)


LOOKAHEAD = {"type": "object", "properties": {"a": {"type": "string", "pattern": "(?=a)"}}}
DEPTH_64 = {"type": "string"}
for _ in range(64):
    DEPTH_64 = {"type": "object", "properties": {"p": DEPTH_64}}
# Unrepairable by the ladder, and measured against the library: the compiler reports two contradictory limits and
# names no property, so there is no node for a rung to widen.  `minProperties`/`maxProperties` are both legal
# keywords here (the gate accepts the schema), and the only "repair" would be to guess which of the two the client
# meant - a decision, not a relaxation - so the request gets the compiler's own words (D-065).
UNREPAIRABLE = {"type": "object", "minProperties": 5, "maxProperties": 2}
BAD_TOOL = {"type": "function", "function": {"name": "search", "parameters": {
    "type": "object", "required": ["q"], "properties": {
        "q": {"type": "string", "pattern": "(?=a)"},
        "limit": {"type": "integer"},
    }}}}


class Message(unittest.TestCase):
    """Which tool parameter a compile message points at, without the library: the strings are the probe's."""

    # probe §E: the failing line is the one above the caret, and the parameter's name is the template literal.
    PARAM = '''at 189(22): invalid regex "(?:(?=a))" (in regex): regex parse error:
    (?:(?=a))
       ^^^
error: look-around, including look-ahead and look-behind, is not supported
 187 | call_t0_do_thing: "do_thing>\\n" params_t0_do_thing::0x0
 188 | // bit 0 = a (required)
 189 | param_t0_do_thing_0: "<parameter=a>\\n" /(?:(?=a))/ "\\n</parameter>\\n"
                            ^
 190 | // bit 1 = b (optional)
 191 | param_t0_do_thing_1: "<parameter=b>\\n" %json {"type": "integer"} "\\n</parameter>\\n"
'''
    SCHEMA = '''at 173(16): failed to compile JSON schema: regex parse error:
    (?=a)
    ^^^
error: look-around, including look-ahead and look-behind, is not supported
 172 | // >>> SPLICE_JSON_SCHEMA_BEGIN
 173 | json_response: %json {"type": "object", "properties": {"a": {"type": "string", "pattern": "(?=a)"}}}
                      ^
 174 | // <<< SPLICE_JSON_SCHEMA_END
'''

    def test_the_parameter_is_read_off_the_caret_line(self):
        self.assertEqual(pl.parameter_of(self.PARAM), (0, "a"))

    def test_the_rule_line_is_the_one_the_caret_points_at(self):
        self.assertTrue(pl.failing_line(self.PARAM).startswith("param_t0_do_thing_0: "))
        self.assertTrue(pl.failing_line(self.SCHEMA).startswith("json_response: %json "))

    def test_a_message_about_the_response_schema_names_no_parameter(self):
        # Do not guess: replacing a parameter here would relax one the compiler accepted.
        self.assertIsNone(pl.parameter_of(self.SCHEMA))

    def test_a_message_with_no_caret_names_no_parameter(self):
        self.assertIsNone(pl.parameter_of('failed to parse "%json": recursion limit exceeded at line 1 column 2492'))


class Ladder(unittest.TestCase):
    """The compiler wired into the pipeline: what a schema costs before a request is admitted.

    Skipped without `libllguidance.so` (`vocabulary` above): "it compiles" is the property every case here is about.
    The vocabulary is the committed stand-in, whose ids need not be a pack's - a ladder fact is a schema-shape fact
    (D-064), and no case below asserts a mask.
    """

    @classmethod
    def setUpClass(cls):
        cls.v = vocabulary()

    def test_a_schema_that_compiles_is_not_touched(self):
        p = pl.Pipeline(Stub(), compiler=self.v)
        g = p.for_entry("entry_answer_json", schema=SCHEMA)
        self.assertEqual(g.repairs, ())
        self.assertEqual(g.text, bg.for_entry("entry_answer_json", None, SCHEMA))
        self.assertEqual(p.stats(), {"hits": 0, "misses": 1, "shapes": 1, "max": pl.CACHE_MAX})

    def test_a_lookahead_pattern_is_dropped_at_rung_two(self):
        p = pl.Pipeline(Stub(), compiler=self.v)
        g = p.for_entry("entry_answer_json", schema=LOOKAHEAD)
        self.assertNotIn('"pattern"', g.text)
        self.assertEqual(rule_names(g.text), rule_names(bg.for_entry("entry_answer_json", None, LOOKAHEAD)))
        self.assertEqual([(s.rung, s.cls) for s in g.repairs], [(2, "unsupported")])
        self.assertIsNone(self.v.compile_error(g.text))

    def test_the_nesting_limit_is_capped_at_rung_two(self):
        p = pl.Pipeline(Stub(), compiler=self.v)
        g = p.for_entry("entry_answer_json", schema=DEPTH_64)
        self.assertEqual([(s.rung, s.cls) for s in g.repairs], [(2, "parse")])
        self.assertIsNone(self.v.compile_error(g.text))         # the text really compiles now
        self.assertLess(g.text.count('"properties":'), 64)      # the chain was cut, not just the deepest node

    def test_an_unrepairable_schema_is_the_clients_400(self):
        p = pl.Pipeline(Stub(), compiler=self.v)
        with self.assertRaises(pl.GrammarUnavailable) as caught:
            p.for_entry("entry_answer_json", schema=UNREPAIRABLE)
        e = caught.exception
        self.assertEqual(e.status, 400)
        self.assertEqual(e.kind, "invalid_request_error")
        self.assertIn("minProperties > maxProperties", str(e))  # the compiler's own words (D-065)
        self.assertIn("cannot be compiled", str(e))

    def test_a_bad_parameter_becomes_arbitrary_and_the_call_survives(self):
        p = pl.Pipeline(Stub(), compiler=self.v)
        g = p.for_entry("entry_answer_tools", tools=[BAD_TOOL])
        self.assertNotIn("(?=a)", g.text)
        # `{}` is any JSON value: the parameter keeps its `<parameter=q>` block and the model judges the value.
        self.assertIn('param_t0_search_0: "<parameter=q>\\n" %json {} "\\n</parameter>\\n"', g.text)
        self.assertIn('param_t0_search_1: "<parameter=limit>\\n" %json {"type": "integer"}', g.text)
        self.assertEqual(len(g.repairs), 1)
        self.assertEqual((g.repairs[0].rung, g.repairs[0].cls), (2, "unsupported"))
        self.assertIn("search", g.repairs[0].note)
        self.assertIn("'q'", g.repairs[0].note)
        self.assertIsNone(self.v.compile_error(g.text))

    def test_the_request_shape_is_not_mutated_by_a_repair(self):
        # The cache key is the request's own shape, so a repair must not reach back into it.
        p = pl.Pipeline(Stub(), compiler=self.v)
        tools = json.loads(json.dumps([BAD_TOOL]))
        p.for_entry("entry_answer_tools", tools=tools)
        self.assertEqual(tools, [BAD_TOOL])

    def test_no_compiler_is_the_pure_splice(self):
        p = pl.Pipeline(Stub())                          # what every Service without a pack_dir is not (D-064)
        self.assertEqual(p.for_entry("entry_answer_json", schema=LOOKAHEAD).repairs, ())
        self.assertEqual(p.for_entry("entry_answer_json", schema=LOOKAHEAD).text,
                         bg.for_entry("entry_answer_json", None, LOOKAHEAD))


if __name__ == "__main__":
    unittest.main()
