"""The grammar asset: the skeleton, the two splices and the entry table.

    python -m unittest serve.test_grammar -v

These checks need no library and no GPU.  That a grammar *compiles* and that its masks say what they should is
`test_grammar_matcher.py`, which skips when llguidance has not been built (`tools/build_llguidance.sh`).
"""
import json
import re
import unittest

from serve.grammar import build as bg
from serve.grammar import entries as be
from serve.grammar.tools_fragment import compile_tools_fragment, slugify

RULE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*(?:::_)?)\s*:", re.M)
# What Lark accepts as a rule name.  `RULE` above is deliberately loose (it reads back what any fragment wrote);
# this one is the language's rule, and it is what a slug has to satisfy before the grammar can compile at all.
LARK_RULE = re.compile(r"^[a-z][a-z0-9_]*$")

WEATHER = {"type": "function", "function": {
    "name": "get_weather",
    "description": "the weather",
    "parameters": {"type": "object", "required": ["city"], "properties": {
        "city": {"type": "string", "description": "where"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    }},
}}
NO_ARGS = {"type": "function", "function": {"name": "list_files", "parameters": {"type": "object"}}}
SCHEMA = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "string"}}}


def rules(grammar: str) -> set:
    return set(RULE.findall(grammar))


def json_payloads(grammar: str) -> list:
    """Every inline `%json` payload, cut out by brace matching: the payload is JSON, not Lark."""
    out, i = [], 0
    while True:
        i = grammar.find("%json ", i)
        if i < 0:
            return out
        i += len("%json ")
        depth, j, instr, esc = 0, i, False, False
        while j < len(grammar):
            c = grammar[j]
            if instr:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    instr = False
            elif c == '"':
                instr = True
            elif c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            j += 1
        out.append(grammar[i:j])
        i = j


class Skeleton(unittest.TestCase):
    def test_building_nothing_reproduces_the_skeleton(self):
        self.assertEqual(bg.build(), bg.skeleton())

    def test_each_splice_point_appears_exactly_once(self):
        text = bg.skeleton()
        for marker in (bg.JSON_BEGIN, bg.JSON_END, bg.TOOLS_BEGIN, bg.TOOLS_END):
            self.assertEqual(text.count(marker), 1, marker)

    def test_the_start_marker_appears_exactly_once(self):
        self.assertEqual(bg.skeleton().count(bg.START), 1)

    def test_every_static_entry_rule_is_present(self):
        present = rules(bg.skeleton())
        for phase in ("think", "answer"):
            for kind in ("plain", "tools", "tools_required", "json"):
                self.assertIn(f"entry_{phase}_{kind}", present)
        for mid in ("entry_call_name",):
            self.assertIn(mid, present)

    def test_a_duplicated_splice_marker_is_refused(self):
        doubled = bg.skeleton().replace(bg.JSON_END, bg.JSON_END + "\n" + bg.JSON_END)
        with self.assertRaises(bg.GrammarError):
            bg._splice(doubled, bg.JSON_BEGIN, bg.JSON_END, "x")

    def test_reversed_splice_markers_are_refused(self):
        text = bg.skeleton().replace(bg.JSON_BEGIN, "@@").replace(bg.JSON_END, bg.JSON_BEGIN).replace("@@", bg.JSON_END)
        with self.assertRaises(bg.GrammarError):
            bg._splice(text, bg.JSON_BEGIN, bg.JSON_END, "x")

    def test_selecting_an_entry_rewrites_only_that_line(self):
        out = bg.select_entry(bg.build(), "entry_answer_tools")
        self.assertIn("start: entry_answer_tools", out)
        self.assertNotIn(bg.START, out)
        self.assertEqual(out.count("entry_answer_tools"), bg.build().count("entry_answer_tools") + 1)

    def test_a_missing_start_marker_is_refused(self):
        with self.assertRaises(bg.GrammarError):
            bg.select_entry(bg.build().replace(bg.START, ""), "entry_think_plain")


class Splices(unittest.TestCase):
    def test_a_schema_is_pasted_verbatim_after_the_json_keyword(self):
        out = bg.build(schema=SCHEMA)
        self.assertIn("json_response: %json " + json.dumps(SCHEMA), out)
        self.assertEqual(rules(out), rules(bg.skeleton()))   # the splice adds no rule of its own

    def test_a_pretty_printed_schema_also_splices(self):
        out = bg.build(schema=SCHEMA)
        self.assertIn('"required": ["answer"]', out)

    def test_tools_splice_leaves_the_placeholder_behind(self):
        out = bg.build(tools=[WEATHER])
        self.assertNotIn("NEVER_MATCH", out)

    def test_no_tools_keeps_the_never_matching_placeholder(self):
        self.assertIn(compile_tools_fragment([]).text, bg.build())

    def test_a_schema_is_not_applied_when_tools_are_declared(self):
        # The template allows tool calls or one JSON object, never both, so tools win.
        out = bg.build(tools=[WEATHER], schema=SCHEMA)
        self.assertIn(bg.DEFAULT_JSON_RULE, out)

    def test_every_json_payload_parses(self):
        # The library parses these with serde_json; a payload that does not parse is a grammar that cannot compile,
        # and the mistake is easiest to see here.
        out = bg.build(tools=[
            {"type": "function", "function": {"name": "count", "parameters": {
                "type": "object", "properties": {
                    "n": {"type": "integer", "minimum": 1},
                    "named": {"type": "string", "enum": ["a", "b"]},
                }}}},
        ], schema=SCHEMA)
        payloads = json_payloads(out)
        self.assertEqual(len(payloads), 2)          # the integer parameter, plus the always-present json_response
        for payload in payloads:
            json.loads(payload)

    def test_the_response_schema_is_the_only_json_payload_in_json_mode(self):
        payloads = json_payloads(bg.build(schema=SCHEMA))
        self.assertEqual(payloads, [json.dumps(SCHEMA)])


class Fragment(unittest.TestCase):
    def test_dispatch_offers_exactly_the_declared_names(self):
        out = compile_tools_fragment([WEATHER, NO_ARGS]).text
        self.assertEqual(out.split("\n\n", 1)[0],
                         "tool_dispatch: call_t0_get_weather\n             | call_t1_list_files")

    def test_a_required_parameter_guards_the_exit(self):
        out = compile_tools_fragment([WEATHER]).text
        # one required of two parameters: the exit is guarded by bit 0 only
        self.assertIn('params_t0_get_weather::_ : "" %if is_ones([0:1])', out)
        self.assertIn("param_t0_get_weather_0", out)
        self.assertIn("param_t0_get_weather_1", out)

    def test_a_string_parameter_is_raw_and_an_enum_is_spelled_out(self):
        out = compile_tools_fragment([WEATHER]).text
        self.assertIn('"<parameter=city>\\n" raw_param_end', out)
        self.assertIn('"<parameter=unit>\\n" ("celsius" | "fahrenheit")', out)

    def test_a_singleton_string_type_list_is_raw(self):
        tool = {"type": "function", "function": {"name": "capture", "parameters": {
            "type": "object", "properties": {"value": {"type": ["string"]}}}}}
        out = compile_tools_fragment([tool]).text
        self.assertIn('"<parameter=value>\\n" raw_param_end', out)

    def test_a_singleton_string_type_list_keeps_its_pattern(self):
        tool = {"type": "function", "function": {"name": "capture", "parameters": {
            "type": "object", "properties": {"value": {"type": ["string"], "pattern": "^a+$"}}}}}
        out = compile_tools_fragment([tool]).text
        self.assertIn('"<parameter=value>\\n" /(?:a+)/', out)

    def test_a_multi_line_raw_value_is_not_restricted_to_one_line(self):
        # raw_param_end is the skeleton's lazy terminal; nothing here narrows it.
        self.assertIn("raw_param_end", compile_tools_fragment([WEATHER]).text)
        self.assertNotIn("[^\\n]", compile_tools_fragment([WEATHER]).text)

    def test_a_tool_without_parameters_exits_immediately(self):   # noqa: D401
        self.assertIn('params_t0_list_files: ""', compile_tools_fragment([NO_ARGS]).text)

    def test_a_non_string_parameter_is_json(self):
        tool = {"type": "function", "function": {"name": "count", "parameters": {
            "type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}}}}
        self.assertIn('"<parameter=n>\\n" %json {"type": "integer", "minimum": 1}', compile_tools_fragment([tool]).text)

    def test_a_string_union_keeps_the_raw_branch(self):
        tool = {"type": "function", "function": {"name": "maybe", "parameters": {
            "type": "object", "properties": {"v": {"type": ["string", "integer"]}}}}}
        out = compile_tools_fragment([tool]).text
        self.assertIn("raw_param_end |", out)
        self.assertIn('%json {"type": "integer"}', out)

    def test_a_ref_to_a_string_definition_is_inlined(self):
        tool = {"type": "function", "function": {"name": "ref", "parameters": {
            "type": "object", "$defs": {"City": {"type": "string", "enum": ["Oslo"]}},
            "properties": {"city": {"$ref": "#/$defs/City"}}}}}
        self.assertIn('("Oslo")', compile_tools_fragment([tool]).text)

    def test_two_names_differing_in_punctuation_get_different_slugs(self):
        self.assertNotEqual(slugify("a.b", 0), slugify("a-b", 1))
        self.assertEqual(slugify("a.b", 3), "t3_a_b")

    def test_a_slug_is_a_lark_rule_name_whatever_the_tool_name(self):
        # A Lark rule name is `[a-z][_a-z0-9]*`: an uppercase letter ends it, and the parser then reports
        # `Expected token ':', found 'TOKEN_NAME'` on a line Strata wrote itself.  That refusal used to reach the
        # client as an invalid request, so the fold is the whole fix and is pinned here for every spelling.
        for name in ("get_weather", "GET_WEATHER", "getWeather", "rememberYourself", "REMEMBER_YOURSELF",
                     "mcp__Fetch", "a.b", "naïve", "名前"):
            self.assertRegex(slugify(name, 0), LARK_RULE, name)

    def test_folding_case_cannot_collide_because_the_index_is_the_position(self):
        # The only thing that keeps two tools apart is their index: `Foo` and `foo` are two positions, so they are
        # two slugs, and no request can put two tools in one position.
        self.assertEqual(slugify("Foo", 0), "t0_foo")
        self.assertEqual(slugify("foo", 1), "t1_foo")
        self.assertEqual(slugify("GET_WEATHER", 2), "t2_get_weather")

    def test_an_uppercase_tool_name_writes_only_lowercase_rules(self):
        tool = {"type": "function", "function": {"name": "REMEMBER_YOURSELF", "parameters": {
            "type": "object", "required": ["note"], "properties": {"note": {"type": "string"}}}}}
        out = compile_tools_fragment([tool]).text
        for name in rules(out):
            self.assertRegex(name, LARK_RULE, name)
        # The literal the model must write keeps the request's own spelling: only the rule name is folded.
        self.assertIn('"REMEMBER_YOURSELF>\\n"', out)
        self.assertIn('"<parameter=note>\\n"', out)

    def test_the_fragment_names_no_tool_of_its_own(self):
        out = compile_tools_fragment([WEATHER]).text
        self.assertNotIn("example_function", out)


class Entries(unittest.TestCase):
    def test_thinking_on(self):
        self.assertEqual(be.resolve_entry("<|im_start|>assistant\n<think>\n"), "entry_think_plain")
        self.assertEqual(be.resolve_entry("<|im_start|>assistant\n<think>\n", tools=[WEATHER]),
                         "entry_think_tools")
        self.assertEqual(be.resolve_entry("<|im_start|>assistant\n<think>\n", tools=[WEATHER], required=True),
                         "entry_think_tools_required")
        self.assertEqual(be.resolve_entry("<|im_start|>assistant\n<think>\n", schema=SCHEMA), "entry_think_json")

    def test_thinking_off_starts_at_the_answer(self):
        # The template writes the empty reasoning block into the PROMPT, so the reply begins at the answer.
        tail = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.assertEqual(be.resolve_entry(tail), "entry_answer_plain")
        self.assertEqual(be.resolve_entry(tail, tools=[WEATHER]), "entry_answer_tools")
        self.assertEqual(be.resolve_entry(tail, tools=[WEATHER], required=True), "entry_answer_tools_required")
        self.assertEqual(be.resolve_entry(tail, schema=SCHEMA), "entry_answer_json")

    def test_thinking_off_never_selects_a_thinking_entry(self):
        # Both tails contain <think>; only the bytes after it may decide.
        for tools, schema, required in (([], None, False), ([WEATHER], None, False),
                                        ([WEATHER], None, True), ([], SCHEMA, False)):
            entry = be.resolve_entry("<think>\n\n</think>\n\n", tools=tools, schema=schema, required=required)
            self.assertTrue(entry.startswith("entry_answer_"), entry)

    def test_the_resolver_reads_only_the_end_of_a_prompt(self):
        tail = "user: hi\n<|im_end|>\n<|im_start|>assistant\n<think>\n"
        self.assertEqual(be.resolve_entry(tail), "entry_think_plain")

    def test_a_continuation_tail_starts_at_the_answer(self):
        # D-045: the budget wrap-up puts text, </think> and the template's two newlines into the PROMPT, so the next
        # pass begins at the answer.  Only the end bytes decide: what the model wrote before the token is irrelevant.
        for tail in ("</think>\n\n",
                     "I have thought about this long enough; time to give my answer.\n</think>\n\n"):
            self.assertEqual(be.resolve_entry(tail), "entry_answer_plain")
            self.assertEqual(be.resolve_entry(tail, tools=[WEATHER]), "entry_answer_tools")
            self.assertEqual(be.resolve_entry(tail, tools=[WEATHER], required=True),
                             "entry_answer_tools_required")
            self.assertEqual(be.resolve_entry(tail, schema=SCHEMA), "entry_answer_json")
        # THINKING_OFF_TAIL ends in the same bytes.  Both mean the answer, but the thinking-off branch must be checked
        # first, or the continuation branch would be the one explaining a prompt this grammar already had an entry for.
        self.assertTrue(be.THINKING_OFF_TAIL.endswith(be.CONTINUATION_TAIL))
        self.assertEqual(be.resolve_entry("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
                         "entry_answer_plain")
        # The thinking-on tail stops at <think> and one newline, so the continuation branch must not claim it.
        self.assertEqual(be.resolve_entry("<|im_start|>assistant\n<think>\n"), "entry_think_plain")
        self.assertEqual(be.resolve_entry("<|im_start|>assistant\n<think>\n", tools=[WEATHER]),
                         "entry_think_tools")
        # A prompt inside a call is checked before every thinking tail, and its last bytes are the name, not
        # </think> and the template's newlines: neither branch can be reached from it.
        call_tail = "<tool_call>\n<function=get_weather>\n"
        self.assertFalse(call_tail.endswith(be.CONTINUATION_TAIL))
        self.assertEqual(be.resolve_entry(call_tail, tools=[WEATHER, NO_ARGS]),
                         "entry_call_named_t0_get_weather")

    def test_a_named_call_in_the_prompt(self):
        self.assertEqual(be.resolve_entry("<tool_call>\n<function=get_weather>\n", tools=[WEATHER, NO_ARGS]),
                         "entry_call_named_t0_get_weather")
        self.assertEqual(be.resolve_entry("<tool_call>\n<function=list_files>\n", tools=[WEATHER, NO_ARGS]),
                         "entry_call_named_t1_list_files")

    def test_a_name_still_to_write(self):
        self.assertEqual(be.resolve_entry("<tool_call>\n<function=", tools=[WEATHER]), "entry_call_name")
        self.assertEqual(be.resolve_entry("<tool_call>\n<function=", tools=[WEATHER], required=True),
                         "entry_call_name")

    def test_an_undeclared_name_in_the_prompt_is_refused(self):
        with self.assertRaises(be.UnresolvedTail):
            be.resolve_entry("<tool_call>\n<function=delete_file>\n", tools=[WEATHER])

    def test_a_tail_in_no_known_state_is_refused(self):
        with self.assertRaises(be.UnresolvedTail):
            be.resolve_entry("<|im_start|>assistant\n")

    def test_requiring_a_call_without_tools_is_refused(self):
        with self.assertRaises(be.UnresolvedTail):
            be.resolve_entry("<think>\n", required=True)

    def test_every_entry_the_resolver_returns_exists_in_the_grammar(self):
        # A typo in the table must fail here, without needing the library.
        cases = [
            ({}, None, False), ({"tools": [WEATHER, NO_ARGS]}, None, False),
            ({"tools": [WEATHER]}, None, True), ({}, SCHEMA, False),
        ]
        tails = ["<think>\n", "<think>\n\n</think>\n\n", "<tool_call>\n<function=",
                 "<tool_call>\n<function=get_weather>\n"]
        for kw, schema, required in cases:
            present = rules(bg.build(kw.get("tools"), schema))
            for tail in tails:
                try:
                    entry = be.resolve_entry(tail, schema=schema, required=required, **kw)
                except be.UnresolvedTail:
                    continue          # an undeclared name is the point of another test
                self.assertIn(entry, present, (tail, kw, schema, required))

    def test_the_named_call_entries_come_from_the_fragment(self):
        present = rules(bg.build([WEATHER, NO_ARGS]))
        self.assertIn("entry_call_named_t0_get_weather", present)
        self.assertIn("entry_call_name_t1_list_files", present)


if __name__ == "__main__":
    unittest.main()
