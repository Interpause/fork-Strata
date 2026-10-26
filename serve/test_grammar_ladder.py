"""The compile ladder without the library: every message is a verbatim capture from `tmp/research/ladder-probe.md`.

    python -m unittest serve.test_grammar_ladder -v

No `.so`, no tokenizer, no server.  The compiler is a fake that returns a canned message for a payload, which is
exactly the contract `serve/grammar/ladder.py` is built on: `compile_text(text) -> str | None`.  The message text
is copied out of the probe report, never invented, because the classifier is only as good as the strings
llguidance 1.9.1 actually prints.
"""

import copy
import json
import unittest

from serve.grammar import ladder

# --- probe §1, case 1: unknown format, with the envelope and the numbered source excerpt ---------------
FORMAT_MESSAGE = '''at 173(16): failed to compile JSON schema: Unknown format: nope
 171 | 
 172 | // >>> SPLICE_JSON_SCHEMA_BEGIN
 173 | json_response: %json {"type": "string", "format": "nope"}
                      ^
 174 | // <<< SPLICE_JSON_SCHEMA_END
 175 | 
'''

# --- probe §1, case 2: an unimplemented keyword ------------------------------------------------------
KEYS_MESSAGE = '''at 173(16): failed to compile JSON schema: Unimplemented keys: ["not"]
 171 | 
 172 | // >>> SPLICE_JSON_SCHEMA_BEGIN
 173 | json_response: %json {"type": "object", "properties": {"a": {"not": {}}}}
                      ^
 174 | // <<< SPLICE_JSON_SCHEMA_END
 175 | 
'''

# --- probe §1, case 3: look-ahead, envelope A --------------------------------------------------------
LOOKAHEAD_MESSAGE = '''at 173(16): failed to compile JSON schema: regex parse error:
    (?=a)
    ^^^
error: look-around, including look-ahead and look-behind, is not supported
 171 | 
 172 | // >>> SPLICE_JSON_SCHEMA_BEGIN
 173 | json_response: %json {"type": "object", "properties": {"a": {"type": "string", "pattern": "(?=a)"}}}
                      ^
 174 | // <<< SPLICE_JSON_SCHEMA_END
 175 | 
'''

# --- probe §1, case 4: a backreference.  The echoed text is the decoded pattern, `(a)\1` ---------------
BACKREF_MESSAGE = r'''at 173(16): failed to compile JSON schema: regex parse error:
    (a)\1
       ^^
error: backreferences are not supported
 171 | 
 172 | // >>> SPLICE_JSON_SCHEMA_BEGIN
 173 | json_response: %json {"type": "object", "properties": {"ok": {"type": "string"}, "bad": {"type": "string", "pattern": "(a)\\1"}}}
                      ^
 174 | // <<< SPLICE_JSON_SCHEMA_END
 175 | 
'''

# --- probe §1, case 6: a $ref that does not resolve ---------------------------------------------------
REFERENCE_MESSAGE = '''at 173(16): failed to compile JSON schema: Pointer '/$defs/missing' does not exist
 171 | 
 172 | // >>> SPLICE_JSON_SCHEMA_BEGIN
 173 | json_response: %json {"type": "object", "properties": {"a": {"$ref": "#/$defs/missing"}}}
                      ^
 174 | // <<< SPLICE_JSON_SCHEMA_END
 175 | 
'''

# --- probe §1, case 7: invalid, and it names the keyword ----------------------------------------------
INVALID_MESSAGE = '''at 173(16): failed to compile JSON schema: prefixItems must be an array
 171 | 
 172 | // >>> SPLICE_JSON_SCHEMA_BEGIN
 173 | json_response: %json {"type": "array", "prefixItems": {}}
                      ^
 174 | // <<< SPLICE_JSON_SCHEMA_END
 175 | 
'''

# --- probe §1, case 8: a resource limit ---------------------------------------------------------------
RESOURCE_MESSAGE = '''at 173(16): failed to compile JSON schema: schema too large
 171 | 
 172 | // >>> SPLICE_JSON_SCHEMA_BEGIN
 173 | json_response: %json {"anyOf": [{"const": 0}, {"const": 1}, …, {"const": 39999}]}
                      ^
 174 | // <<< SPLICE_JSON_SCHEMA_END
 175 | 
'''

# --- probe §1, case 9: envelope B, and the only exact offset -------------------------------------------
PARSE_DEPTH_MESSAGE = 'failed to parse "%json": recursion limit exceeded at line 1 column 2492'

# --- probe §1, case 11: the one message that names a property ------------------------------------------
UNSATISFIABLE_MESSAGE = '''at 173(16): failed to compile JSON schema: Unsatisfiable schema: required property 'a' is unsatisfiable
 171 | 
 172 | // >>> SPLICE_JSON_SCHEMA_BEGIN
 173 | json_response: %json {"type": "object", "required": ["ok", "a"], "properties": {"ok": {"type": "string"}, "a": {"type": "string", "minLength": 5, "maxLength": 2}}}
                      ^
 174 | // <<< SPLICE_JSON_SCHEMA_END
 175 | 
'''

# --- probe §1, "config": our own x-guidance object -----------------------------------------------------
CONFIG_MESSAGE = '''at 173(16): failed to compile JSON schema: unknown field `bogus`, expected one of `item_separator`, `key_separator`, `whitespace_flexible`, `whitespace_pattern`, `coerce_one_of`, `lenient`, `json_allowed_escapes`, `json_allow_general_unicode_escapes`
 171 | 
 173 | json_response: %json {"type": "string", "x-guidance": {"bogus": 1}}
                      ^
'''

# --- probe §E2: envelope C, a string tool parameter compiled as a Lark regex ---------------------------
TOOL_PARAMETER_MESSAGE = '''at 189(22): invalid regex "(?:(?=a))" (in regex): regex parse error:
    (?:(?=a))
       ^^^
error: look-around, including look-ahead and look-behind, is not supported
'''


def compile_failures(*cases):
    """A fake compiler: the first `(needle, message)` whose needle is in the payload decides."""
    def compile_text(text):
        for needle, message in cases:
            if needle in text:
                return message
        return None
    return compile_text


class Classification(unittest.TestCase):
    """Design §2's table, one verbatim probe message per class."""

    def test_every_class_the_design_names_exists(self):
        self.assertEqual(set(ladder.CLASSES),
                         {"ok", "unsupported", "resource", "reference", "invalid", "unsatisfiable", "parse",
                          "config", "unknown"})

    def test_a_clean_compile_is_ok(self):
        for message in (None, "", " \n "):
            finding = ladder.classify(message)
            self.assertTrue(finding.ok, message)
            self.assertEqual(finding.cls, "ok")

    def test_unknown_format_is_unsupported_and_names_the_format(self):
        finding = ladder.classify(FORMAT_MESSAGE)
        self.assertFalse(finding.ok)
        self.assertEqual(finding.cls, "unsupported")
        self.assertEqual(finding.core, "Unknown format: nope")
        self.assertEqual(finding.named, ("nope",))
        self.assertEqual(finding.channel, "none")
        self.assertIsNone(finding.pointer)

    def test_unimplemented_keys_is_unsupported_and_lists_the_keys(self):
        finding = ladder.classify(KEYS_MESSAGE)
        self.assertEqual(finding.cls, "unsupported")
        self.assertEqual(finding.core, 'Unimplemented keys: ["not"]')
        self.assertEqual(finding.named, ("not",))

    def test_a_regex_error_is_unsupported_and_carries_the_echo(self):
        finding = ladder.classify(LOOKAHEAD_MESSAGE)
        self.assertEqual(finding.cls, "unsupported")
        self.assertEqual(finding.channel, "regex")
        self.assertEqual(finding.pattern, "(?=a)")
        self.assertEqual(finding.caret, 0)
        self.assertEqual(finding.keyword, "pattern")
        self.assertEqual(finding.core,
                         "regex parse error:\n    (?=a)\n    ^^^\n"
                         "error: look-around, including look-ahead and look-behind, is not supported")

    def test_envelope_c_classifies_the_same_and_echoes_the_generated_regex(self):
        finding = ladder.classify(TOOL_PARAMETER_MESSAGE)
        self.assertEqual(finding.cls, "unsupported")
        self.assertEqual(finding.channel, "regex")
        self.assertEqual(finding.pattern, "(?:(?=a))")
        self.assertEqual(finding.caret, 3)
        self.assertIn("error: look-around", finding.core)

    def test_a_missing_pointer_is_reference_and_names_the_pointer(self):
        finding = ladder.classify(REFERENCE_MESSAGE)
        self.assertEqual(finding.cls, "reference")
        self.assertEqual(finding.channel, "named")
        self.assertEqual(finding.named, ("/$defs/missing",))
        self.assertEqual(finding.core, "Pointer '/$defs/missing' does not exist")

    def test_prefix_items_is_invalid_and_names_the_keyword(self):
        finding = ladder.classify(INVALID_MESSAGE)
        self.assertEqual(finding.cls, "invalid")
        self.assertEqual(finding.keyword, "prefixItems")
        self.assertEqual(finding.core, "prefixItems must be an array")

    def test_a_size_limit_is_resource(self):
        finding = ladder.classify(RESOURCE_MESSAGE)
        self.assertEqual(finding.cls, "resource")
        self.assertEqual(finding.core, "schema too large")

    def test_an_unsatisfiable_required_property_names_the_property(self):
        finding = ladder.classify(UNSATISFIABLE_MESSAGE)
        self.assertEqual(finding.cls, "unsatisfiable")
        self.assertEqual(finding.channel, "named")
        self.assertEqual(finding.named, ("a",))
        self.assertEqual(finding.core,
                         "Unsatisfiable schema: required property 'a' is unsatisfiable")

    def test_an_unsatisfiable_message_that_names_nothing_has_no_property(self):
        message = ("at 173(16): failed to compile JSON schema: Unsatisfiable schema: "
                   "minLength (5) is greater than maxLength (2)\n 173 | x\n")
        finding = ladder.classify(message)
        self.assertEqual(finding.cls, "unsatisfiable")
        self.assertEqual(finding.named, ())
        self.assertIsNone(finding.pointer)

    def test_envelope_b_is_parse_and_keeps_the_column_in_the_core(self):
        finding = ladder.classify(PARSE_DEPTH_MESSAGE)
        self.assertEqual(finding.cls, "parse")
        self.assertEqual(finding.channel, "offset")
        self.assertEqual(finding.core, "recursion limit exceeded at line 1 column 2492")

    def test_a_bad_x_guidance_field_is_config(self):
        # `unknown field `bogus`, expected one of ...` also contains the word "expected", which the invalid
        # row matches; the config row has to win or the class is wrong (see the delivery report).
        finding = ladder.classify(CONFIG_MESSAGE)
        self.assertEqual(finding.cls, "config")
        self.assertTrue(finding.core.startswith("unknown field `bogus`"))

    def test_a_panic_is_unknown_and_terminal(self):
        finding = ladder.classify("panic: called Option::unwrap() on a None value")
        self.assertEqual(finding.cls, "unknown")
        self.assertEqual(finding.channel, "none")

    def test_an_unfetchable_resource_is_reference_and_not_unknown(self):
        # taxonomy §1.3: `$ref` to an external document, and an unknown `$schema`, both report this.  The design's
        # table has no row for it; without one it would land in `unknown`.
        message = ('at 173(16): failed to compile JSON schema: Resource \'https://example.com/x\' is not present '
                   'in a registry and retrieving it failed: Default retriever does not fetch resources\n 173 | x\n')
        finding = ladder.classify(message)
        self.assertEqual(finding.cls, "reference")
        self.assertEqual(finding.named, ("https://example.com/x",))

    def test_an_unfetchable_ref_is_replaced_with_an_empty_object(self):
        message = ('at 173(16): failed to compile JSON schema: Resource \'https://example.com/x\' is not present '
                   'in a registry and retrieving it failed: Default retriever does not fetch resources\n 173 | x\n')
        schema = {"type": "object", "properties": {"a": {"$ref": "https://example.com/x"}}}
        finding = ladder.locate(ladder.classify(message), ladder.payload(schema))
        self.assertEqual(finding.pointer, "/properties/a")
        fresh, _ = ladder.repair(schema, finding)
        self.assertEqual(fresh["properties"]["a"], {})

    def test_an_unsupported_construct_the_design_does_not_prefix_is_still_unsupported(self):
        message = ('at 173(16): failed to compile JSON schema: oneOf constraints are not supported. Enable '
                   "'coerce_one_of' option to approximate oneOf with anyOf\n 173 | x\n")
        self.assertEqual(ladder.classify(message).cls, "unsupported")

    def test_nothing_raises_on_a_message_with_no_envelope(self):
        for message in ("Unknown format: nope", "garbage", "\x00"):
            self.assertIn(ladder.classify(message).cls, ladder.CLASSES)

    def test_a_whole_schema_terminal_is_a_400_and_a_config_failure_is_a_500(self):
        self.assertEqual(ladder.status(ladder.classify(INVALID_MESSAGE)), 400)
        self.assertEqual(ladder.status(ladder.classify(CONFIG_MESSAGE)), 500)


class Offsets(unittest.TestCase):
    """Design §3.1: `column N` is a 1-based offset into `payload(schema)`."""

    def test_a_depth_column_points_at_the_key_of_the_failing_level(self):
        schema = {"type": "string"}
        for _ in range(3):
            schema = {"type": "object", "properties": {"p": schema}}
        text = ladder.payload(schema)
        # Hand count on the literal payload: the third `"p"` key starts at index 112, so its 1-based column is
        # 113.  The parse limit fires at the key of the level one deeper than the limit allows, and the pointer
        # is that key's value (probe §B2: `/properties/p` x 64 for depth 64).
        self.assertEqual(
            text,
            '{"type": "object", "properties": {"p": {"type": "object", "properties": {"p": {"type": "object", '
            '"properties": {"p": {"type": "string"}}}}}}}')
        message = 'failed to parse "%json": recursion limit exceeded at line 1 column 113'
        finding = ladder.locate(ladder.classify(message), text)
        self.assertEqual(finding.cls, "parse")
        self.assertEqual(finding.pointer, "/properties/p/properties/p/properties/p")
        self.assertFalse(finding.ambiguous)

    def test_the_real_depth_64_column_points_64_levels_down(self):
        # probe §B2: depth 64 is a 2642-character payload and `column 2492`; the probe's pointer is
        # `/properties/p` x 64.  This is the measured number, not a hand-computed one.
        schema = {"type": "string"}
        for _ in range(64):
            schema = {"type": "object", "properties": {"p": schema}}
        text = ladder.payload(schema)
        self.assertEqual(len(text), 2642)
        finding = ladder.locate(ladder.classify(PARSE_DEPTH_MESSAGE), text)
        self.assertEqual(finding.pointer, "/properties/p" * 64)

    def test_an_infinity_payload_is_walked_without_json_loads(self):
        # §7: `maximum: 1e400` becomes Python `inf` and `json.dumps` writes a bare `Infinity`, which no strict
        # JSON parser accepts (Python's own `json.loads` does, unless told otherwise).  The design says this case
        # needs no pointer; the scanner produces one anyway, and the important assertion is that it does not
        # raise or give up.
        schema = {"type": "object", "properties": {"a": {"type": "number", "maximum": float("inf")}}}
        text = ladder.payload(schema)
        self.assertIn("Infinity", text)

        def reject(constant):
            raise ValueError(constant)

        with self.assertRaises(ValueError):
            json.loads(text, parse_constant=reject)
        finding = ladder.locate(ladder.classify(
            'failed to parse "%json": expected value at line 1 column 70'), text)
        self.assertEqual(finding.cls, "parse")
        self.assertEqual(finding.pointer, "/properties/a/maximum")

    def test_an_offset_past_the_end_is_no_pointer(self):
        finding = ladder.locate(ladder.classify(PARSE_DEPTH_MESSAGE), '{"a": 1}')
        self.assertIsNone(finding.pointer)

    def test_an_offset_between_members_names_the_enclosing_container(self):
        text = '{"a": {"b": 1}, "c": 2}'
        # index 14 is the comma between the two members: inside no key or value, inside the root object.
        self.assertEqual(text[14], ",")
        self.assertEqual(ladder.pointer_at(text, 14), "")
        self.assertEqual(ladder.pointer_at(text, 12), "/a/b")


class RegexEcho(unittest.TestCase):
    """Design §3.2 / probe §B1: the echoed pattern is decoded, the payload holds it escaped."""

    SCHEMA = {"type": "object", "properties": {"ok": {"type": "string"},
                                               "bad": {"type": "string", "pattern": "(a)\\1"}}}

    def test_the_decoded_pattern_is_absent_and_the_escaped_one_is_present(self):
        text = ladder.payload(self.SCHEMA)
        self.assertNotIn("(a)\\1", text)                     # what the compiler printed
        self.assertIn(json.dumps("(a)\\1")[1:-1], text)      # what the payload holds
        self.assertEqual(json.dumps("(a)\\1")[1:-1], "(a)\\\\1")

    def test_the_pointer_is_the_bad_pattern_not_its_sibling(self):
        text = ladder.payload(self.SCHEMA)
        finding = ladder.locate(ladder.classify(BACKREF_MESSAGE), text)
        self.assertEqual(finding.pattern, "(a)\\1")
        self.assertEqual(finding.caret, 3)
        self.assertEqual(finding.pointer, "/properties/bad/pattern")
        self.assertFalse(finding.ambiguous)
        self.assertEqual(finding.candidates, ("/properties/bad/pattern",))

    def test_a_deeper_pattern_pointer_is_the_whole_path(self):
        schema = {"type": "object", "properties": {"q": {"type": "array", "items": {
            "type": "object", "properties": {"x": {"type": "string", "pattern": "[a-z"}}}}}}
        message = '''at 173(16): failed to compile JSON schema: regex parse error:
    [a-z
    ^
error: unclosed character class
'''
        finding = ladder.locate(ladder.classify(message), ladder.payload(schema))
        self.assertEqual(finding.pointer, "/properties/q/items/properties/x/pattern")


class Ambiguity(unittest.TestCase):
    """Design §3.2: several hits are a real answer, not a failure to answer."""

    def test_the_same_bad_pattern_in_two_properties_is_ambiguous(self):
        schema = {"type": "object", "properties": {
            "a": {"type": "string", "pattern": "(?=a)"},
            "b": {"type": "string", "pattern": "(?=a)"}}}
        finding = ladder.locate(ladder.classify(LOOKAHEAD_MESSAGE), ladder.payload(schema))
        self.assertTrue(finding.ambiguous)
        self.assertIsNone(finding.pointer)
        self.assertEqual(finding.candidates, ("/properties/a/pattern", "/properties/b/pattern"))

    def test_the_pattern_text_earlier_in_a_description_is_not_a_hit(self):
        schema = {"description": "avoid (?=a)", "type": "object", "properties": {
            "a": {"type": "string", "pattern": "(?=a)"}}}
        text = ladder.payload(schema)
        self.assertLess(text.index("(?=a)"), text.index('"pattern"'))
        finding = ladder.locate(ladder.classify(LOOKAHEAD_MESSAGE), text)
        self.assertFalse(finding.ambiguous)
        self.assertEqual(finding.pointer, "/properties/a/pattern")

    def test_a_pattern_echoed_nowhere_has_no_pointer(self):
        # envelope C echoes the *generated* regex, which a schema-level payload does not contain.
        schema = {"type": "object", "properties": {"p": {"type": "string", "pattern": "(?=a)"}}}
        finding = ladder.locate(ladder.classify(TOOL_PARAMETER_MESSAGE), ladder.payload(schema))
        self.assertIsNone(finding.pointer)
        self.assertFalse(finding.ambiguous)


class Repairs(unittest.TestCase):
    """Design §4: exactly one relaxation, on a deep copy, root-only where it has to be."""

    def located(self, schema, message):
        return ladder.locate(ladder.classify(message), ladder.payload(schema))

    def test_rung_one_relaxes_the_root_for_its_two_classes_only(self):
        for message in (FORMAT_MESSAGE, KEYS_MESSAGE):
            schema = {"type": "object", "properties": {"a": {"format": "nope", "not": {}}}}
            before = copy.deepcopy(schema)
            repaired = ladder.repair(schema, self.located(schema, message))
            self.assertIsNotNone(repaired, message)
            fresh, note = repaired
            self.assertEqual(fresh["x-guidance"], {"lenient": True})
            self.assertNotIn("x-guidance", fresh["properties"]["a"])
            self.assertIn("root", note)
            self.assertEqual(schema, before)

    def test_rung_one_is_not_used_for_the_other_classes(self):
        cases = [
            (LOOKAHEAD_MESSAGE, {"type": "string", "pattern": "(?=a)"}),
            (INVALID_MESSAGE, {"type": "array", "prefixItems": {}}),
            (REFERENCE_MESSAGE, {"$ref": "#/$defs/missing"}),
            (UNSATISFIABLE_MESSAGE, {"type": "object", "required": ["a"],
                                     "properties": {"a": {"type": "string", "minLength": 5, "maxLength": 2}}}),
        ]
        for message, schema in cases:
            finding = self.located(schema, message)
            repaired = ladder.repair(schema, finding)
            self.assertIsNotNone(repaired, message)
            fresh, _ = repaired
            self.assertNotIn("x-guidance", fresh if isinstance(fresh, dict) else {}, message)

    def test_a_regex_repair_deletes_only_the_offending_pattern(self):
        schema = {"type": "object", "properties": {
            "ok": {"type": "string", "pattern": "abc"},
            "bad": {"type": "string", "pattern": "(?=a)", "description": "keep me"}}}
        before = copy.deepcopy(schema)
        fresh, note = ladder.repair(schema, self.located(schema, LOOKAHEAD_MESSAGE))
        self.assertNotIn("pattern", fresh["properties"]["bad"])
        self.assertEqual(fresh["properties"]["bad"]["description"], "keep me")
        self.assertEqual(fresh["properties"]["ok"]["pattern"], "abc")
        self.assertEqual(note, "deleted the pattern at /properties/bad/pattern")
        self.assertEqual(schema, before)

    def test_an_ambiguous_regex_repair_deletes_every_candidate_and_says_so(self):
        schema = {"type": "object", "properties": {
            "a": {"type": "string", "pattern": "(?=a)"},
            "b": {"type": "string", "pattern": "(?=a)"}}}
        finding = self.located(schema, LOOKAHEAD_MESSAGE)
        fresh, note = ladder.repair(schema, finding)
        self.assertNotIn("pattern", fresh["properties"]["a"])
        self.assertNotIn("pattern", fresh["properties"]["b"])
        self.assertIn("more than one place", note)

    def test_a_missing_ref_becomes_an_empty_object(self):
        schema = {"type": "object", "properties": {
            "a": {"$ref": "#/$defs/missing"},
            "b": {"type": "string"}}}
        before = copy.deepcopy(schema)
        fresh, note = ladder.repair(schema, self.located(schema, REFERENCE_MESSAGE))
        self.assertEqual(fresh["properties"]["a"], {})
        self.assertEqual(fresh["properties"]["b"], {"type": "string"})
        self.assertEqual(fresh["type"], "object")
        self.assertIn("/properties/a", note)
        self.assertEqual(schema, before)

    def test_an_unsatisfiable_property_is_relaxed_to_an_empty_schema(self):
        schema = {"type": "object", "required": ["ok", "a"], "properties": {
            "ok": {"type": "string"},
            "a": {"type": "string", "minLength": 5, "maxLength": 2}}}
        before = copy.deepcopy(schema)
        fresh, note = ladder.repair(schema, self.located(schema, UNSATISFIABLE_MESSAGE))
        self.assertEqual(fresh["properties"]["a"], {})
        self.assertEqual(fresh["properties"]["ok"], {"type": "string"})
        self.assertEqual(fresh["required"], ["ok", "a"])
        self.assertIn("/properties/a", note)
        self.assertEqual(schema, before)

    def test_the_named_keyword_is_deleted_at_every_node(self):
        schema = {"type": "array", "prefixItems": [{"type": "string"}], "items": {
            "type": "array", "prefixItems": {}}}
        before = copy.deepcopy(schema)
        fresh, note = ladder.repair(schema, self.located(schema, INVALID_MESSAGE))
        self.assertNotIn("prefixItems", fresh)
        self.assertNotIn("prefixItems", fresh["items"])
        self.assertEqual(fresh["type"], "array")
        self.assertIn("2 node(s)", note)
        self.assertEqual(schema, before)

    def test_an_invalid_message_that_names_no_keyword_has_no_repair(self):
        message = ('at 173(16): failed to compile JSON schema: schema must be an object or boolean\n 173 | x\n')
        schema = [1, 2]
        self.assertIsNone(ladder.repair(schema, self.located(schema, message)))

    def test_a_terminal_class_has_no_repair(self):
        schema = {"anyOf": [{"const": 0}]}
        finding = self.located(schema, RESOURCE_MESSAGE)
        self.assertEqual(finding.cls, "resource")
        self.assertIsNone(finding.pointer)
        self.assertIsNone(ladder.repair(schema, finding))

    def test_rung_one_does_nothing_when_lenient_is_already_true(self):
        schema = {"type": "string", "format": "nope", "x-guidance": {"lenient": True}}
        self.assertIsNone(ladder.repair(schema, self.located(schema, FORMAT_MESSAGE)))

    def test_the_nesting_limit_is_capped_at_the_schema_depth_it_allows(self):
        # probe §9 plus a measurement on this build: a 63-level `properties` chain compiles, 64 fails, and the
        # message's column does **not** move when the deepest node shrinks - so a single-node collapse repairs
        # nothing and the ladder stops.  The repair is a depth cap: schema nodes below `ladder.DEPTH_LIMIT`
        # become `true`, which is a legal schema that accepts anything and is not a container.
        schema = {"type": "string"}
        for _ in range(ladder.DEPTH_LIMIT // 2 + 4):
            schema = {"type": "object", "properties": {"p": schema}}
        before = copy.deepcopy(schema)

        def compile_text(text):
            # serde_json counts containers; each `properties` level is two.  The fake fires while a chain is
            # deeper than the cap allows and is clean once the cap has been applied.
            return PARSE_DEPTH_MESSAGE if text.count('{"type": "object"') > ladder.DEPTH_LIMIT // 2 else None

        outcome = ladder.run(schema, compile_text)
        self.assertIsNone(outcome.terminal, "the cap should make it compile")
        self.assertEqual([(s.rung, s.cls) for s in outcome.steps], [(2, "parse")])
        self.assertIn("below %d levels" % ladder.DEPTH_LIMIT, outcome.steps[0].note)
        # The outermost constraint survives, the tail is `true`, and the input was not touched.
        self.assertEqual(outcome.schema["type"], "object")
        node, levels = outcome.schema, 0
        while isinstance(node, dict) and "p" in node.get("properties", {}):
            node, levels = node["properties"]["p"], levels + 1
        self.assertIs(node, True)
        # 50 `properties` hops survive, which is 100 containers - exactly the cap, so the 51st is what became
        # `true`.  Asserted as the measured count rather than a formula, because the exact off-by-one is the walk's
        # `level >= limit` test, and that is the thing worth pinning.
        self.assertEqual(levels, 50)
        self.assertEqual(schema, before)

    def test_a_nesting_failure_that_cannot_be_capped_is_terminal(self):
        # Already shallow: nothing is below the limit, so there is no repair to make and the run must stop rather
        # than report success it did not achieve.
        schema = {"type": "object", "properties": {"p": {"type": "string"}}}

        def compile_text(text):
            return PARSE_DEPTH_MESSAGE

        outcome = ladder.run(schema, compile_text)
        self.assertIsNotNone(outcome.terminal)
        self.assertIsNone(outcome.schema)
        self.assertEqual(outcome.steps, ())


class Bounds(unittest.TestCase):
    """Design §5 / D-055: a repair that removes nothing must be terminal, not a loop."""

    def test_a_failure_that_never_changes_terminates_with_a_terminal_finding(self):
        calls = []

        def stuck(text):
            calls.append(text)
            return FORMAT_MESSAGE

        schema = {"type": "string", "format": "nope"}
        outcome = ladder.run(schema, stuck)
        self.assertFalse(outcome.ok)
        self.assertIsNone(outcome.schema)
        self.assertIsNotNone(outcome.terminal)
        self.assertEqual(outcome.terminal.cls, "unsupported")
        self.assertLessEqual(len(calls), ladder.MAX_RUNGS + 1)
        self.assertLessEqual(len(outcome.steps), ladder.MAX_RUNGS)

    def test_a_keyword_that_comes_back_is_terminal(self):
        def always_prefix_items(text):
            return INVALID_MESSAGE

        outcome = ladder.run({"type": "array", "prefixItems": {}}, always_prefix_items)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.terminal.cls, "invalid")
        self.assertEqual(len(outcome.steps), 1)
        self.assertIn("deleted 'prefixItems'", outcome.steps[0].note)

    def test_a_message_with_no_rung_is_terminal_immediately(self):
        calls = []

        def panic(text):
            calls.append(text)
            return "panic: called Option::unwrap() on a None value"

        outcome = ladder.run({"type": "string"}, panic)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.terminal.cls, "unknown")
        self.assertEqual(outcome.steps, ())
        self.assertEqual(len(calls), 1)

    def test_the_rung_ceiling_is_honoured(self):
        def stuck(text):
            return FORMAT_MESSAGE

        outcome = ladder.run({"type": "string", "format": "nope"}, stuck, limit=1)
        self.assertLessEqual(len(outcome.steps), 1)

    def test_a_clean_compile_returns_the_schema_and_no_steps(self):
        schema = {"type": "object"}
        outcome = ladder.run(schema, lambda text: None)
        self.assertTrue(outcome.ok)
        self.assertIs(outcome.schema, schema)
        self.assertEqual(outcome.steps, ())

    def test_the_compiler_is_handed_the_payload_not_the_object(self):
        seen = []

        def record(text):
            seen.append(text)
            return None

        schema = {"type": "string", "format": "nope"}
        ladder.run(schema, record)
        self.assertEqual(seen, [json.dumps(schema)])

    def test_the_ladder_records_the_rung_it_used(self):
        def lenient_only(text):
            schema = json.loads(text)
            if (schema.get("x-guidance") or {}).get("lenient"):
                return None
            return FORMAT_MESSAGE

        outcome = ladder.run({"type": "string", "format": "nope"}, lenient_only)
        self.assertTrue(outcome.ok)
        self.assertEqual([step.rung for step in outcome.steps], [1])
        self.assertEqual(outcome.steps[0].cls, "unsupported")
        self.assertEqual(outcome.schema["x-guidance"], {"lenient": True})

    def test_two_faults_take_two_rungs_in_order(self):
        def two_faults(text):
            schema = json.loads(text)
            if not (schema.get("x-guidance") or {}).get("lenient"):
                return FORMAT_MESSAGE
            if "(?=a)" in text:
                return LOOKAHEAD_MESSAGE
            return None

        schema = {"type": "object", "format": "nope", "properties": {
            "a": {"type": "string", "pattern": "(?=a)"}}}
        outcome = ladder.run(schema, two_faults)
        self.assertTrue(outcome.ok)
        self.assertEqual([step.rung for step in outcome.steps], [1, 2])
        self.assertEqual(outcome.steps[1].pointer, "/properties/a/pattern")
        self.assertNotIn("pattern", outcome.schema["properties"]["a"])

    def test_the_invalid_rung_is_rung_three(self):
        def compiles_without_prefix_items(text):
            return None if "prefixItems" not in text else INVALID_MESSAGE

        outcome = ladder.run({"type": "array", "prefixItems": {}}, compiles_without_prefix_items)
        self.assertTrue(outcome.ok)
        self.assertEqual([step.rung for step in outcome.steps], [3])


class Truncation(unittest.TestCase):
    """Design §5: a 40 000-branch schema's message carries the whole payload, so it is cut."""

    def big_message(self):
        schema = {"anyOf": [{"const": i} for i in range(40000)]}
        text = ladder.payload(schema)
        self.assertGreater(len(text), 600000)
        return ("at 173(16): failed to compile JSON schema: schema too large\n"
                " 171 | \n"
                " 172 | // >>> SPLICE_JSON_SCHEMA_BEGIN\n"
                " 173 | json_response: %json " + text + "\n"
                "                      ^\n"
                " 174 | // <<< SPLICE_JSON_SCHEMA_END\n"
                " 175 | \n")

    def test_the_core_is_the_inner_message_and_the_raw_is_cut(self):
        message = self.big_message()
        finding = ladder.classify(message)
        self.assertEqual(finding.cls, "resource")
        self.assertEqual(finding.core, "schema too large")
        suffix = "... (%d bytes)" % len(message.encode("utf-8"))
        self.assertTrue(finding.raw.endswith(suffix), finding.raw[-40:])
        self.assertEqual(finding.raw, message[:ladder.MESSAGE_LIMIT] + suffix)
        self.assertEqual(len(finding.raw), ladder.MESSAGE_LIMIT + len(suffix))

    def test_a_terminal_finding_is_stored_cut(self):
        message = self.big_message()
        outcome = ladder.run({"anyOf": [{"const": 0}]}, lambda text: message)
        self.assertFalse(outcome.ok)
        self.assertLessEqual(len(outcome.terminal.raw), ladder.MESSAGE_LIMIT + 40)


if __name__ == "__main__":
    unittest.main()
