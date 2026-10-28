"""The grammar against the real library and a real pack vocabulary: what the masks allow.

    python -m unittest serve.test_grammar_matcher -v

Skipped, loudly, when either half is absent: `tools/build_llguidance.sh` for the library, and a pack's `tokenizer/`
for the vocabulary (named by `STRATA_PACK_TOKENIZER_DIR`, or found next to this checkout).  A skip here is not a
pass: `test_grammar.py` is the part that always runs.

Every check below is about a mask, because a mask is the only thing that constrains a sample.  A grammar that merely
"rejects the text later" constrains nothing.
"""
import glob
import os
import unittest
from pathlib import Path
from unittest import mock

from serve.grammar import build as bg
from serve.grammar import entries as be
from serve.grammar import matcher as M

WEATHER = {"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "required": ["city"], "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    }}}}
LISTING = {"type": "function", "function": {"name": "list_files", "parameters": {"type": "object"}}}
SCHEMA = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "string"}},
          "additionalProperties": False}

ROOT = Path(__file__).resolve().parents[1]


def pack_dir() -> Path | None:
    named = os.environ.get("STRATA_PACK_TOKENIZER_DIR", "").strip()
    if named:
        p = Path(named).expanduser()
        return p if (p / "vocab.json").exists() else None
    for pattern in (str(ROOT.parent / "Strata-data" / "packs" / "*" / "tokenizer"),
                    str(ROOT / "packs" / "*" / "tokenizer")):
        for candidate in sorted(glob.glob(pattern)):
            if (Path(candidate) / "vocab.json").exists():
                return Path(candidate)
    return None


def vocabulary() -> M.Vocabulary:
    try:
        path = M.library_path()
    except M.LibraryMissing as e:
        raise unittest.SkipTest(str(e))
    directory = pack_dir()
    if directory is None:
        raise unittest.SkipTest("no pack tokenizer (set STRATA_PACK_TOKENIZER_DIR)")
    try:
        return M.vocabulary(directory)
    except (ImportError, OSError, RuntimeError) as e:
        raise unittest.SkipTest(f"cannot build the vocabulary from {directory}: {e}")


class Discovery(unittest.TestCase):
    """Nameless: these run without the library, because a missing library must be a clear error, not a crash."""

    def test_the_environment_names_the_library(self):
        with mock.patch.dict(os.environ, {M.LIB_ENV: __file__}):
            self.assertEqual(M.library_path(), Path(__file__))

    def test_a_missing_library_is_an_error_that_names_the_build_script(self):
        with mock.patch.dict(os.environ, {M.LIB_ENV: "tmp/not-built/libllguidance.so"}):
            with self.assertRaises(M.LibraryMissing) as caught:
                M.library_path()
        self.assertIn("build_llguidance.sh", str(caught.exception))


class Masks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = vocabulary()

    def open(self, entry: str, **kw) -> M.Matcher:
        m = self.v.new_matcher(bg.for_entry(entry, **kw))
        self.assertIsNone(m.error(), entry)
        return m

    def test_every_static_entry_compiles_and_takes_a_first_mask(self):
        cases = [
            ("entry_think_plain", {}), ("entry_think_tools", {"tools": [WEATHER, LISTING]}),
            ("entry_think_tools_required", {"tools": [WEATHER]}), ("entry_think_json", {"schema": SCHEMA}),
            ("entry_answer_plain", {}), ("entry_answer_tools", {"tools": [WEATHER, LISTING]}),
            ("entry_answer_tools_required", {"tools": [WEATHER]}), ("entry_answer_json", {"schema": SCHEMA}),
            ("entry_call_name", {"tools": [WEATHER, LISTING]}),
        ]
        for entry, kw in cases:
            with self.subTest(entry=entry):
                m = self.open(entry, **kw)
                allowed, rc = m.mask()
                self.assertEqual(rc, 0, entry)
                self.assertGreater(len(allowed), 0, entry)
                m.close()

    def test_the_named_call_entry_compiles(self):
        m = self.open("entry_call_named_t0_get_weather", tools=[WEATHER])
        self.assertGreater(len(m.mask()[0]), 0)
        m.close()

    def test_thinking_cannot_end_the_turn_or_open_a_call(self):
        m = self.open("entry_think_plain")
        self.assertNotIn(self.v.eos, m.mask()[0])            # D3: the turn cannot end inside thinking
        text = self.v.ids("I should look at the files.")
        self.assertTrue(m.consume(text))
        self.assertNotIn(self.v.eos, m.mask()[0])
        m.close()
        m = self.open("entry_think_tools", tools=[WEATHER])
        self.assertNotIn(self.v.special["<tool_call>"], m.mask()[0])   # D3: and it cannot open a call
        m.close()

    def test_an_answer_without_tools_cannot_open_a_call(self):
        m = self.open("entry_answer_plain")
        allowed = m.mask()[0]
        self.assertIn(self.v.eos, allowed)                    # an empty answer is legal
        self.assertNotIn(self.v.special["<tool_call>"], allowed)
        m.close()

    def test_an_answer_with_tools_can_open_a_call_at_byte_zero(self):
        m = self.open("entry_answer_tools", tools=[WEATHER])
        self.assertIn(self.v.special["<tool_call>"], m.mask()[0])
        m.close()

    def test_a_required_call_forbids_the_prose_only_reply(self):
        m = self.open("entry_answer_tools_required", tools=[WEATHER])
        self.assertNotIn(self.v.eos, m.mask()[0])            # tool_choice: required, at the first token
        m.close()

    def test_json_mode_must_write_an_object_before_it_can_stop(self):
        m = self.open("entry_answer_json", schema=SCHEMA)
        allowed = m.mask()[0]
        self.assertNotIn(self.v.eos, allowed)
        self.assertIn(self.v.ids("{")[0], allowed)
        m.close()

    def test_a_wrong_type_is_refused_at_the_offending_token(self):
        m = self.open("entry_answer_json", schema=SCHEMA)
        self.assertTrue(m.consume(self.v.ids('{"answer": ')))
        # a number cannot begin a string property, so the next token is not in the mask
        digits = self.v.ids("1")
        self.assertFalse(m.allowed(digits[0]))
        m.close()

    def test_a_missing_required_property_cannot_close_the_object(self):
        m = self.open("entry_answer_json", schema=SCHEMA)
        self.assertTrue(m.consume(self.v.ids('{"answer"')))
        self.assertFalse(m.allowed(self.v.ids("}")[0]))
        m.close()

    def test_a_complete_answer_stops_at_the_end_token_only(self):
        m = self.open("entry_answer_plain")
        text = "Two plus two is four."
        self.assertTrue(m.consume(self.v.ids(text)))
        self.assertTrue(m.consume([self.v.eos]))
        self.assertTrue(m.is_stopped())
        allowed, rc = m.mask()
        self.assertEqual(rc, 0)
        self.assertEqual(allowed, [self.v.eos])
        m.close()

    def test_the_structural_delimiters_are_unreachable_as_text(self):
        # D4: the words spelled as ordinary pieces are prose; the delimiter tokens are structure.
        m = self.open("entry_answer_plain")
        for name in ("<think>", "</think>", "<tool_call>", "</tool_call>", "<|im_start|>", "<|endoftext|>"):
            with self.subTest(name=name):
                self.assertNotIn(self.v.special[name], m.mask()[0])
        m.close()

    def test_no_non_eos_special_id_is_offered_mid_answer_text(self):
        """D-042: the decoder may read an id as structure *because* prose cannot emit one.  Measured at the first
        mask above; this is the same claim mid-string, on both halves of a delimiter, where the exploration measured
        the mask widening again after `</` (`tmp/grammar-explore/STATE.md` §5.3)."""
        special = {name: self.v.special[name] for name in
                   ("<think>", "</think>", "<tool_call>", "</tool_call>", "<|im_start|>", "<|endoftext|>")}
        for entry, kw in (("entry_answer_tools", {"tools": [WEATHER]}), ("entry_answer_plain", {})):
            for prefix in ("hello", "the end</", "<", "</", "call <tool", "line\n"):
                with self.subTest(entry=entry, prefix=prefix):
                    m = self.open(entry, **kw)
                    self.assertTrue(m.consume(self.v.ids(prefix)), prefix)
                    allowed = m.mask()[0]
                    for name, tid in special.items():
                        self.assertNotIn(tid, allowed, f"{name} after {prefix!r}")
                    self.assertIn(self.v.eos, allowed)      # the answer can still end: the probe is not vacuous
                    m.close()

    def test_end_of_text_is_unreachable_so_a_turn_ends_only_at_the_end_token(self):
        m = self.open("entry_answer_plain")
        self.assertNotIn(self.v.special["<|endoftext|>"], m.mask()[0])
        self.assertTrue(m.consume(self.v.ids("stuck")))
        self.assertNotIn(self.v.special["<|endoftext|>"], m.mask()[0])
        m.close()


class Tools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = vocabulary()

    def open(self, entry: str, **kw) -> M.Matcher:
        m = self.v.new_matcher(bg.for_entry(entry, **kw))
        self.assertIsNone(m.error(), entry)
        return m

    def test_an_undeclared_name_is_impossible(self):
        # D6: the name position offers the declared names only, so an undeclared one dies at its first token.
        m = self.open("entry_call_name", tools=[WEATHER, LISTING])
        self.assertNotIn(self.v.ids("d")[0], m.mask()[0])
        self.assertFalse(m.consume(self.v.ids("delete_file>\n")))
        self.assertTrue(m.is_error())
        self.assertIsNotNone(m.error())
        m.close()

    def test_a_declared_name_is_accepted_in_order(self):
        # entry_call_name is the state after the prompt's "<function", so "=" comes first.
        m = self.open("entry_call_name", tools=[WEATHER, LISTING])
        self.assertTrue(m.consume(self.v.ids("=")))
        self.assertTrue(m.consume(self.v.ids("get")))
        self.assertNotIn(self.v.ids("l")[0], m.mask()[0], "only get_weather can still be spelled")
        self.assertTrue(m.consume(self.v.ids("_weather>\n")))
        # then the required parameter's own opener
        self.assertTrue(m.consume(self.v.ids("<parameter=city>\n")))
        m.close()

    def test_a_required_parameter_cannot_be_skipped(self):
        m = self.open("entry_call_named_t0_get_weather", tools=[WEATHER])
        self.assertFalse(m.consume(self.v.ids("</function>")), "the required city parameter is missing")
        m.close()

    def test_a_parameter_value_may_contain_a_delimiter_that_is_not_alone_on_its_line(self):
        # D5: only a standalone "\n</parameter>\n" line closes a value.
        m = self.open("entry_call_named_t0_get_weather", tools=[WEATHER])
        self.assertTrue(m.consume(self.v.ids("<parameter=city>\n")))
        self.assertTrue(m.consume(self.v.ids("a </parameter> inside one line\nand a second line")))
        self.assertFalse(m.allowed(self.v.eos))
        self.assertTrue(m.consume(self.v.ids("\n</parameter>\n")))
        m.close()

    def test_a_tool_without_parameters_closes_immediately(self):
        m = self.open("entry_call_named_t1_list_files", tools=[WEATHER, LISTING])
        self.assertTrue(m.consume(self.v.ids('</function>\n')))
        self.assertTrue(m.consume([self.v.special["</tool_call>"]]))
        m.close()

    def test_json_schema_with_tools_accepts_a_final_object_or_prose_and_a_call(self):
        m = self.open("entry_answer_json_tools", tools=[WEATHER], schema=SCHEMA)
        self.assertNotIn(self.v.eos, m.mask()[0])
        self.assertTrue(m.consume(self.v.ids('{"answer":"yes"}')))
        self.assertTrue(m.consume([self.v.eos]))
        self.assertTrue(m.is_accepting())
        m.close()

        bad = self.open("entry_answer_json_tools", tools=[WEATHER], schema=SCHEMA)
        self.assertFalse(bad.consume(self.v.ids('{"other":"no"}')))
        bad.close()

        call = self.v.ids('<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n'
                          '</function>\n</tool_call>')
        for prefix in ("", "I will check that now.\n\n"):
            with self.subTest(prefix=prefix):
                tool_turn = self.open("entry_answer_json_tools", tools=[WEATHER], schema=SCHEMA)
                self.assertTrue(tool_turn.consume(self.v.ids(prefix) + call), prefix)
                self.assertTrue(tool_turn.consume([self.v.eos]))
                self.assertTrue(tool_turn.is_accepting())
                tool_turn.close()

    def test_json_schema_with_tools_handles_thinking_and_required_choice(self):
        thinking = self.open("entry_think_json_tools", tools=[WEATHER], schema=SCHEMA)
        self.assertNotIn(self.v.eos, thinking.mask()[0])
        self.assertNotIn(self.v.special["<tool_call>"], thinking.mask()[0])
        self.assertTrue(thinking.consume(self.v.ids("I should check this first.\n")))
        self.assertTrue(thinking.consume([self.v.special["</think>"]]))
        self.assertTrue(thinking.consume(self.v.ids("\n\n{" + '"answer":"yes"}')))
        self.assertTrue(thinking.consume([self.v.eos]))
        self.assertTrue(thinking.is_accepting())
        thinking.close()

        required = self.open("entry_answer_tools_required", tools=[WEATHER])
        self.assertNotIn(self.v.eos, required.mask()[0])
        self.assertTrue(required.consume(self.v.ids('{"answer":"yes"}')))
        self.assertFalse(required.consume([self.v.eos]), "the required-call path cannot finish with schema prose")
        required.close()

        required_thinking = self.open("entry_think_tools_required", tools=[WEATHER])
        self.assertNotIn(self.v.eos, required_thinking.mask()[0])
        self.assertNotIn(self.v.special["<tool_call>"], required_thinking.mask()[0])
        self.assertTrue(required_thinking.consume(self.v.ids("Need to check.\n")))
        self.assertTrue(required_thinking.consume([self.v.special["</think>"]]))
        self.assertTrue(required_thinking.consume(self.v.ids("\n\n")))
        self.assertIn(self.v.special["<tool_call>"], required_thinking.mask()[0])
        required_thinking.close()


class Validation(unittest.TestCase):
    """The speculative-draft path: a whole window validated on a clone, the original left alone (D7)."""

    @classmethod
    def setUpClass(cls):
        cls.v = vocabulary()

    def test_validate_reports_the_accepted_prefix_and_changes_nothing(self):
        m = self.v.new_matcher(bg.for_entry("entry_answer_plain"))
        before, _ = m.mask()
        good = self.v.ids("Two plus two is four.")
        bad = good + [self.v.special["<think>"]] + good
        self.assertEqual(m.valid_prefix(bad), len(good))
        after, _ = m.mask()
        self.assertEqual(before, after)
        m.close()

    def test_a_clone_takes_the_same_state_independently(self):
        m = self.v.new_matcher(bg.for_entry("entry_answer_plain"))
        self.assertTrue(m.consume(self.v.ids("abc")))
        self.assertTrue(m.consume([self.v.eos]))
        self.assertTrue(m.is_stopped())
        c = m.clone()
        self.assertTrue(c.is_stopped())
        self.assertTrue(c.reset())
        self.assertFalse(c.is_stopped())     # the clone rewound...
        self.assertTrue(m.is_stopped())      # ...and the original did not move
        c.close()
        m.close()

    def test_rollback_counts_tokens(self):
        m = self.v.new_matcher(bg.for_entry("entry_answer_plain"))
        ids = self.v.ids("abcdef")
        self.assertTrue(m.consume(ids))
        self.assertTrue(m.rollback(len(ids)))
        self.assertTrue(m.rollback(0))
        self.assertFalse(m.rollback(len(ids) + 1), "a rollback past the start must fail")
        m.close()


if __name__ == "__main__":
    unittest.main()
