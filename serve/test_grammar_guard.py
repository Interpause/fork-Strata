"""The startup guard: a doctored id table, a doctored template and a real pack.

    python -m unittest serve.test_grammar_guard -v

The template half runs anywhere.  The pack half is skipped with a named reason when no pack is present, so this
module never reports green for a check it did not make.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.grammar import guard

ROOT = Path(__file__).resolve().parents[1]


TRUE_IDS = dict(guard.DELIMITER_IDS)


def fake_encoder(split: str | None = None, raises: str | None = None):
    """An encoder that gets every delimiter right, except for the one named.

    It reads TRUE_IDS, not the table, so doctoring the table is what the test changes.
    """
    def encode(text, parse_special=False, plain=()):
        if raises and text == raises:
            raise KeyError(text)
        if split and text == split:
            return [TRUE_IDS[text] // 100, TRUE_IDS[text] % 100]
        return [TRUE_IDS[text]]
    return encode


class Ids(unittest.TestCase):
    def test_the_table_is_right(self):
        self.assertEqual(guard.check_ids(fake_encoder()), None)      # names the encoder, not the grammar

    def test_a_doctored_table_names_the_delimiter(self):
        for text in ("<|im_end|>", "<think>", "</tool_call>"):
            with self.subTest(text=text):
                with mock.patch.dict(guard.DELIMITER_IDS, {text: 1}):
                    with self.assertRaises(guard.GuardError) as caught:
                        guard.check_ids(fake_encoder())
                self.assertIn(repr(text), str(caught.exception))
                self.assertIn("re-export the grammar", str(caught.exception))

    def test_a_delimiter_that_splits_into_pieces_is_refused(self):
        # As wrong as a different id: the grammar names the token, so pieces could never spell it.
        with self.assertRaises(guard.GuardError) as caught:
            guard.check_ids(fake_encoder(split="</think>"))
        self.assertIn("'</think>'", str(caught.exception))

    def test_an_encoder_that_refuses_a_delimiter_is_a_guard_error(self):
        with self.assertRaises(guard.GuardError) as caught:
            guard.check_ids(fake_encoder(raises="<tool_call>"))
        self.assertIn("<tool_call>", str(caught.exception))


class Template(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "serve" / "chat_template.jinja").read_text(encoding="utf-8")

    def doctored(self, old: str, new: str) -> ChatTemplate:
        self.assertIn(old, self.source, "this test doctors bytes the template no longer has")
        self.assertEqual(self.source.count(old), 1, old)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat_template.jinja"
            path.write_text(self.source.replace(old, new), encoding="utf-8")
            return ChatTemplate(path)

    def test_the_repository_template_still_matches_the_grammar(self):
        guard.check_template(ChatTemplate(ROOT / "serve" / "chat_template.jinja"))

    def test_a_changed_thinking_off_tail_is_refused(self):
        template = self.doctored(r"'<think>\n\n</think>\n\n'", r"'<think>\n\n</think>\n'")
        with self.assertRaises(guard.GuardError) as caught:
            guard.check_template(template)
        message = str(caught.exception)
        self.assertIn("<think>", message)
        self.assertIn("re-export", message)

    def test_a_changed_separator_before_a_call_is_refused(self):
        template = self.doctored(r"'\n\n<tool_call>\n<function='", r"'\n<tool_call>\n<function='")
        with self.assertRaises(guard.GuardError) as caught:
            guard.check_template(template)
        self.assertIn("tool call after", str(caught.exception))

    def test_changed_parameter_framing_is_refused(self):
        template = self.doctored("'<parameter='", "'<argument='")
        with self.assertRaises(guard.GuardError) as caught:
            guard.check_template(template)
        self.assertIn("<parameter=", str(caught.exception))


class Pack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from serve.test_grammar_matcher import pack_dir
        directory = pack_dir()
        if directory is None:
            raise unittest.SkipTest("no pack tokenizer (set STRATA_PACK_TOKENIZER_DIR)")
        import sys
        tools = str(ROOT / "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        import strata_tokenizer as ST
        vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for text, i in vocab.items():
            tokens[i] = text
        merges = (directory / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((directory / "token_type.json").read_text())
        cls.tok = ST.Tokenizer(tokens, merges, types)

    def test_the_pack_encoder_matches_the_table(self):
        # The same encoder the server passes, from the same files server.main() reads.
        guard.check_ids(self.tok.encode)


class Wiring(unittest.TestCase):
    def test_server_main_reaches_both_checks_before_it_serves(self):
        # Nothing cheaper proves a startup check is *reached*: a guard that is written but never called is exactly
        # the silent failure D-013 exists for.
        import inspect
        import serve.server
        source = inspect.getsource(serve.server.main)
        self.assertIn("guard.check_ids(", source)
        self.assertIn("guard.check_template(", source)


if __name__ == "__main__":
    unittest.main()
