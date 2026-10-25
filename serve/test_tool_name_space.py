"""serve/test_tool_name_space.py - a tool call written `<function= NAME>` (a space after the `=`) is the call NAME.

Seen live (Qwen3.8-Flash-Next IQ2_XS, Zed's agent panel): the model wrote `<function= capture>` inside its thinking;
the name was read as " capture", matched no declared tool, stayed reasoning text (#804), and the turn ended with no
call, so the agent stopped.  After the thinking the same call came out named " capture", and its arguments were
typed without the tool's schema.

The reply path no longer parses calls at all (P6): `StructureParser` decides structure from the grammar's tokens,
so a reply's `<function= ...>` is never read as text here.  `parse_tool_call` is still the reader the chat-template
capability probe uses, and this module now pins that function.

    python -m unittest serve.test_tool_name_space -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import parse_tool_call  # noqa: E402

# "42" for a string parameter stays a string only when the call finds its tool's schema
SCHEMA = {"name": "capture", "parameters": {"properties": {"content": {"type": "string"}}}}
BODY = "<function= capture>\n<parameter=content>\n42\n</parameter>\n</function>"


class SpacedName(unittest.TestCase):
    def test_the_space_is_not_part_of_the_name(self):
        self.assertEqual(parse_tool_call(BODY, SCHEMA).name, "capture")

    def test_the_schema_still_types_the_value(self):
        """`42` is the string `"42"` for a string parameter; without the schema it is the number."""
        call = parse_tool_call(BODY, SCHEMA)
        self.assertEqual((call.name, call.arguments), ("capture", {"content": "42"}))
        self.assertEqual(parse_tool_call(BODY).arguments, {"content": 42})


if __name__ == "__main__":
    unittest.main()
