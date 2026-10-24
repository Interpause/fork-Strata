"""The decoder: structure from the grammar's own terminals, and nothing else.

    python -m unittest serve.test_grammar_decoder -v

No library and no pack: these feed the decoder token ids and bytes directly, because that is what it takes as input.
Two things are checked here that no other file can check:

* **The labels describe the grammar.**  Every structural literal the skeleton writes is one the decoder knows, and
  every literal the fragment wrote is in its labels.  An edit to `skeleton.lark` that the decoder does not follow
  therefore fails here instead of mislabelling a reply at runtime.
* **One code path, any split.**  The same reply, fed one byte per token and fed whole, produces the same steps.
  Streamed and whole cannot disagree, because there is nothing to disagree about: a delimiter split across two
  tokens is not a case the decoder can get wrong.

The decoys are the point of the design (D-042): a delimiter's *bytes*, spelled as ordinary pieces, are prose.  Only a
special-token id or a labelled literal is structure.
"""

import json
import re
import unittest

from serve.grammar import build as bg
from serve.grammar import decode as dec
from serve.grammar import guard
from serve.grammar.tools_fragment import Labels, compile_tools_fragment

WEATHER = {"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "required": ["city"], "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    }}}}
LISTING = {"type": "function", "function": {"name": "list_files", "parameters": {"type": "object"}}}

PAYLOAD_ID = 1_000_000          # any id that is not a delimiter's


class Special:
    """A delimiter token in a reply under test: structure, by id, with no payload bytes."""

    def __init__(self, name: str):
        self.name = name


def stream(parts, size: int):
    """The reply as tokens: specials by id, payload split into `size`-byte pieces."""
    out = []
    for part in parts:
        if isinstance(part, Special):
            out.append((guard.DELIMITER_IDS[part.name], ""))
        else:
            for i in range(0, len(part), size):
                out.append((PAYLOAD_ID + len(out), part[i:i + size]))
    return out


def run(entry: str, labels: Labels, parts, size: int = 1):
    d = dec.Decoder(entry, labels)
    steps = []
    for tid, text in stream(parts, size):
        steps += d.feed(tid, text)
    steps += d.finish()
    return normalise(steps)


def normalise(steps):
    """Adjacent text of the same kind and name is one step: how the payload was split is not the decoder's news."""
    out = []
    for step in steps:
        if out and out[-1].kind == step.kind and out[-1].name == step.name and step.text:
            out[-1] = dec.Step(step.kind, step.name, out[-1].text + step.text)
        else:
            out.append(step)
    return out


def labels_of(*tools) -> Labels:
    return compile_tools_fragment(list(tools)).labels


WEATHER_LABELS = labels_of(WEATHER)
BOTH_LABELS = labels_of(WEATHER, LISTING)


def reply(content="Checking the weather.", city="Paris", unit="celsius",
          reasoning="\nI should look at the city.\n", think=True, calls=1):
    """The bytes the chat template writes, in order: the reply this decoder is asked to read."""
    parts = []
    if think:
        parts += [reasoning, Special("</think>"), "\n\n"]
    parts += [content]
    if calls:
        parts += ["\n\n" if content else "", Special("<tool_call>"), "\n<function=", "get_weather>\n",
                  "<parameter=city>\n", city, "\n</parameter>\n",
                  "<parameter=unit>\n", unit, "\n</parameter>\n",
                  "</function>\n", Special("</tool_call>")]
        for _ in range(calls - 1):
            parts += ["\n", Special("<tool_call>"), "\n<function=", "get_weather>\n",
                      "</function>\n", Special("</tool_call>")]
    parts.append(Special("<|im_end|>"))
    return parts


CALL_STEPS = [
    dec.Step("call_open", "get_weather"),
    dec.Step("param", "city", "Paris"), dec.Step("param_end", "city"),
    dec.Step("param", "unit", "celsius"), dec.Step("param_end", "unit"),
    dec.Step("call_end", "get_weather"),
]


class Replies(unittest.TestCase):
    def test_a_thinking_reply_with_one_call(self):
        self.assertEqual(
            run("entry_think_tools", WEATHER_LABELS, reply()),
            [dec.Step("reasoning", text="\nI should look at the city.\n"),
             dec.Step("content", text="Checking the weather.")] + CALL_STEPS + [dec.Step("turn_end")])

    def test_the_split_of_the_tokens_does_not_change_the_steps(self):
        # The whole point of decoding tokens rather than text: every split point is the same code path.
        want = run("entry_think_tools", WEATHER_LABELS, reply(), size=64)
        for size in (1, 2, 3, 5, 7, 13):
            with self.subTest(size=size):
                self.assertEqual(run("entry_think_tools", WEATHER_LABELS, reply(), size=size), want)

    def test_a_call_at_byte_zero_has_no_content_and_no_separator(self):
        self.assertEqual(
            run("entry_answer_tools", WEATHER_LABELS, reply(content="", think=False)),
            CALL_STEPS + [dec.Step("turn_end")])

    def test_two_calls_are_two_call_ends(self):
        steps = run("entry_answer_tools", WEATHER_LABELS, reply(content="", think=False, calls=2))
        self.assertEqual([s.kind for s in steps].count("call_end"), 2)
        self.assertEqual(steps[-1], dec.Step("turn_end"))

    def test_an_answer_without_tools_is_one_content_step(self):
        self.assertEqual(
            run("entry_answer_plain", Labels(), ["just prose", Special("<|im_end|>")]),
            [dec.Step("content", text="just prose"), dec.Step("turn_end")])

    def test_a_json_answer_is_content(self):
        self.assertEqual(
            run("entry_answer_json", Labels(), ['{"answer": "x"}', Special("<|im_end|>")]),
            [dec.Step("content", text='{"answer": "x"}'), dec.Step("turn_end")])

    def test_the_mid_call_entries_start_where_the_prompt_left_off(self):
        # tool_choice "required": the prompt already ends at "<tool_call>\n<function" - so "=" and the name follow.
        self.assertEqual(
            run("entry_call_name", WEATHER_LABELS,
                ["=", "get_weather>\n", "<parameter=city>\n", "Oslo", "\n</parameter>\n",
                 "<parameter=unit>\n", "celsius", "\n</parameter>\n", "</function>\n",
                 Special("</tool_call>"), Special("<|im_end|>")]),
            CALL_STEPS[:1] + [dec.Step("param", "city", "Oslo"), dec.Step("param_end", "city")] + CALL_STEPS[3:]
            + [dec.Step("turn_end")])

    def test_a_named_mid_call_entry_starts_at_the_parameters(self):
        # The prompt already wrote "<function=NAME>\n", so only the parameters and the close are generated.
        self.assertEqual(
            run("entry_call_named_t0_get_weather", WEATHER_LABELS,
                ["<parameter=city>\n", "Oslo", "\n</parameter>\n", "</function>\n",
                 Special("</tool_call>"), Special("<|im_end|>")]),
            [dec.Step("param", "city", "Oslo"), dec.Step("param_end", "city"),
             dec.Step("call_end", "get_weather"), dec.Step("turn_end")])

    def test_the_named_mid_call_entry_needs_a_declared_tool(self):
        with self.assertRaises(dec.StructureError):
            dec.Decoder("entry_call_named_t9_not_declared", WEATHER_LABELS)


class Decoys(unittest.TestCase):
    """The bytes of a delimiter are prose.  Only an id or a labelled literal is structure."""

    def test_a_spelled_tool_call_inside_content_stays_content(self):
        prose = 'I will use <tool_call> blocks, e.g. <function=get_weather>:'
        self.assertEqual(
            run("entry_answer_tools", WEATHER_LABELS, [prose, Special("<|im_end|>")]),
            [dec.Step("content", text=prose), dec.Step("turn_end")])

    def test_a_spelled_think_close_inside_reasoning_stays_reasoning(self):
        reasoning = "\nthe tag </think> is how it ends\n"     # noqa: S105 - a delimiter, not a secret
        self.assertEqual(
            run("entry_think_plain", Labels(),
                [reasoning, Special("</think>"), "\n\n", "ok", Special("<|im_end|>")]),
            [dec.Step("reasoning", text=reasoning), dec.Step("content", text="ok"), dec.Step("turn_end")])

    def test_a_fenced_call_in_prose_produces_no_structure(self):
        prose = "```\n<tool_call>\n<function=get_weather>\n```"
        steps = run("entry_answer_tools", WEATHER_LABELS, [prose, Special("<|im_end|>")])
        self.assertEqual([s.kind for s in steps], ["content", "turn_end"])


class ParameterValues(unittest.TestCase):
    def test_only_a_standalone_closer_line_ends_a_value(self):
        # D5/A7: `</parameter>` inside a value, with anything else on its line, is part of the value.
        value = "a </parameter> that is not alone\nand a second line</parameter> end"
        steps = run("entry_answer_tools", WEATHER_LABELS,
                    ["", Special("<tool_call>"), "\n<function=", "get_weather>\n",
                     "<parameter=city>\n", value, "\n</parameter>\n", "</function>\n",
                     Special("</tool_call>"), Special("<|im_end|>")])
        self.assertEqual(steps[1:3], [dec.Step("param", "city", value), dec.Step("param_end", "city")])

    def test_a_cut_reply_flushes_a_partial_closer_as_value_text(self):
        # "\n</para" could still become "\n</parameter>\n", so it is held; the end of the reply makes it text.
        self.assertEqual(
            run("entry_answer_tools", WEATHER_LABELS,
                ["", Special("<tool_call>"), "\n<function=", "get_weather>\n",
                 "<parameter=city>\n", "Oslo\n</para"], size=3),
            [dec.Step("call_open", "get_weather"), dec.Step("param", "city", "Oslo\n</para")])

    def test_an_empty_value_is_a_param_end_with_no_text(self):
        steps = run("entry_answer_tools", WEATHER_LABELS,
                    ["", Special("<tool_call>"), "\n<function=", "get_weather>\n",
                     "<parameter=city>\n", "\n</parameter>\n", "</function>\n",
                     Special("</tool_call>"), Special("<|im_end|>")])
        self.assertEqual(steps[1:2], [dec.Step("param_end", "city")])


class Drift(unittest.TestCase):
    """The decoder's alphabet against the two files that write it."""

    def test_the_literals_the_decoder_knows_are_the_ones_the_skeleton_writes(self):
        skeleton = strip_comments(bg.skeleton())
        for literal in dec.SKELETON_LITERALS:
            with self.subTest(literal=literal):
                self.assertIn(json.dumps(literal), skeleton,
                              f"decode.py knows {literal!r}, which skeleton.lark no longer writes")

    def test_the_skeletons_literals_are_exactly_the_ones_the_decoder_knows(self):
        skeleton = strip_comments(blank_json_payloads(bg.skeleton()))
        found = {json.loads(m.group(0)) for m in re.finditer(r'"(?:[^"\\]|\\.)*"', skeleton)}
        self.assertTrue(found, "no literal found: the scan is broken")
        self.assertEqual(found, set(dec.SKELETON_LITERALS),
                         "the skeleton and the decoder disagree about a structural literal")

    def test_the_special_tokens_the_skeleton_names_are_the_ones_the_decoder_knows(self):
        skeleton = strip_comments(blank_json_payloads(bg.skeleton()))
        for name in ("</think>", "<tool_call>", "</tool_call>", "<|im_end|>"):
            with self.subTest(name=name):
                self.assertIn(name, skeleton)
                self.assertIn(guard.DELIMITER_IDS[name], dec._NAME_OF_ID)

    def test_the_fragments_literals_are_the_ones_the_decoder_knows(self):
        # The close of a parameter block is the fragment's, not the skeleton's: the skeleton only names
        # `raw_param_end`, whose pattern is where the same bytes live as a regex.
        for literal in dec.FRAGMENT_LITERALS:
            with self.subTest(literal=literal):
                self.assertIn(json.dumps(literal), compile_tools_fragment([WEATHER]).text)

    def test_the_fragment_labels_name_the_literals_it_wrote(self):
        fragment = compile_tools_fragment([WEATHER, LISTING])
        self.assertEqual(len(fragment.labels.tools), 2)
        self.assertEqual([t.name for t in fragment.labels.tools], ["get_weather", "list_files"])
        for label in fragment.labels.tools:
            self.assertIn(json.dumps(label.literal), fragment.text)
        for label in fragment.labels.params:
            self.assertIn(json.dumps(label.literal), fragment.text)
        # ... and every parameter block in the text is one of them (a value rule cannot hide a parameter).
        written = set(re.findall(r"<parameter=([^>\\]+)>", fragment.text))
        self.assertEqual(written, {p.name for p in fragment.labels.params})

    def test_a_fragment_with_no_tools_has_no_labels(self):
        self.assertEqual(compile_tools_fragment([]).labels, Labels())
        self.assertEqual(compile_tools_fragment([]).text, bg.DEFAULT_TOOLS)


def strip_comments(text: str) -> str:
    """Lark comments are prose about the grammar; they quote delimiters the decoder has no business matching."""
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def blank_json_payloads(text: str) -> str:
    """`%json` payloads are JSON (serde_json parses them), so their quoted strings are not Lark literals."""
    out, i = [], 0
    while True:
        at = text.find("%json ", i)
        if at < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:at + len("%json ")])
        depth, j, instr, esc = 0, at + len("%json "), False, False
        while j < len(text):
            c = text[j]
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
        out.append("0" * (j - (at + len("%json "))))
        i = j


if __name__ == "__main__":
    unittest.main()
