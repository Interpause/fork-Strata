"""The transport: one `grammar=<path>` key on the engine line, one file that does not outlive its request.

    python -m unittest serve.test_grammar_transport -v

Host-only: the request is driven through `Service.run` with the mock engine, the same way the server tests do.  The
server compiles nothing, so what these tests are about is the file alone - written, held, adopted, removed - and they
need no `tools/build_llguidance.sh`.  Since D-030 a request that cannot be given a grammar is refused rather than
decoded unconstrained, so `Refusals` below pins each of the six conditions and `ClientVisible` pins what a client
sees on the four request shapes.
"""
import base64
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import unittest.mock
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server_module  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.grammar import matcher as grammar_matcher  # noqa: E402
from serve.grammar import pipeline as grammar_pipeline  # noqa: E402
from serve.server import ByteTokenizer, FIXTURE_TOKENIZER, MockEngine, Service, StrataEngine  # noqa: E402
from serve.server import drop_grammar, grammar_file, hold_grammar, hold_grammar_file  # noqa: E402
from serve.server import serve, sweep_grammar_dirs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096
ANSWER = "Hello."                               # a script `entry_answer_plain` accepts: the answer, nothing before it
# A thinking turn's prompt tail already ends at the template's `<think>\n` (entries.THINKING_ON_TAIL), so the reply
# starts inside the thinking block: some reasoning, the close, the two-newline separator, then the answer.
THINKING = "\n\n</think>\n\n" + ANSWER


def thinking_call(name):
    """The reply a required-call thinking entry allows: a closed thinking block, then one call to `name`."""
    return "\n\n</think>\n\n<tool_call>\n<function=%s>\n</function>\n</tool_call>" % name


class Recorder(MockEngine):
    """Records what the server sent, and reads the grammar file while the engine would still need it."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        path = sampling.get("strata_grammar")
        self.calls.append({"sampling": dict(sampling), "path": path,
                           "existed": bool(path) and Path(path).exists(),
                           "text": Path(path).read_text(encoding="utf-8") if path else None})
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class SixTagTokenizer(ByteTokenizer):
    """The stand-in with all six structural tags at the pack's own ids.

    `ByteTokenizer` carries the grammar's four delimiters; `<tool_response>` and `</tool_response>` are the two it
    lacks, and a fence ban names them too (D-067).  Only a whole tag is resolved here, and only for the resolver's
    own call: the prompt's encoding stays the stand-in's byte-for-byte, so a marker inside a message is still text
    (which is what `test_mcp`'s prompt assertions read).
    """

    STRUCTURAL = {"<tool_response>": 248066, "</tool_response>": 248067}

    def encode(self, text, parse_special=False, plain=()):
        if parse_special and not plain and text in self.STRUCTURAL:
            return [self.STRUCTURAL[text]]
        return super().encode(text, parse_special=parse_special, plain=plain)


def service(pack_dir=None, reply=ANSWER, tok=None):
    """`reply` is the script the engine replays, so it must be one the grammar of the entry state applied to it can
    actually emit: an answer-only turn for `thinking=False`, a reply that starts inside the thinking block (the
    prompt's tail already ends at `<think>\\n`) for `thinking=True`.  A list gives one script per request, in the
    order they run."""
    tok = tok or ByteTokenizer()
    engine = Recorder(tok, reply, max_context=CTX)
    # The engine says it compiles a grammar: `begin_grammar` needs `INFO grammar=1` as well as a pipeline (7ee998b5,
    # D-043), because a file it would strip is a file there is nothing to read the reply with.
    engine.info = {"grammar": 1} if pack_dir is not None else {}
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), pack_dir=pack_dir)
    return svc, engine, tok


def run(svc, tok, thinking=False, tools=None, force=None, sampling=None):
    ids = svc.prepare([{"role": "user", "content": "hi"}], tools, {"enable_thinking": thinking},
                      req=sampling if sampling is not None else {}, force=force)[0]
    cancel = threading.Event()
    return list(svc.run(ids, thinking, tools, 8, dict(sampling or {}), cancel, force=force))


class Key(unittest.TestCase):
    def test_a_grammar_is_one_key_before_the_ids(self):
        # Base64 always (D-047): the engine decodes it before opening the path.
        want = base64.b64encode(b"/dev/shm/g/ab12.gr").decode("ascii")
        self.assertEqual(StrataEngine.sampling_keys({"strata_grammar": "/dev/shm/g/ab12.gr"}),
                         " grammar=" + want)

    def test_no_grammar_is_no_key(self):
        for sampling in ({}, {"strata_grammar": None}, {"strata_grammar": ""}, {"strata_grammar": 7}):
            with self.subTest(sampling=sampling):
                self.assertEqual(StrataEngine.sampling_keys(sampling), "")

    def test_a_path_with_a_space_travels(self):
        # A space ends a key on this line, so an unencoded path with one used to be dropped here and decode
        # unconstrained with nothing said - every Windows profile under a name with a space (D-047).
        keys = StrataEngine.sampling_keys({"strata_grammar": "/C:/Users/John Smith/a b.c.gr"})
        self.assertEqual(keys, " grammar=" + base64.b64encode(b"/C:/Users/John Smith/a b.c.gr").decode("ascii"))
        self.assertNotIn(" ", keys.strip())

    def test_the_key_sits_beside_the_other_keys(self):
        keys = StrataEngine.sampling_keys({"temperature": 0.5, "strata_grammar": "/x/y.gr"})
        self.assertEqual(keys, " temperature=0.5 grammar=" + base64.b64encode(b"/x/y.gr").decode("ascii"))

    # D-067: the ids a Markdown code fence hides, in the grammar branch and nowhere else.
    FENCE_IDS = (248058, 248059, 248066, 248067, 248068, 248069)   # the six, ascending

    def test_a_fence_ban_rides_beside_the_grammar(self):
        keys = StrataEngine.sampling_keys({"strata_grammar": "/dev/shm/g/ab12.gr",
                                           "strata_fence_ban": list(self.FENCE_IDS)})
        want = base64.b64encode(b"/dev/shm/g/ab12.gr").decode("ascii")
        self.assertEqual(keys, " grammar=" + want
                         + " fence_ban=248058,248059,248066,248067,248068,248069")
        self.assertNotIn(" fence_ban= ", keys)                  # never an empty value: the engine refuses one
        self.assertNotIn(" ", keys.split("fence_ban=")[1])      # decimal ids and commas only

    def test_no_grammar_sends_no_fence_ban(self):
        # The engine tracks no fence without a grammar, so this line must be byte-identical to the one it sent before.
        for sampling in ({}, {"strata_fence_ban": list(self.FENCE_IDS)},
                         {"strata_fence_ban": list(self.FENCE_IDS), "temperature": 0.5}):
            with self.subTest(sampling=sampling):
                self.assertNotIn("fence_ban", StrataEngine.sampling_keys(sampling))

    def test_a_list_the_engine_would_refuse_is_not_sent(self):
        want = " grammar=" + base64.b64encode(b"/x/y.gr").decode("ascii")
        for empty in ([], (), ["248058"], [True], [-1], [248058, "248059"]):
            with self.subTest(fence_ban=empty):
                keys = StrataEngine.sampling_keys({"strata_grammar": "/x/y.gr", "strata_fence_ban": empty})
                self.assertEqual(keys, want)

    def test_an_unordered_list_is_sent_ascending(self):
        keys = StrataEngine.sampling_keys({"strata_grammar": "/x/y.gr",
                                           "strata_fence_ban": [248069, 248058, 248067]})
        self.assertTrue(keys.endswith(" fence_ban=248058,248067,248069"), keys)


class FenceBan(unittest.TestCase):
    """D-067: which ids the server names, resolved from the pack's tokenizer rather than from a table.

    Six structural tags, never a control token, deterministic (ascending) so the line can be pinned.  A tag the
    tokenizer cannot give a single id is omitted; a tokenizer that resolves none sends no key at all.
    """

    IDS = Key.FENCE_IDS

    def test_the_six_structural_tags_resolve_ascending(self):
        self.assertEqual(server_module.fence_ban_ids(SixTagTokenizer()), self.IDS)

    def test_no_control_token_is_named(self):
        tok = SixTagTokenizer()
        banned = set(server_module.fence_ban_ids(tok))
        for tag in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            with self.subTest(tag=tag):
                self.assertNotIn(tok.encode(tag, parse_special=True)[0], banned)
        # and not the vision markers either: a fenced picture marker is content, not an image slot
        for tag in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"):
            with self.subTest(tag=tag):
                self.assertNotIn(tok.encode(tag, parse_special=True)[0], banned)

    def test_a_tag_that_does_not_resolve_is_omitted(self):
        class NoToolResponse(ByteTokenizer):
            def encode(self, text, parse_special=False, plain=()):
                if text in ("<tool_response>", "</tool_response>"):
                    raise KeyError(text)            # a pack that does not know the tag
                return super().encode(text, parse_special=parse_special, plain=plain)

        self.assertEqual(server_module.fence_ban_ids(NoToolResponse()), (248058, 248059, 248068, 248069))

    def test_a_tag_split_over_several_ids_is_omitted(self):
        class Bytes(ByteTokenizer):
            def encode(self, text, parse_special=False, plain=()):
                return list(text.encode())          # one id per byte: no single id to ban

        self.assertEqual(server_module.fence_ban_ids(Bytes()), ())

    def test_a_request_with_a_grammar_tells_the_engine_the_six(self):
        with tempfile.TemporaryDirectory() as d:
            pack = Path(d)
            (pack / "vocab.json").write_text("{}", encoding="utf-8")   # a pack-shaped directory
            svc, engine, tok = service(pack, tok=SixTagTokenizer())
            run(svc, tok)
            sampling = engine.calls[0]["sampling"]
            self.assertEqual(sampling.get("strata_fence_ban"), self.IDS)
            self.assertTrue(StrataEngine.sampling_keys(sampling).endswith(
                " fence_ban=248058,248059,248066,248067,248068,248069"))

    def test_a_request_without_a_grammar_sends_none(self):
        # D-030 refuses a chat-shaped request with no grammar, so the no-grammar path is the one that returns no file
        # (as `StatusHandover` drives): nothing about that line changes.
        with tempfile.TemporaryDirectory() as d:
            pack = Path(d)
            (pack / "vocab.json").write_text("{}", encoding="utf-8")
            svc, engine, tok = service(pack, tok=SixTagTokenizer())
            svc.begin_grammar = lambda *a, **k: None
            list(svc.run(tok.encode("hi"), False, None, 8, {}, threading.Event()))
            sampling = engine.calls[0]["sampling"]
            self.assertNotIn("strata_fence_ban", sampling)
            self.assertEqual(StrataEngine.sampling_keys(sampling), "")


class Request(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.pack = Path(self.dir.name)
        (self.pack / "vocab.json").write_text("{}", encoding="utf-8")   # a pack-shaped directory, not a real pack
        self.svc, self.engine, self.tok = service(self.pack)

    def tearDown(self):
        self.dir.cleanup()

    def test_the_engine_gets_one_file_that_is_the_spliced_grammar(self):
        run(self.svc, self.tok)
        self.assertEqual(len(self.engine.calls), 1)
        call = self.engine.calls[0]
        self.assertTrue(call["existed"], "the file must exist while the engine runs")
        self.assertEqual(call["path"], str(Path(call["path"])))
        self.assertTrue(call["path"].endswith(".gr"))
        self.assertIn("start: entry_answer_plain", call["text"])
        self.assertEqual(self.svc.pipeline.stats()["misses"], 1)

    def test_the_file_is_named_for_its_own_bytes(self):
        run(self.svc, self.tok)
        call = self.engine.calls[0]
        grammar = self.svc.pipeline.for_request(self.svc.tok.encode("x<think>\n\n</think>\n\n"))
        self.assertEqual(Path(call["path"]).stem, grammar.id)

    def test_a_shared_grammar_file_outlives_the_first_request_that_drops_it(self):
        # Two requests with the same prompt tail are sent the same file (its name is the grammar's identity), so the
        # first one to finish must not delete what the second has yet to read - both run at once, each in its slot.
        path = grammar_file(self.pack, "ab12", "start: entry_answer_plain")
        hold_grammar(path)
        hold_grammar(path)
        drop_grammar(path)
        self.assertTrue(path.exists(), "the second holder still needs it")
        drop_grammar(path)
        self.assertFalse(path.exists(), "the last holder deletes it")

    def test_a_drop_with_no_hold_leaves_the_file_alone(self):
        # The engine's `fopen` reads the file later than the server writes it.  A drop that holds nothing must
        # therefore not delete: the file may be the one a request adopted a moment ago and has not sent yet (the live
        # `ERR grammar: <path> cannot be read` that reached a client as a 400).
        path = hold_grammar_file(self.pack, "ab12", "start: entry_answer_plain")
        for _ in range(3):
            drop_grammar(path)                   # one hold, three drops
        self.assertFalse(path.exists(), "the one hold it had was released")
        drop_grammar(path)                       # nobody holds it any more: no hold of ours to release
        again = grammar_file(self.pack, "ab12", "start: entry_answer_plain")
        self.assertTrue(again.exists(), "writing a file nobody holds is not undone by a stale drop")
        drop_grammar(again)                      # still nobody holds it: left alone, as it was written
        self.assertTrue(again.exists())

    def test_adopting_a_shared_file_cannot_race_the_drop_that_deletes_it(self):
        # The window the fix closes: request P finds Q's file already written (so nothing is written), and Q's drop
        # lands before P's hold.  Adopting is one step now, so a path a live holder was given always has its file -
        # whichever thread gets there first.  The pair is run many times because the window is a few instructions
        # wide, and every round must hold.
        text = "start: entry_answer_plain"
        missing = []
        for _ in range(200):
            first = hold_grammar_file(self.pack, "ab12", text)      # Q holds the shape already

            def other():                                            # Q finishes while P is adopting
                drop_grammar(first)

            t = threading.Thread(target=other)
            t.start()
            adopted = hold_grammar_file(self.pack, "ab12", text)    # P adopts the same shape
            t.join(5)
            if not adopted.exists():
                missing.append(str(adopted))
            self.assertTrue(adopted.exists(), "Q's drop must not take the file P was given")
            drop_grammar(adopted)                                   # P's own drop is what removes it
            self.assertFalse(adopted.exists())
        self.assertEqual(missing, [], "a file a live holder was given was missing")

    def test_the_file_is_gone_after_the_request(self):
        run(self.svc, self.tok)
        self.assertFalse(Path(self.engine.calls[0]["path"]).exists())

    def test_the_file_is_gone_when_the_request_fails(self):
        svc, engine, tok = service(self.pack)
        ids = svc.prepare([{"role": "user", "content": "hi"}], None, {"enable_thinking": False})[0]
        stream = svc.run(ids, False, None, 8, {}, threading.Event())
        next(stream)                                        # the request has started: the file exists
        path = engine.calls[0]["path"]
        self.assertTrue(Path(path).exists())
        stream.close()                                      # the client walked away
        self.assertFalse(Path(path).exists())

    def test_two_request_shapes_are_two_files_and_one_splice_each(self):
        # One script per request, in the order they run: an answer-only turn, then a thinking one.
        self.svc, self.engine, self.tok = service(self.pack, [ANSWER, THINKING])
        run(self.svc, self.tok)
        run(self.svc, self.tok, thinking=True)
        paths = [call["path"] for call in self.engine.calls]
        self.assertEqual(len(set(paths)), 2)
        self.assertEqual(self.svc.pipeline.stats()["misses"], 2)
        for call in self.engine.calls:
            self.assertFalse(Path(call["path"]).exists())

    def test_the_same_shape_twice_is_one_file_and_one_splice(self):
        run(self.svc, self.tok)
        run(self.svc, self.tok)
        self.assertEqual(len(set(call["path"] for call in self.engine.calls)), 1)
        self.assertEqual(self.svc.pipeline.stats(), {"hits": 1, "misses": 1, "shapes": 1,
                                                     "max": self.svc.pipeline.max})

    def test_a_required_call_drops_the_prose_only_branch(self):
        tools = [{"name": "f", "parameters": {"type": "object", "properties": {}}}]   # as the frontend passes them
        # A thinking turn starts at entry_think*, and a required call's branch has no prose: the script is the call.
        self.svc, self.engine, self.tok = service(self.pack, thinking_call("f"))
        run(self.svc, self.tok, thinking=True, tools=tools, force=("required", None))
        self.assertIn("start: entry_think_tools_required", self.engine.calls[0]["text"])

    def test_a_named_choice_offers_only_that_tool(self):
        """A named `tool_choice` narrows the fragment: `tool_dispatch` would otherwise offer every declared tool."""
        tools = [{"name": n, "parameters": {"type": "object", "properties": {}}} for n in ("write", "read")]
        # the fragment's rules carry each tool's index: call_t0_write is "write", call_t1_read is "read"
        # A thinking turn starts at entry_think*, and a named choice's branch has no prose: the script is the call.
        self.svc, self.engine, self.tok = service(self.pack, thinking_call("write"))
        run(self.svc, self.tok, thinking=True, tools=tools, force=("named", "write"))
        self.assertIn("call_t0_write", self.engine.calls[-1]["text"])
        self.assertNotIn("call_t1_read", self.engine.calls[-1]["text"])
        # without the narrowing both are offered, so the assertions above cannot pass vacuously
        run(self.svc, self.tok, thinking=True, tools=tools, force=None)
        self.assertIn("call_t0_write", self.engine.calls[-1]["text"])
        self.assertIn("call_t1_read", self.engine.calls[-1]["text"])

    def test_without_thinking_a_required_call_starts_at_the_answer_entry(self):
        # Nothing is written into the prompt for tool_choice: with thinking off it already ends at the template's
        # answer tail (an empty thinking block), so the reply starts at the answer entry with the prose-only branch
        # dropped - there is no mid-call prompt here to start inside (D-053).
        tools = [{"name": "f", "parameters": {"type": "object", "properties": {}}}]   # as the frontend passes them
        run(self.svc, self.tok, thinking=False, tools=tools, force=("required", None))
        self.assertIn("start: entry_answer_tools_required", self.engine.calls[0]["text"])
        self.assertNotIn("start: entry_call_name", self.engine.calls[0]["text"])

    def test_a_prompt_without_a_template_tail_is_refused_as_a_server_fault(self):
        # A completion-style prompt: every prompt a chat route decodes is rendered by this template, so a tail with
        # no entry is template drift, and the request fails instead of decoding unconstrained (D-030).
        ids = self.tok.encode("plain prompt, no chat template")
        with self.assertRaises(grammar_pipeline.GrammarUnavailable) as caught:
            list(self.svc.run(ids, False, None, 8, {}, threading.Event()))
        self.assertEqual(caught.exception.status, 500)
        self.assertIn("no state the grammar has an entry for", str(caught.exception))
        self.assertEqual(self.engine.calls, [], "nothing may decode a request with no grammar")

    def test_a_server_without_a_pack_tokenizer_refuses_every_chat(self):
        # The old property - "carries no grammar, so the reply is unconstrained" - is exactly what D-030 deletes.
        svc, engine, tok = service(None)
        self.assertIsNone(svc.pipeline)
        with self.assertRaises(grammar_pipeline.GrammarUnavailable) as caught:
            run(svc, tok)
        self.assertEqual(caught.exception.status, 503)
        self.assertIn("no grammar pipeline", str(caught.exception))
        self.assertEqual(engine.calls, [])

    def test_the_json_entry_uses_the_requests_schema(self):
        sampling = {"response_format": {"type": "json_schema", "json_schema": {
            "name": "answer", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}}}
        run(self.svc, self.tok, sampling=sampling)
        text = self.engine.calls[0]["text"]
        self.assertIn("start: entry_answer_json", text)
        self.assertIn('"a"', text)

    def test_a_response_format_that_is_not_a_schema_fails_before_the_engine(self):
        # The property this replaces was "no grammar is sent and the reply is still decoded" - which is the one
        # D-030 deletes.  `format_schema` is the single reader of response_format's shapes, and `[]` is not a
        # schema: the request is malformed (400), so it never reaches the engine and no grammar file exists for it.
        sampling = {"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": []}}}
        with self.assertRaises(ValueError) as caught:
            run(self.svc, self.tok, sampling=sampling)
        self.assertIn("json_schema needs a schema object", str(caught.exception))
        self.assertEqual(self.engine.calls, [], "a refused shape must not be decoded")


class Refusals(unittest.TestCase):
    """D-030: a request that cannot be constrained is refused, with a message and a status of its own.

    Six conditions used to send no grammar and say so once per process.  Each is now a
    `grammar_pipeline.GrammarUnavailable`, the status names whose condition it is (503 the server cannot supply or
    write a grammar, 500 Strata's copy is at fault, 400 the request asks for something no grammar can express), and
    the engine is never called - so no reply decodes unconstrained.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.pack = Path(self.dir.name)
        (self.pack / "vocab.json").write_text("{}", encoding="utf-8")
        self.svc, self.engine, self.tok = service(self.pack)

    def tearDown(self):
        self.dir.cleanup()

    def refused(self, ids=None, **run_kw):
        """Run one request and return the refusal, asserting nothing decoded."""
        if ids is None:
            ids = self.svc.prepare([{"role": "user", "content": "hi"}], run_kw.pop("tools", None),
                                   {"enable_thinking": False}, req=run_kw.get("sampling") or {},
                                   force=run_kw.get("force"))[0]
        with self.assertRaises(grammar_pipeline.GrammarUnavailable) as caught:
            list(self.svc.run(ids, run_kw.get("thinking", False), run_kw.get("tools"), 8,
                              dict(run_kw.get("sampling") or {}), threading.Event(), force=run_kw.get("force")))
        self.assertEqual(self.engine.calls, [], "the engine must not decode a request with no grammar")
        return caught.exception

    def test_no_pack_tokenizer_refuses_with_its_own_message(self):
        svc, engine, tok = service(None)
        with self.assertRaises(grammar_pipeline.GrammarUnavailable) as caught:
            run(svc, tok)
        self.assertEqual(caught.exception.status, 503)
        self.assertIn("no grammar pipeline", str(caught.exception))
        self.assertIn("without a pack tokenizer", str(caught.exception))
        self.assertEqual(engine.calls, [])

    def test_an_engine_without_grammar_one_refuses_with_its_own_message(self):
        self.engine.info = {}                     # an older engine, or one built without tools/build_llguidance.sh
        e = self.refused()
        self.assertEqual(e.status, 503)
        self.assertIn("INFO line has no grammar=1", str(e))
        self.assertIn("tools/build_llguidance.sh", str(e))

    def test_a_prompt_tail_the_template_never_writes_is_refused_as_a_server_fault(self):
        e = self.refused(ids=self.tok.encode("plain prompt, no chat template"))
        self.assertEqual(e.status, 500)           # the server rendered every prompt: this is template drift
        self.assertIn("no state the grammar has an entry for", str(e))

    def test_a_prompt_inside_an_undeclared_call_is_refused_as_the_requests_fault(self):
        e = self.refused(ids=self.tok.encode("<|im_start|>assistant\n<function=gone>\n", parse_special=True))
        self.assertEqual(e.status, 400)           # the conversation names a tool this request did not declare
        self.assertIn("'gone'", str(e))
        self.assertIn("did not declare", str(e))

    def test_a_required_call_with_no_tools_is_refused_as_the_requests_fault(self):
        e = self.refused(ids=self.tok.encode("<think>\n\n</think>\n\n"), force=("required", None))
        self.assertEqual(e.status, 400)           # tool_choice required with no tools is the request's own contradiction
        self.assertIn("requires a call but the request declared no tools", str(e))

    def test_a_skeleton_that_no_longer_splices_is_refused_as_a_server_fault(self):
        with unittest.mock.patch.object(server_module.grammar_pipeline.bg, "skeleton",
                                        return_value="no splice markers here"):
            e = self.refused()
        self.assertEqual(e.status, 500)
        self.assertIn("skeleton no longer splices", str(e))
        self.assertIn("SPLICE", str(e))           # the builder names the marker it could not find

    def test_a_grammar_file_that_cannot_be_written_is_refused_as_a_capacity_fault(self):
        with unittest.mock.patch.object(server_module, "hold_grammar_file",
                                        side_effect=OSError("No space left on device")):
            e = self.refused()
        self.assertEqual(e.status, 503)
        self.assertIn("grammar file cannot be written", str(e))
        self.assertIn("No space left on device", str(e))

    def test_a_grammar_file_with_an_unusable_name_is_refused_as_a_server_fault(self):
        with unittest.mock.patch.object(server_module, "hold_grammar_file",
                                        side_effect=ValueError("a grammar file needs a name")):
            e = self.refused()
        self.assertEqual(e.status, 500)
        self.assertIn("name is unusable", str(e))

    def test_a_response_schema_the_ladder_cannot_repair_is_the_requests_400(self):
        # D-065: the ladder is run before the request is admitted, and when it runs out of rungs the client is told
        # why in the compiler's own words.  Skipped without the library, because only compiling can tell whether a
        # schema is repairable.  `minProperties` > `maxProperties` is legal JSON Schema (the gate accepts it), the
        # compiler calls it unsatisfiable, and the message names no property for a rung to widen.
        try:
            grammar_matcher.vocabulary(FIXTURE_TOKENIZER)
        except grammar_matcher.LibraryMissing as e:
            self.skipTest(f"needs libllguidance ({e})")
        sampling = {"response_format": {"type": "json_schema", "json_schema": {
            "name": "limits", "schema": {"type": "object", "minProperties": 5, "maxProperties": 2}}}}
        e = self.refused(sampling=sampling)
        self.assertEqual(e.status, 400)
        self.assertEqual(e.kind, "invalid_request_error")
        self.assertIn("minProperties > maxProperties", str(e))

    def test_a_repaired_schema_says_which_rung_it_took(self):
        # D-058's reporting rule: one line per request that relaxed something, naming the rung, the class and the
        # pointer, so "which schema failed and why" needs no reply to inspect.  The same request without a repair
        # says nothing at all: a log line per request would bury the ones that matter.
        try:
            grammar_matcher.vocabulary(FIXTURE_TOKENIZER)
        except grammar_matcher.LibraryMissing as e:
            self.skipTest(f"needs libllguidance ({e})")
        repairable = {"response_format": {"type": "json_schema", "json_schema": {
            "name": "look", "schema": {"type": "object", "properties": {
                "a": {"type": "string", "pattern": "(?=a)"}}}}}}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            list(self.svc.run(self.svc.prepare([{"role": "user", "content": "hi"}], None,
                                               {"enable_thinking": False}, req=repairable)[0],
                              False, None, 8, dict(repairable), threading.Event()))
        logged = out.getvalue()
        self.assertIn("[strata] grammar entry_answer_json", logged)
        self.assertIn("rung 2 unsupported", logged)
        self.assertIn("/properties/a/pattern", logged)
        self.assertEqual(len([line for line in logged.splitlines() if "rung " in line]), 1)

        clean = {"response_format": {"type": "json_schema", "json_schema": {
            "name": "look", "schema": {"type": "object"}}}}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            list(self.svc.run(self.svc.prepare([{"role": "user", "content": "hi"}], None,
                                               {"enable_thinking": False}, req=clean)[0],
                              False, None, 8, dict(clean), threading.Event()))
        self.assertNotIn("rung ", out.getvalue())


class ClientVisible(unittest.TestCase):
    """What the client sees when `begin_grammar` refuses inside a generator that has not yielded yet.

    The raise happens on the first `next()`, before the engine is called and before any answer text, so a
    non-streaming request is answered with the refusal's status.  A streaming request has already sent its 200 and
    its headers, so the refusal travels as that surface's error event instead - with the same message and code.
    """

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, "</think>\n\nhi", max_context=CTX)
        # No pack: no pipeline, which is condition 1 - a server-side condition that must not be a 400.  `pack_dir`
        # is explicit because the Service's default is now a lazily made one (AUTO_PACK), which would build a
        # pipeline and answer 200 instead of refusing.
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), pack_dir=None)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read().decode()
            return response.status, data if "event-stream" in response.headers.get("Content-Type", "") else json.loads(data)

    def test_a_chat_request_with_no_grammar_is_a_503_server_error_not_a_400(self):
        code, body = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 503)
        self.assertEqual((body["error"]["type"], body["error"]["code"]), ("server_error", "grammar_unavailable"))
        self.assertIn("no grammar pipeline", body["error"]["message"])

    def test_a_streaming_chat_sends_the_refusal_as_an_error_event_after_its_headers(self):
        # `_sse()` writes the 200 and its headers before the first `next()`, so this cannot be a status any more.
        code, raw = self.post("/v1/chat/completions",
                              {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        self.assertEqual(code, 200)
        self.assertIn('"code": "grammar_unavailable"', raw)
        self.assertIn('"type": "server_error"', raw)
        self.assertIn("no grammar pipeline", raw)
        self.assertIn("data: [DONE]", raw)

    def test_a_responses_request_with_no_grammar_is_a_503_server_error(self):
        code, body = self.post("/v1/responses", {"input": "hi"})
        self.assertEqual(code, 503)
        self.assertEqual((body["error"]["type"], body["error"]["code"]), ("server_error", "grammar_unavailable"))

    def test_a_streaming_responses_request_reports_the_refusal_as_response_failed(self):
        code, raw = self.post("/v1/responses", {"input": "hi", "stream": True})
        self.assertEqual(code, 200)
        self.assertIn("response.failed", raw)
        self.assertIn('"grammar_unavailable"', raw)


class Routing(unittest.TestCase):
    """Which requests a batching engine keeps off its slots.

    A request whose `grammar=` will actually travel runs in a slot when the engine masks each row of its batch
    windows (`INFO` then carries `gbatch=1`).  An engine with no such claim - one whose slots are pipelined in
    groups, where every launched row is kept and a row that must draw nothing cannot stop - keeps the request on the
    solo path, because a slot would decode it unconstrained while the request says it is masked (D-027)."""

    @staticmethod
    def engine(batch, info):
        eng = object.__new__(StrataEngine)               # solo_only reads batch and info only
        eng.batch, eng.info = batch, info
        return eng

    def test_a_grammar_request_is_solo_when_the_engine_cannot_mask(self):
        self.assertTrue(self.engine(3, {"grammar": 1}).solo_only({"strata_grammar": "/dev/shm/g/ab12.gr"}))

    def test_a_grammar_request_keeps_its_slot_when_the_engine_can_mask(self):
        self.assertFalse(self.engine(3, {"grammar": 1, "gbatch": 1})
                         .solo_only({"strata_grammar": "/dev/shm/g/ab12.gr"}))

    def test_a_plain_request_keeps_its_slot(self):
        eng = self.engine(3, {"grammar": 1})
        for sampling in ({}, None, {"temperature": 0.5}, {"strata_grammar": ""}):
            with self.subTest(sampling=sampling):
                self.assertFalse(eng.solo_only(sampling))

    def test_an_engine_that_cannot_compile_one_keeps_its_slots(self):
        # keys() drops the key for such an engine, so nothing has to be serialised on its behalf.
        self.assertFalse(self.engine(3, {}).solo_only({"strata_grammar": "/dev/shm/g/ab12.gr"}))

    def test_an_engine_without_slots_is_unaffected(self):
        self.assertTrue(self.engine(0, {"grammar": 1}).solo_only({"strata_grammar": "/dev/shm/g/ab12.gr"}))


class Sweep(unittest.TestCase):
    """A server that died without its `finally` leaves its directory behind, and the next server to start removes it
    (D-046).  The pid is in the directory's name, so a live one is how a server that is still running is spared."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.pack = Path(self.dir.name)
        self.shm = Path(self.dir.name) / "shm"
        self.shm.mkdir()
        patcher = unittest.mock.patch.object(server_module, "SHM_DIR", self.shm)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.dir.cleanup)

    def dead_pid(self):
        """A process id nothing is using: a child that has exited and been waited for."""
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        return child.pid

    def made(self, root, pid, name="ab12.gr"):
        d = root / f"strata-grammar-{pid}"
        d.mkdir(parents=True)
        (d / name).write_text("start: entry_answer_plain", encoding="utf-8")
        return d

    def test_the_directory_of_a_server_that_is_gone_goes(self):
        d = self.made(self.shm, self.dead_pid())
        sweep_grammar_dirs(self.pack)
        self.assertFalse(d.exists(), "a dead server's files would otherwise stay in memory until a reboot")

    def test_a_server_that_is_running_keeps_its_directory(self):
        live = self.made(self.shm, os.getpid())                 # this process is alive, like a server on its pack
        dead = self.made(self.shm, self.dead_pid())
        sweep_grammar_dirs(self.pack)
        self.assertTrue(live.exists(), "another server's files are not ours to remove")
        self.assertFalse(dead.exists())

    def test_the_pack_directory_is_swept_too(self):
        # Windows, and a Linux host whose /dev/shm has no room: the file is beside the model.
        d = self.made(self.pack, self.dead_pid())
        sweep_grammar_dirs(self.pack)
        self.assertFalse(d.exists())

    def test_a_name_that_is_not_a_pid_is_left_alone(self):
        d = self.made(self.shm, "not-a-pid")
        sweep_grammar_dirs(self.pack)
        self.assertTrue(d.exists(), "only this server's own naming scheme is swept")


class Cleanup(unittest.TestCase):
    """The request's `finally` releases every hold whatever happened, so no file and no hold outlives it (D-046)."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.pack = Path(self.dir.name)
        (self.pack / "vocab.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        self.dir.cleanup()

    def test_a_file_that_will_not_go_does_not_keep_the_next_one_held(self):
        a = hold_grammar_file(self.pack, "ab12", "start: entry_answer_plain")
        b = hold_grammar_file(self.pack, "cd34", "start: entry_answer_tools")
        real = Path.unlink

        def only_a_fails(path, *a_, **kw):
            if path.name == "ab12.gr":
                raise PermissionError("held open by something else")
            return real(path, *a_, **kw)

        with unittest.mock.patch.object(Path, "unlink", only_a_fails):
            drop_grammar(a)
            drop_grammar(b)                                      # the second drop still runs
        self.assertFalse(b.exists(), "a failure on one file must not skip the next file's drop")
        self.assertNotIn(str(a), server_module._grammar_holds, "a hold that could not be released is not kept")

    def test_a_request_that_fails_before_it_starts_leaves_no_file(self):
        # `begin_grammar` is inside the request's try, so a failure between writing the file and starting the first
        # pass still drops it.  The structure reader is the first thing after the file that can throw.
        svc, engine, tok = service(self.pack)
        ids = svc.prepare([{"role": "user", "content": "hi"}], None, {"enable_thinking": False})[0]
        before = set(self.pack.iterdir())
        with unittest.mock.patch.object(server_module, "StructureParser", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                list(svc.run(ids, False, None, 8, {}, threading.Event()))
        self.assertEqual(set(self.pack.iterdir()), before, "the file and its pid directory must both be gone")
        self.assertEqual(server_module._grammar_holds, {}, "and its hold released")


class Compiler(unittest.TestCase):
    """Which vocabulary a `Service` compiles a schema against (D-064).

    A service with a pack directory always gets a compiler; a pack-shaped directory with no tokenizer in it falls
    back to the committed fixture rather than to no compiler at all.  The compile cases skip without
    `libllguidance.so` (a `Vocabulary` cannot be built); the missing-library case is the one that must pass without
    it, because a server has to survive it.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def fixture_ok(self):
        try:
            grammar_matcher.vocabulary(FIXTURE_TOKENIZER)
        except grammar_matcher.LibraryMissing as e:
            raise unittest.SkipTest(f"needs libllguidance ({e})")

    def test_a_pack_shaped_directory_without_a_tokenizer_uses_the_fixture(self):
        self.fixture_ok()
        pack = Path(self.dir.name)
        (pack / "vocab.json").write_text("{}", encoding="utf-8")     # what `auto_pack_dir` leaves
        svc, _, _ = service(pack)
        self.assertEqual(svc.pipeline.compiler.dir, FIXTURE_TOKENIZER.resolve())

    def test_a_complete_tokenizer_directory_is_compiled_against_itself(self):
        self.fixture_ok()
        pack = Path(self.dir.name)
        for name in ("vocab.json", "token_type.json", "merges.txt"):
            (pack / name).write_bytes((FIXTURE_TOKENIZER / name).read_bytes())
        svc, _, _ = service(pack)
        self.assertEqual(svc.pipeline.compiler.dir, pack.resolve())

    def test_a_missing_library_turns_the_ladder_off_instead_of_crashing(self):
        # The one case that must not stop the server: an engine built without the grammar decoder refuses a chat
        # request later with its own 503, so the pipeline can exist without a compiler.
        with unittest.mock.patch.dict(grammar_matcher._VOCABULARIES, clear=True), \
                unittest.mock.patch.object(grammar_matcher, "Library",
                                           side_effect=grammar_matcher.LibraryMissing("no libllguidance.so")):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                svc, _, _ = service(server_module.auto_pack_dir())
        self.assertIsNone(svc.pipeline.compiler)
        self.assertIn("the compile ladder is off", out.getvalue())


if __name__ == "__main__":
    unittest.main()
