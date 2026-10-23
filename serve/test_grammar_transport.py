"""The transport: one `grammar=<path>` key on the engine line, one file that does not outlive its request.

    python -m unittest serve.test_grammar_transport -v

Host-only: the request is driven through `Service.run` with the mock engine, the same way the server tests do.  The
server compiles nothing, so what these tests are about is the file alone - written, held, adopted, removed - and they
need no `tools/build_llguidance.sh`, which is also the state a machine without it is in: the file and the key still
travel, nothing is constrained (D-020).
"""
import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import FORCE_ANY, ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine  # noqa: E402
from serve.server import drop_grammar, grammar_file, hold_grammar, hold_grammar_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096


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


def service(pack_dir=None):
    tok = ByteTokenizer()
    engine = Recorder(tok, "</think>\n\nHello.", max_context=CTX)
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), pack_dir=pack_dir)
    return svc, engine, tok


def run(svc, tok, thinking=False, tools=None, force=None, sampling=None):
    ids = svc.prepare([{"role": "user", "content": "hi"}], tools, {"enable_thinking": thinking},
                      req=sampling if sampling is not None else {}, force=force)[0]
    cancel = threading.Event()
    return list(svc.run(ids, thinking, tools, 8, dict(sampling or {}), cancel, force=force))


class Key(unittest.TestCase):
    def test_a_grammar_is_one_key_before_the_ids(self):
        self.assertEqual(StrataEngine.sampling_keys({"strata_grammar": "/dev/shm/g/ab12.gr"}),
                         " grammar=/dev/shm/g/ab12.gr")

    def test_no_grammar_is_no_key(self):
        for sampling in ({}, {"strata_grammar": None}, {"strata_grammar": ""}, {"strata_grammar": 7}):
            with self.subTest(sampling=sampling):
                self.assertEqual(StrataEngine.sampling_keys(sampling), "")

    def test_a_path_with_a_space_is_not_sent(self):
        # A space ends a key on this line: sending it would turn the rest of the path into the ids.
        self.assertEqual(StrataEngine.sampling_keys({"strata_grammar": "/a b/c.gr"}), "")

    def test_the_key_sits_beside_the_other_keys(self):
        keys = StrataEngine.sampling_keys({"temperature": 0.5, "strata_grammar": "/x/y.gr"})
        self.assertEqual(keys, " temperature=0.5 grammar=/x/y.gr")


class Capability(unittest.TestCase):
    """An engine that cannot compile a grammar must not be sent one: it refuses the whole request (C1's engine
    answers `ERR grammar: no grammar tokenizer was loaded ...`), and a killed request is worse than the
    unconstrained reply this phase already gives.  `INFO grammar=1` is how the engine says it can."""

    def setUp(self):
        StrataEngine._no_grammar_said = False
        self.eng = object.__new__(StrataEngine)          # keys() reads only info
        self.eng.info = {}

    def test_an_engine_with_grammar_gets_the_key(self):
        self.eng.info = {"grammar": 1}
        self.assertEqual(self.eng.keys({"strata_grammar": "/dev/shm/g/ab12.gr"}), " grammar=/dev/shm/g/ab12.gr")

    def test_an_engine_without_it_gets_the_other_keys_and_no_grammar(self):
        keys = self.eng.keys({"temperature": 0.5, "strata_grammar": "/dev/shm/g/ab12.gr"})
        self.assertEqual(keys, " temperature=0.5")

    def test_the_drop_is_said_once(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.eng.keys({"strata_grammar": "/a.gr"})
            self.eng.keys({"strata_grammar": "/b.gr"})
        self.assertEqual(out.getvalue().count("decodes unconstrained"), 1)
        self.assertIn("no grammar=1", out.getvalue())

    def test_a_request_without_a_grammar_says_nothing(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(self.eng.keys({"temperature": 0.5}), " temperature=0.5")
        self.assertEqual(out.getvalue(), "")


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
        # first one to finish must not delete what the second has yet to read - the serialised request of D-029 is
        # admitted exactly then.
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
        run(self.svc, self.tok, thinking=True, tools=tools, force=FORCE_ANY)
        self.assertIn("start: entry_think_tools_required", self.engine.calls[0]["text"])

    def test_a_call_the_prompt_already_opened_is_the_mid_call_entry(self):
        # Without thinking the forced opening ends the prompt, so the reply starts inside the call.
        tools = [{"name": "f", "parameters": {"type": "object", "properties": {}}}]   # as the frontend passes them
        run(self.svc, self.tok, thinking=False, tools=tools, force=FORCE_ANY)
        self.assertIn("start: entry_call_name", self.engine.calls[0]["text"])

    def test_a_prompt_without_a_template_tail_sends_no_grammar(self):
        # A completion-style prompt: the grammar has no entry for it, so nothing is sent and the reply is unchanged.
        list(self.svc.run(self.tok.encode("plain prompt, no chat template"), False, None, 8, {},
                          threading.Event()))
        self.assertIsNone(self.engine.calls[0]["path"])
        self.assertTrue(self.svc.grammar_notes)

    def test_a_server_without_a_pack_carries_no_grammar(self):
        svc, engine, tok = service(None)
        self.assertIsNone(svc.pipeline)
        run(svc, tok)
        self.assertIsNone(engine.calls[0]["path"])

    def test_the_json_entry_uses_the_requests_schema(self):
        sampling = {"response_format": {"type": "json_schema", "json_schema": {
            "name": "answer", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}}}
        run(self.svc, self.tok, sampling=sampling)
        text = self.engine.calls[0]["text"]
        self.assertIn("start: entry_answer_json", text)
        self.assertIn('"a"', text)

    def test_a_broken_schema_sends_no_grammar(self):
        sampling = {"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": []}}}
        run(self.svc, self.tok, sampling=sampling)
        self.assertIsNone(self.engine.calls[0]["path"])


class Routing(unittest.TestCase):
    """TEMPORARY(D-029) - P4 deletes this class with the routing.

    A request whose `grammar=` will actually travel decodes on the solo path even when the engine runs batch slots,
    because the engine refuses a `grammar=` on a batch admission: its batch windows sample through `sample_rows`,
    which has no bias path.  P4 gives the batch path the per-slot mask, and then a grammar request runs in a slot
    like any other and these tests go away - `solo_only` returning False everywhere is the same assertion as the
    routing being gone."""

    @staticmethod
    def engine(batch, info):
        eng = object.__new__(StrataEngine)               # solo_only reads batch and info only
        eng.batch, eng.info = batch, info
        return eng

    def test_a_grammar_request_is_solo_on_a_batching_engine(self):
        self.assertTrue(self.engine(3, {"grammar": 1}).solo_only({"strata_grammar": "/dev/shm/g/ab12.gr"}))

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


if __name__ == "__main__":
    unittest.main()
