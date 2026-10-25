"""The transport: one `grammar=<path>` key on the engine line, one file that does not outlive its request.

    python -m unittest serve.test_grammar_transport -v

Host-only: the request is driven through `Service.run` with the mock engine, the same way the server tests do.  The
server compiles nothing, so what these tests are about is the file alone - written, held, adopted, removed - and they
need no `tools/build_llguidance.sh`, which is also the state a machine without it is in: the file and the key still
travel, nothing is constrained (D-020).
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server_module  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine  # noqa: E402
from serve.server import drop_grammar, grammar_file, hold_grammar, hold_grammar_file  # noqa: E402
from serve.server import sweep_grammar_dirs  # noqa: E402

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


def service(pack_dir=None, reply="</think>\n\nHello."):
    tok = ByteTokenizer()
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
        want = base64.b64encode(b"/dev/shm/g/ab12.gr").decode("ascii")
        self.assertEqual(self.eng.keys({"strata_grammar": "/dev/shm/g/ab12.gr"}), " grammar=" + want)

    def test_an_engine_without_it_gets_the_other_keys_and_no_grammar(self):
        keys = self.eng.keys({"temperature": 0.5, "strata_grammar": "/dev/shm/g/ab12.gr"})
        self.assertEqual(keys, " temperature=0.5")

    def test_the_drop_is_said_once(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.eng.keys({"strata_grammar": "/a.gr"})
            self.eng.keys({"strata_grammar": "/b.gr"})
        self.assertEqual(out.getvalue().count("[strata] WARNING:"), 1)
        self.assertIn("no grammar=1", out.getvalue())
        self.assertIn("Unsafe", out.getvalue())

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
        run(self.svc, self.tok, thinking=True, tools=tools, force=("required", None))
        self.assertIn("start: entry_think_tools_required", self.engine.calls[0]["text"])

    def test_a_named_choice_offers_only_that_tool(self):
        """A named `tool_choice` narrows the fragment: `tool_dispatch` would otherwise offer every declared tool."""
        tools = [{"name": n, "parameters": {"type": "object", "properties": {}}} for n in ("write", "read")]
        # the fragment's rules carry each tool's index: call_t0_write is "write", call_t1_read is "read"
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

    def test_the_unconstrained_line_is_said_once_and_calls_the_reply_unsafe(self):
        # D-050: one line per process, and it says the text reader is unprotected - not merely "unconstrained".
        Service._unconstrained_said = False
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.svc.grammar_note("first reason")
            self.svc.grammar_note("second reason")
        said = out.getvalue()
        self.assertEqual(said.count("[strata] WARNING:"), 1)
        self.assertIn("first reason", said)
        self.assertIn("Unsafe", said)
        self.assertIn("D-050", said)
        self.assertEqual(self.svc.grammar_notes, {"first reason", "second reason"})

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


if __name__ == "__main__":
    unittest.main()
