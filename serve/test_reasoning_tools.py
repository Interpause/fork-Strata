"""serve/test_reasoning_tools.py - a `<tool_call>` written inside the reasoning is reasoning text, never a call.

`StructureParser` reports a call only from the grammar's delimiter tokens, so a `<tool_call>` spelled in the thinking
block reaches the client as the text it is.  `OverHttp` drives that reader over both APIs' HTTP endpoints.

    python -m unittest serve.test_reasoning_tools -v
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
from serve.test_server import CTX  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PROPS = {"path": {"type": "string"}}
# The template's own shape: reasoning, the answer, then exactly two newlines and the call.  `answer_tools` in
# skeleton.lark has no production with prose after a call, so the answer has to come before it.
SCRIPT = ("Let me look.\n</think>\n\nChecking.\n\n<tool_call>\n<function=write>\n<parameter=path>\n"
          "file.txt\n</parameter>\n</function>\n</tool_call>")


class OverHttp(unittest.TestCase):
    """The structured route over HTTP: `Service.run` builds `StructureParser` from the token ids, and the two
    endpoint layers render that one `Event` stream."""

    def setUp(self):
        self.pack = tempfile.mkdtemp()      # sweep_grammar_dirs runs on Service.__init__: never reuse a directory
        tok = ByteTokenizer()
        engine = MockEngine(tok, SCRIPT, max_context=CTX)
        engine.info = {"grammar": 1}        # grammar_ready(): the engine compiles the file the server writes for it
        self.httpd = serve(Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"),
                                   pack_dir=self.pack), port=0)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        shutil.rmtree(self.pack, ignore_errors=True)

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode()

    def test_both_apis_render_one_event_stream_with_each_block_in_order(self):
        raw = self.post("/v1/messages", {
            "model": "x", "max_tokens": 500, "stream": True, "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "write", "input_schema": {"type": "object", "properties": PROPS}}],
        }, {"anthropic-version": "2023-06-01"})
        evs = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        # the reasoning, the answer, the call: one stream, so the endpoint can only report them in this order
        self.assertEqual([e["content_block"]["type"] for e in evs if e["type"] == "content_block_start"],
                         ["thinking", "text", "tool_use"])
        args = "".join(e["delta"]["partial_json"] for e in evs
                       if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta")
        self.assertEqual(json.loads(args), {"path": "file.txt"})
        self.assertEqual([e["delta"]["stop_reason"] for e in evs if e["type"] == "message_delta"], ["tool_use"])

        choice = json.loads(self.post("/v1/chat/completions", {
            "model": "x", "max_tokens": 500, "stream": False, "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "write",
                      "parameters": {"type": "object", "properties": PROPS}}}],
        }))["choices"][0]
        calls = choice["message"]["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "write")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "file.txt"})
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["content"], "Checking.")
        self.assertEqual(choice["message"]["reasoning_content"], "Let me look.\n")


if __name__ == "__main__":
    unittest.main()
