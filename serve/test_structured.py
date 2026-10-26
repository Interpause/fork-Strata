"""Exercise the JSON contract through HTTP, including H3's strict Story shape."""
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve import structured
from serve import server as server_module
from serve.server import ByteTokenizer, MockEngine, Service, serve


STORY = {"title": "Photobooth", "sheet": "A single continuous shot.", "cast": [],
         "boxes": [{"box": 1, "duration_ms": 15000, "new_shot": True, "beat": "Hailee changes poses.",
                    "on_screen": ["Hailee"], "references": [], "ends": "She smiles."}], "suggestions": []}


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


TEXT = {"type": "string"}
STORY_SCHEMA = object_schema({
    "title": TEXT, "sheet": TEXT, "cast": {"type": "array", "items": object_schema({"name": TEXT,
        "references": {"type": "array", "items": TEXT, "maxItems": 0}})},
    "boxes": {"type": "array", "minItems": 1, "maxItems": 1, "items": object_schema({
        "box": {"type": "integer", "minimum": 1, "maximum": 1},
        "duration_ms": {"type": "integer", "minimum": 1, "maximum": 15000}, "new_shot": {"type": "boolean"},
        "beat": TEXT, "on_screen": {"type": "array", "items": TEXT},
        "references": {"type": "array", "items": TEXT, "maxItems": 0}, "ends": TEXT})},
    "suggestions": {"type": "array", "items": object_schema({"what": TEXT, "why": TEXT,
        "boxes": {"type": "array", "items": {"type": "integer"}}, "description": TEXT})}})
FORMAT = {"type": "json_schema", "json_schema": {"name": "h3_story", "strict": True, "schema": STORY_SCHEMA}}


class Structured(unittest.TestCase):
    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read().decode()
            return response.status, data if "event-stream" in response.headers.get("Content-Type", "") else json.loads(data)

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, json.dumps(STORY), max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat(self, script=None, **kwargs):
        if script is not None:
            self.engine = self.svc.engine = MockEngine(self.tok, script, max_context=16384)
        req = {"messages": [{"role": "system", "content": "Write a film plan."},
                            {"role": "user", "content": "Photobooth. 15 seconds."}],
               "response_format": FORMAT, "reasoning_effort": "none", **kwargs}
        return self.request("/v1/chat/completions", req)

    def responses(self, **kwargs):
        req = {"model": "qwen3.8-flash-next", "input": "Photobooth. 15 seconds.",
               "reasoning": {"effort": "none"}, **kwargs}
        return self.request("/v1/responses", req)

    def test_the_responses_route_sends_its_schema_to_the_grammar(self):
        # D-058(a): the request is the sampling dict this route runs with, so `text.format`'s schema has to be
        # written onto it.  Without that, `begin_grammar` reads None, `resolve_entry` picks `entry_answer_plain`,
        # and the schema only judges the finished text - which then ends as a 502, not a constrained decode.
        grammars = []
        write = server_module.hold_grammar_file

        def spy(pack_dir, name, text):
            grammars.append(text)
            return write(pack_dir, name, text)

        with mock.patch.object(server_module, "hold_grammar_file", spy):
            code, reply = self.responses(text={"format": {"type": "json_schema", "name": "h3_story",
                                                          "strict": True, "schema": STORY_SCHEMA}})
        self.assertEqual(code, 200, reply)
        self.assertEqual(json.loads(reply["output"][-1]["content"][0]["text"]), STORY)
        self.assertTrue(grammars, "the route must ask the grammar for a reader")
        self.assertIn("start: entry_answer_json", grammars[0])

    def test_a_schema_with_tools_is_refused_on_both_routes(self):
        # The one message, on the route that used to refuse the pair and the one that used to send it anyway
        # (D-058(b)): `build()` drops the schema when tools are present, so that route only validated afterwards.
        chat = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
        responses = [{"type": "function", "name": "search", "parameters": {"type": "object"}}]
        with mock.patch.object(self.svc, "load") as load:
            code, reply = self.chat(tools=chat)
            self.assertEqual(code, 400)
            self.assertEqual(reply["error"]["message"], structured.FORMAT_WITH_TOOLS)
            code, reply = self.responses(tools=responses, text={"format": {"type": "json_schema", "name": "story",
                                                                          "schema": STORY_SCHEMA}})
            self.assertEqual(code, 400, reply)
            self.assertEqual(reply["error"]["message"], structured.FORMAT_WITH_TOOLS)
            self.assertEqual(reply["error"]["param"], "text.format")
            load.assert_not_called()
        # Without a schema the same tools are fine on the responses route: nothing to conflict with.
        self.assertEqual(self.responses(tools=responses)[0], 200)

    def test_a_number_json_cannot_write_is_named_on_both_routes(self):
        # `1e400` is a legal JSON number and Python reads it as `inf`; the grammar carries the schema as a JSON
        # payload, and `json.dumps` writes `Infinity` there, which the compiler rejects as a parse error about a
        # number the client wrote correctly (measured: `failed to parse "%json": expected value at line 1 column
        # 32`).  Named at the gate instead, on both routes, so the client is told which bound to fix.
        schema = {"type": "object", "properties": {"n": {"type": "number", "maximum": 1e400}}}
        expected = structured.NON_FINITE % ("/properties/n/maximum", float("inf"))
        with mock.patch.object(self.svc, "load") as load:
            code, reply = self.chat(response_format={"type": "json_schema",
                                                     "json_schema": {"name": "n", "schema": schema}})
            self.assertEqual(code, 400, reply)
            self.assertEqual(reply["error"]["message"], expected)
            code, reply = self.responses(text={"format": {"type": "json_schema", "name": "n", "schema": schema}})
            self.assertEqual(code, 400, reply)
            # The Responses route names the parameter its client actually sent (`_responses_prepare` rewrites the
            # chat route's name), so the message is the same one with that one word changed.
            self.assertEqual(reply["error"]["message"], expected.replace("response_format", "text.format"))
            self.assertEqual(reply["error"]["param"], "text.format")
            load.assert_not_called()
        # A bound JSON *can* write is a different matter from one it cannot: 1e308 is finite, so the gate passes it,
        # and the compiler then refuses the float range it would have to generate.  Measured on this build -
        # `Failed to generate regex for float range: min=None, max=Some(1e308)` - and left as a terminal: which of
        # two bounds to drop is a guess, so the client gets the compiler's words instead (D-065).
        code, reply = self.chat(response_format={
            "type": "json_schema",
            "json_schema": {"name": "n", "schema": {"type": "object", "properties": {
                "n": {"type": "number", "maximum": 1e308}}}}})
        self.assertEqual(code, 400, reply)
        self.assertIn("float range", reply["error"]["message"])
        # An everyday bound compiles, so the refusal above is about the extreme one and not about `maximum`.
        self.assertEqual(self.chat(response_format={
            "type": "json_schema",
            "json_schema": {"name": "n", "schema": {"type": "object", "properties": {
                "n": {"type": "number", "maximum": 15000}}}}})[0], 200)

    def test_h3_schema_success_and_stream_with_reasoning(self):
        code, reply = self.chat()
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(reply["choices"][0]["message"]["content"]), STORY)
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertIn("OUTPUT FORMAT REQUIREMENT", prompt)
        self.assertIn('"duration_ms"', prompt)
        # the template writes exactly two newlines after the </think> token (skeleton.lark's `thinking` rule)
        code, raw = self.chat("Consider the shot.\n</think>\n\n" + json.dumps(STORY), stream=True, reasoning_effort="low")
        self.assertEqual(code, 200)
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        self.assertEqual(json.loads(content), STORY)
        self.assertIn("reasoning_content", raw)
        self.assertIn("data: [DONE]", raw)

    def test_invalid_generation_is_an_error_and_never_streams_content(self):
        for script in ("## Film Plan - Photobooth", '{"title":"Only a title"}',
                       json.dumps({**STORY, "extra": True}),
                       json.dumps({**STORY, "boxes": [{**STORY["boxes"][0], "duration_ms": "15000"}]})):
            with self.subTest(script=script):
                code, reply = self.chat(script)
                self.assertEqual(code, 502)
                self.assertEqual(reply["error"]["code"], "structured_output_failed")
                _, raw = self.chat(script, stream=True)
                self.assertIn('"structured_output_failed"', raw)
                self.assertNotIn('"delta"', raw)
                self.assertIn("data: [DONE]", raw)

    def test_object_mode_strict_json_and_truncation(self):
        fmt = {"type": "json_object"}
        self.assertEqual(self.chat('{"answer":4}', response_format=fmt)[0], 200)
        for script in ('[]', '{"x":NaN}', '{"x":1,"x":2}', '{"x":1e999}', 'no json at all', '{"x":'):
            self.assertEqual(self.chat(script, response_format=fmt)[0], 502)
        # #762: the JSON object is taken out of a code fence or the prose around it (the object itself stays strict)
        for script in ('```json\n{}\n```', 'Here it is: {"answer":4} - done'):
            self.assertEqual(self.chat(script, response_format=fmt)[0], 200, script)
        self.assertEqual(self.chat('{"answer":4}', response_format=fmt, max_tokens=5)[0], 502)
        # Plain text callers retain normal streaming, without a JSON validation gate.
        self.assertEqual(self.chat("Prose.", response_format={"type": "text"})[0], 200)

    def test_rejected_schema_does_not_load_the_engine_or_retry(self):
        with mock.patch.object(self.svc, "load") as load:
            code, _ = self.chat(response_format={"type": "json_schema"})
            self.assertEqual(code, 400)
            load.assert_not_called()
        self.engine = self.svc.engine = MockEngine(self.tok, "Markdown is not JSON", max_context=16384)
        with mock.patch.object(self.engine, "generate", wraps=self.engine.generate) as generate:
            self.assertEqual(self.chat()[0], 502)
            self.assertEqual(generate.call_count, 1)

    def test_invalid_schema_is_400_and_local_refs_work(self):
        for schema in ({"type": "object", "properties": {"x": {"type": "invalid"}}},
                       {"type": "object", "$ref": "https://other.example/schema"}):
            fmt = {"type": "json_schema", "json_schema": {"name": "test", "strict": True, "schema": schema}}
            self.assertEqual(self.chat(response_format=fmt)[0], 400)
        schema = object_schema({"answer": {"$ref": "#/$defs/n"}})
        schema["$defs"] = {"n": {"type": "integer"}}
        fmt = {"type": "json_schema", "json_schema": {"name": "local", "schema": schema}}
        self.assertEqual(self.chat('{"answer":4}', response_format=fmt)[0], 200)

    def test_malformed_formats_and_tools_are_rejected_before_loading(self):
        formats = ([], {"type": "unsupported"},
                   {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
                   {"type": "json_schema", "json_schema": {"name": "x", "strict": "true", "schema": {"type": "object"}}})
        with mock.patch.object(self.svc, "load") as load:
            for fmt in formats:
                with self.subTest(fmt=fmt):
                    self.assertEqual(self.chat(response_format=fmt)[0], 400)
            self.assertEqual(self.chat(tools=[{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}])[0], 400)
            self.assertEqual(self.chat(strata_mcp=True)[0], 400)
            self.assertEqual(self.request("/v1/chat/completions", [])[0], 400)
            load.assert_not_called()

    def test_a_pattern_only_the_engine_can_judge_is_not_a_false_400(self):
        # `\p{L}+` is a valid `regex-syntax` pattern for the grammar, but CPython's `re` refuses it (`bad escape
        # \p`): the gate's format assertion is off, so a schema the engine compiles is never a 400 here.
        schema = object_schema({"word": {"type": "string", "pattern": "\\p{L}+"}})
        fmt = {"type": "json_schema", "json_schema": {"name": "letters", "schema": schema}}
        messages, validator = structured.prepare_format(fmt, [{"role": "user", "content": "a word"}])
        self.assertIsNotNone(validator)
        self.assertIn("\\p{L}+", messages[0]["content"])

    def test_a_malformed_pattern_reaches_the_compile_ladder_not_the_gate(self):
        # `(` is not a schema-validity error with the format assertion off, so it is not a dead-end 400: it reaches
        # the compile ladder, which classifies and repairs it (P7 item 5).
        schema = object_schema({"word": {"type": "string", "pattern": "("}})
        fmt = {"type": "json_schema", "json_schema": {"name": "broken", "schema": schema}}
        structured.prepare_format(fmt, [{"role": "user", "content": "a word"}])   # does not raise; the ladder owns it

    def test_the_validity_gate_still_refuses_what_is_not_a_schema(self):
        # The gate keeps its real job. A non-object root is stopped by format_schema's root rule; an unknown `type`
        # inside an object, and a `required` that is not a list, are stopped by jsonschema itself.
        cases = (({"type": "banana"}, "only JSON objects"),
                 ({"type": "object", "required": "a"}, "invalid response_format schema:"),
                 ({"type": "object", "properties": {"a": {"type": "banana"}}}, "invalid response_format schema:"))
        for schema, message in cases:
            fmt = {"type": "json_schema", "json_schema": {"name": "bad", "schema": schema}}
            with self.subTest(schema=schema):
                with self.assertRaises(ValueError) as caught:
                    structured.prepare_format(fmt, [{"role": "user", "content": "hi"}])
                self.assertIn(message, str(caught.exception))

    def test_root_unions_of_object_shapes_are_accepted(self):
        # A root anyOf of object shapes (sent by apps written against llama.cpp's grammar path) still means
        # "one JSON object", so it is accepted; a non-object answer is still refused.
        claim = {"type": "object", "additionalProperties": False, "required": ["claim_text"],
                 "properties": {"claim_text": {"type": "string", "minLength": 1}}}
        empty = {"type": "object", "additionalProperties": False, "required": ["outcome"],
                 "properties": {"outcome": {"type": "string", "enum": ["no_substantive_content"]}}}
        schemas = [
            {"anyOf": [claim, empty]},
            {"oneOf": [claim, empty]},
            {"allOf": [claim, {"required": ["claim_text"]}]},
            {"$ref": "#/$defs/claim", "$defs": {"claim": claim}},
            {"anyOf": [{"$ref": "#/$defs/claim"}, {"anyOf": [empty]}], "$defs": {"claim": claim}},
        ]
        for schema in schemas:
            fmt = {"type": "json_schema", "json_schema": {"name": "union", "strict": True, "schema": schema}}
            with self.subTest(schema=schema):
                self.assertEqual(self.chat('{"claim_text":"The term is 12 months."}', response_format=fmt)[0], 200)
                self.assertEqual(self.chat("[1, 2]", response_format=fmt)[0], 502)

    def test_schemas_that_allow_non_objects_are_still_rejected(self):
        obj = {"type": "object"}
        schemas = [
            {"type": "array", "items": obj},
            {"type": "string"},
            {"type": ["object", "null"]},
            {"anyOf": [obj, {"type": "string"}]},
            {"anyOf": []},
            {"oneOf": [obj, {"type": "array"}]},
            {"allOf": [{"type": "string"}]},
            {"properties": {"x": {"type": "string"}}},           # no type: any JSON value validates
            {"$ref": "#/$defs/a", "$defs": {"a": {"$ref": "#/$defs/a"}}},   # reference cycle
            {"$ref": "#/$defs/missing", "$defs": {}},
            {"$ref": "#/$defs/s", "$defs": {"s": {"type": "string"}}},
        ]
        with mock.patch.object(self.svc, "load") as load:
            for schema in schemas:
                fmt = {"type": "json_schema", "json_schema": {"name": "bad", "schema": schema}}
                with self.subTest(schema=schema):
                    code, reply = self.chat(response_format=fmt)
                    self.assertEqual(code, 400)
                    self.assertIn("only JSON objects", reply["error"]["message"])
            load.assert_not_called()

    def test_the_server_does_not_import_jsonschema_at_start(self):
        import subprocess
        import sys
        code = ("import sys; import serve.server; "
                "sys.exit(1 if any(m == 'jsonschema' or m.startswith('jsonschema.') for m in sys.modules) else 0)")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(Path(__file__).resolve().parents[1]))
        self.assertEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
