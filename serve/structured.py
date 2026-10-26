"""JSON response formats at the HTTP boundary: prompt once, validate before delivery.

`jsonschema` is a dependency (requirements.txt), so a declared `json_schema` is always checked against its
meta-schema before the engine loads: the validity gate is the first rung, and the compile ladder
(serve/grammar/ladder.py) starts after it.

The gate's format assertion is off (`format_checker=None`). Its default asserts the meta-schema's
`format: regex` on every `pattern` with CPython's `re`, which is neither JSON Schema's ECMA-262 nor the engine's
`regex-syntax`: `\\p{L}+` compiles in the grammar but Python raises `bad escape \\p`, a false 400. The engine, not
Python, is the authority on a `pattern`; a malformed `(` is therefore not a validity error and reaches the ladder.

`json_object` needs no schema library - "one JSON object" is the whole contract there - so it keeps `_ObjectOnly`,
a standard-library validator.
"""
import json
import threading


class StructuredOutputError(RuntimeError):
    pass


class _ObjectOnly:
    """The validator for `json_object`: one JSON object is the whole contract there, so no schema library is needed.

    `format_schema` deliberately routes `json_object` to the literal `{"type": "object"}`; this is what validates
    that reply, not a stand-in for a missing `jsonschema`.
    """

    def iter_errors(self, value):
        if not isinstance(value, dict):
            yield _ObjectError()


class _ObjectError:
    absolute_path = ()
    message = "the answer is not a JSON object"


_jsonschema = None                     # (validators, SchemaError, Registry, NoSuchResource), imported once
_jsonschema_lock = threading.Lock()


def jsonschema_modules():
    """`jsonschema` and its helpers (a dependency of every install), imported on first use.

    Imported lazily so a server start does not pay for it; the validity gate calls this when a request declares a
    `json_schema`.
    """
    global _jsonschema
    with _jsonschema_lock:
        if _jsonschema is None:
            from jsonschema import validators
            from jsonschema.exceptions import SchemaError
            from referencing import Registry
            from referencing.exceptions import NoSuchResource
            _jsonschema = (validators, SchemaError, Registry, NoSuchResource)
        return _jsonschema


def _only_objects(node, root, refs=()):
    """True when every value `node` accepts is a JSON object, so "return one JSON object" stays true.

    `type: object`, an anyOf/oneOf whose branches all qualify (e.g. a root union of object shapes, which llama.cpp's
    grammar path accepts and apps send), an allOf with a qualifying member, or a local `$ref` to one of these.
    Anything that can also be an array, string, number, boolean or null (including `type: ["object", "null"]`) is
    not, and a `$ref` cycle never qualifies.
    """
    if not isinstance(node, dict):
        return False
    if node.get("type") == "object":
        return True
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#") and ref not in refs:
        target = root
        for part in ref[1:].split("/")[1:]:
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                return False
            target = target[part]
        return _only_objects(target, root, refs + (ref,))
    for key in ("anyOf", "oneOf"):
        branches = node.get(key)
        if isinstance(branches, list) and branches and all(_only_objects(b, root, refs) for b in branches):
            return True
    branches = node.get("allOf")
    return isinstance(branches, list) and any(_only_objects(b, root, refs) for b in branches)


def format_schema(response_format):
    """The JSON Schema a response_format asks the reply to match, or None.  Raises ValueError on a malformed one.

    One place reads response_format's shapes, because two things depend on the answer: the directive added to the
    prompt (prepare_format) and the grammar's JSON entry (serve/grammar/pipeline.py).
    """
    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")
    kind = response_format.get("type")
    if kind == "text":
        return None
    if kind == "json_object":
        return {"type": "object"}     # the standard library is enough for "one JSON object"
    if kind != "json_schema":
        raise ValueError("response_format.type must be text, json_object or json_schema")
    spec = response_format.get("json_schema")
    if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
        raise ValueError("response_format.json_schema needs a schema object")
    if not isinstance(spec.get("name"), str) or not spec["name"]:
        raise ValueError("response_format.json_schema needs a name")
    if "strict" in spec and not isinstance(spec["strict"], bool):
        raise ValueError("response_format.json_schema.strict must be boolean")
    schema = spec["schema"]
    if not _only_objects(schema, schema):
        raise ValueError("response_format schema must accept only JSON objects at its root "
                         "(type object, or anyOf/oneOf of object schemas)")

    def check_refs(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("$ref", "$dynamicRef") and isinstance(value, str) and not value.startswith("#"):
                    raise ValueError("response_format supports only local schema references (#...)")
                check_refs(value)
        elif isinstance(node, list):
            for value in node:
                check_refs(value)
    check_refs(schema)
    return schema


FORMAT_WITH_TOOLS = ("a response schema with tools is not supported: a reply is one JSON object or a tool call, "
                     "never both, and the grammar cannot express the two - send the schema without tools, or drop "
                     "the schema")


def prepare_format(response_format, messages):
    schema = format_schema(response_format)
    if schema is None:
        return messages, None
    if response_format.get("type") == "json_object":
        validator = _ObjectOnly()          # format_schema sent {"type": "object"}: one JSON object is the contract
    else:
        validators, SchemaError, Registry, NoSuchResource = jsonschema_modules()

        def no_remote(uri):
            raise NoSuchResource(ref=uri)

        try:
            cls = validators.validator_for(schema)
            # format_checker=None on purpose: the default checker asserts the meta-schema's `format: regex` on
            # every `pattern` with CPython's `re`, which is neither JSON Schema's ECMA-262 nor the engine's
            # `regex-syntax`. Measured false 400: `\p{L}+` is a valid pattern for llguidance but Python raises
            # `bad escape \p`. The engine is the authority on `pattern`, so a malformed `(` is not a validity
            # error here: it reaches the compile ladder (serve/grammar/ladder.py), which owns it.
            cls.check_schema(schema, format_checker=None)
            validator = cls(schema, registry=Registry(retrieve=no_remote))
        except SchemaError as exc:
            raise ValueError(f"invalid response_format schema: {exc.message}") from exc
    directive = ("OUTPUT FORMAT REQUIREMENT: Return exactly one JSON object matching the JSON Schema below. "
                 "No Markdown, headings, code fences, commentary, or text outside the JSON. "
                 "Use every required field, correct types, and only allowed fields. "
                 "Put all requested writing inside the appropriate JSON string fields.\nJSON Schema:\n" +
                 json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    messages = [dict(message) for message in messages]
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content") or ""
        if isinstance(content, list):
            messages[0]["content"] = content + [{"type": "text", "text": directive}]
        else:
            messages[0]["content"] = content + "\n\n" + directive
    else:
        messages.insert(0, {"role": "system", "content": directive})
    return messages, validator


def _extract_json(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    if s.startswith("```"):
        first = s.find("\n")
        if first != -1:
            end = s.find("```", first + 1)
            if end != -1:
                s = s[first + 1:end].strip()
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
        else:
            if ch == "\"":
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return s[start:]


def validated_json(text, validator, finish):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj

    def constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    if finish != "stop":
        raise StructuredOutputError(f"structured output was incomplete (finish_reason={finish}); increase the output budget")
    try:
        value = json.loads(_extract_json(text), object_pairs_hook=pairs, parse_constant=constant)
        canonical = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError) as exc:
        raise StructuredOutputError(f"model did not return valid JSON: {exc}") from exc
    try:
        error = next(validator.iter_errors(value), None)
    except Exception as exc:
        raise StructuredOutputError(f"could not validate structured output: {exc}") from exc
    if error is not None:
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        raise StructuredOutputError(f"model output failed the JSON Schema at {path}: {error.message}")
    return canonical
