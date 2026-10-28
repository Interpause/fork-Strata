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
import math
import threading


class StructuredOutputError(RuntimeError):
    pass


#: A schema value JSON itself cannot carry.  `json.loads("1e400")` is `inf` in Python, and the grammar text is a
#: JSON payload, so `json.dumps` would write a bare `Infinity` there and the compiler would report a parse error
#: about a number the client wrote legally (measured: `failed to parse "%json": expected value at line 1 column
#: 32`).  Naming the value is the difference between a client fixing its bound and a client reading a parse error.
NON_FINITE = ("response_format schema: the %s value (%r) is a number JSON cannot write, so it cannot be carried "
              "into the reply's grammar - send a finite bound")

#: A schema nested deeper than Python can walk.  `check_schema` and the validator built from it recurse once per
#: level, as does `format_schema`'s hand-written `check_refs` walk (measured on this build: 98 nested `properties`
#: for the gate, 499 for the walk).  The engine's own parse limit is lower still (64 nested `properties`, where
#: serde_json's 128-level limit fires before llguidance sees the payload), so a schema that trips this could not
#: compile anyway.  The depth Python fails at depends on the stack, and `RecursionError` carries no depth, so the
#: message states the engine's ceiling rather than a number the client could aim at.
TOO_DEEP = ("response_format schema nests too deeply for Python to check: the schema is nested past the "
            "interpreter's recursion limit, and the engine's own ceiling is lower still (64 nested properties), "
            "so it cannot be compiled - send a flatter schema")

#: The most members one schema may hold.  The validity gate's cost is per member, not per byte: `jsonschema`'s
#: `check_schema` takes ~184 µs per `properties` entry and ~380 µs per `anyOf` branch, linear from 1,000 to 20,000
#: (tmp/research/check-schema-report.md §3).  At the server's 256 MiB body ceiling that is hours of CPU for one
#: request.  No real schema needs this many: a tool schema generated for a model is a few hundred members at most.
#: Measured on the development PC (AMD Ryzen 9 5950X, CPython 3.14.8, jsonschema 4.26.0): `prepare_format` takes
#: 424 ms for 2,000 properties and 788 ms for 2,000 `anyOf` branches, so the worst case at this ceiling is under
#: a second.
MEMBER_CEILING = 2000

#: A schema whose member count is past `MEMBER_CEILING`.  The count is what the walks pay for, so the message names
#: the units, the ceiling, and the fix.  It names `response_format` as the other messages do, which is also what the
#: Responses route renames to `text.format`.
TOO_MANY_MEMBERS = (
    "response_format schema has more than %d members, the server's ceiling; the check costs per member (one each "
    "for a property, an anyOf/oneOf/allOf branch, an items schema, a patternProperties entry and a "
    "$defs/definitions entry) - send a smaller schema" % MEMBER_CEILING)

_MEMBER_KEYS = ("properties", "patternProperties", "$defs", "definitions")
_COMBINATION_KEYS = ("anyOf", "oneOf", "allOf")
#: Keywords whose value is data, not schema: a `title` or an `enum` of 50,000 strings holds no members, and the
#: measured cost of a 989 KB `enum` is 12.9 ms - so the walk must not spend itself on it.
_DATA_KEYS = ("enum", "const", "default", "examples")


def _over_member_ceiling(schema):
    """True when `schema` holds more than `MEMBER_CEILING` members.

    An explicit stack, never recursion: a `RecursionError` raised here would be caught by `prepare_format` and
    reported as `TOO_DEEP`, blaming a schema that is wide, not deep.  The walk stops at the first member past the
    ceiling, so a refused schema is not walked to its end either.
    """
    count = 0
    stack = [schema]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _MEMBER_KEYS and isinstance(value, dict):
                    count += len(value)
                    if count > MEMBER_CEILING:
                        return True
                    stack.extend(value.values())
                elif key in _COMBINATION_KEYS and isinstance(value, list):
                    count += len(value)
                    if count > MEMBER_CEILING:
                        return True
                    stack.extend(value)
                elif key == "items":
                    if isinstance(value, dict):
                        count += 1
                        if count > MEMBER_CEILING:
                            return True
                        stack.append(value)
                    elif isinstance(value, list):
                        count += len(value)
                        if count > MEMBER_CEILING:
                            return True
                        stack.extend(value)
                elif key not in _DATA_KEYS and isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return False


def _non_finite(value, path=""):
    """The first `(pointer, value)` in a schema that is infinite or NaN, or None."""
    stack = [(path, value)]
    while stack:
        where, node = stack.pop()
        if isinstance(node, dict):
            stack.extend((f"{where}/{key}", item) for key, item in node.items())
        elif isinstance(node, list):
            stack.extend((f"{where}/{index}", item) for index, item in enumerate(node))
        elif isinstance(node, float) and not math.isfinite(node):
            return where or "/", node
    return None


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
    # One catch covers every recursive walk a declared schema passes through - `format_schema`'s `check_refs` and
    # `_only_objects`, then `check_schema` and the validator built from it - because they all run inside this
    # function, and the first call is `format_schema` itself.  Catching beside `SchemaError` alone would let a
    # deeper `check_refs` or `_only_objects` escape.
    try:
        # Before `format_schema`: its `_only_objects` and `check_refs` walks, and then `check_schema`, all pay per
        # member, and one body at the server's 256 MiB ceiling can hold hours of them.  `json_object` ignores any
        # `json_schema` beside it, so only a declared `json_schema` is counted.
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            spec = response_format.get("json_schema")
            declared = spec.get("schema") if isinstance(spec, dict) else None
            if isinstance(declared, dict) and _over_member_ceiling(declared):
                raise ValueError(TOO_MANY_MEMBERS)
        schema = format_schema(response_format)
        if schema is None:
            return messages, None
        bad = _non_finite(schema)
        if bad is not None:
            raise ValueError(NON_FINITE % bad)
        if response_format.get("type") == "json_object":
            validator = _ObjectOnly()      # format_schema sent {"type": "object"}: one JSON object is the contract
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
    except RecursionError as exc:
        raise ValueError(TOO_DEEP) from exc
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


def validated_json(text, validator, finish, relaxed=None, notice=None):
    """The reply's original JSON text, or `StructuredOutputError`.

    `text` is parsed as it arrived: a schema reply is one `%json` value produced under the grammar, so the text that
    arrives *is* the JSON - there is no fence to strip and no first brace to find (D-068).  `relaxed` is an optional
    zero-argument callable saying whether that grammar was repaired to compile; when it was and the reply still misses
    the original schema, the mismatch is one `[strata]` warning and the answer is delivered, because the relaxation
    was the server's choice and failing the reply would blame the client or the model for it (D-068).  A schema that
    compiled as written keeps its 502, and so does every broken reply: truncation, invalid JSON, duplicate keys,
    a non-finite number and an uncanonicalisable value are not repairs.

    `notice`: an optional dict this fills with `{"at": pointer, "error": why}` when, and only when, it delivers such a
    reply.  The log line is read by whoever runs the server; the notice is how the *client* finds out that the JSON it
    is about to use was checked against a wider schema than the one it sent (D-071) - the shape is settled on the
    server, and the client is the one that can act on the difference.
    """
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj

    if finish != "stop":
        raise StructuredOutputError(f"structured output was incomplete (finish_reason={finish}); increase the output budget")
    try:
        value = json.loads(text, object_pairs_hook=pairs)
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))  # check it can be represented
    except (ValueError, TypeError) as exc:
        raise StructuredOutputError(f"model did not return valid JSON: {exc}") from exc
    try:
        error = next(validator.iter_errors(value), None)
    except Exception as exc:
        raise StructuredOutputError(f"could not validate structured output: {exc}") from exc
    if error is not None:
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        if relaxed is not None and relaxed():
            print(f"[strata] structured output does not match the schema at {path}: {error.message}; the grammar "
                  "was relaxed to compile, so the answer is delivered as generated", flush=True)
            if notice is not None:
                notice.update(at=path, error=error.message)
            return text
        raise StructuredOutputError(f"model output failed the JSON Schema at {path}: {error.message}")
    return text
