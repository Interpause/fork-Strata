"""Compile OpenAI/MCP tool JSON Schemas into an llguidance Lark fragment.

The chat template (`serve/chat_template.jinja`) does not write a tool's arguments as one JSON object.  Per
argument it writes

    <parameter=NAME>\\n VALUE \\n</parameter>\\n

and VALUE is the value **verbatim** when it is a JSON string (unquoted, possibly multi-line), `tojson` output for
every other type.  So the outer `{"type": "object", "properties": ..., "required": ...}` container of
`tool.function.parameters` cannot become a `%json` object: it is unwrapped into one rule per parameter.

Required parameters take bits 0..R-1 and optional ones R..R+O-1, and the empty exit is guarded by `is_ones([0:R])`,
so a call cannot leave the parameter list with a required argument missing.  `bit_count_ge` would be wrong: it is
satisfied by R optional parameters while a required one is missing.

Nothing here reads a tool name or a parameter name from anywhere but the request; `skeleton.lark` names none.
"""

from __future__ import annotations

import json
import re
from typing import Any

# With no tools declared, `tool_dispatch` must be reachable but unable to match anything.  A regex that cannot
# match a byte string in the pack's vocabulary is the cheapest way to say so; picking a `\x00`-`\xff` negation
# keeps it independent of the tokenizer.
NO_TOOLS = "tool_dispatch: NEVER_MATCH\nNEVER_MATCH: /[^\\x00-\\xff]/"


def slugify(name: str, idx: int) -> str:
    """A Lark-safe rule component.  The index keeps two tools whose names differ only in punctuation (a.b / a-b)
    from colliding."""
    return f"t{idx}_" + re.sub(r"[^a-zA-Z0-9_]", "_", name)


def lark_escape(s: str) -> str:
    """A double-quoted Lark string literal.  `ensure_ascii=False` keeps a non-ASCII name as UTF-8 text instead of
    inventing \\uXXXX escapes that Lark would then have to interpret."""
    return json.dumps(s, ensure_ascii=False)


def _json_schema(schema: dict[str, Any]) -> str:
    """The inline payload after `%json`.  serde_json parses it, so it is pasted verbatim: no Lark quoting."""
    return json.dumps(schema)


def compile_param_value_rule(rule_name: str, param_name: str, prop_schema: dict[str, Any],
                             root_defs: dict[str, Any]) -> str:
    """One property's JSON Schema -> the rule for its `<parameter=NAME>` block."""
    prefix = lark_escape(f"<parameter={param_name}>\n")
    close_tag = lark_escape("\n</parameter>\n")

    # A `$ref` to a string definition is inlined; anything else is handed to %json, which resolves refs itself.
    target = prop_schema
    if "$ref" in prop_schema and str(prop_schema["$ref"]).startswith("#/$defs/"):
        target = root_defs.get(str(prop_schema["$ref"]).split("/")[-1], prop_schema)
    kind = target.get("type")

    # A pure string is RAW text: the template writes the value with no quotes and no escapes.
    if kind == "string":
        if isinstance(target.get("const"), str):
            return f"{rule_name}: {prefix} {lark_escape(target['const'])} {close_tag}"
        if target.get("enum") and all(isinstance(x, str) for x in target["enum"]):
            alts = " | ".join(lark_escape(x) for x in target["enum"])
            return f"{rule_name}: {prefix} ({alts}) {close_tag}"
        if isinstance(target.get("pattern"), str):
            # Lark has no anchors: strip the schema's outer ^ and $ and escape the delimiter.
            pat = target["pattern"].lstrip("^").rstrip("$").replace("/", r"\/")
            return f"{rule_name}: {prefix} /(?:{pat})/ {close_tag}"
        # An unconstrained string: `raw_param_end` is the skeleton's lazy terminal, which ends a value only at a
        # standalone "\\n</parameter>\\n" line and can never match the empty string.
        return f"{rule_name}: {prefix} raw_param_end"

    # A union containing "string" keeps the raw form as one branch, so a string-typed argument stays unquoted.
    if isinstance(kind, list) and "string" in kind:
        others = [t for t in kind if t != "string"]
        branch = {**target, "type": others if len(others) > 1 else others[0]}
        if root_defs and "$defs" not in branch and "$ref" in _json_schema(branch):
            branch["$defs"] = root_defs
        return (f"{rule_name}: {prefix} raw_param_end | {prefix} %json {_json_schema(branch)} {close_tag}")

    # Every non-string type is `tojson` output, which is exactly a JSON value.
    embedded = dict(prop_schema)
    if root_defs and "$defs" not in embedded and "$ref" in _json_schema(embedded):
        embedded["$defs"] = root_defs
    return f"{rule_name}: {prefix} %json {_json_schema(embedded)} {close_tag}"


def compile_tools_fragment(tools: list[dict[str, Any]]) -> str:
    """The request's tools -> the text that replaces `skeleton.lark`'s SPLICE_TOOLS block.

    Per tool and in request order: a `tool_dispatch` alternative, `call_<slug>`, `params_<slug>`, one
    `param_<slug>_<i>` per parameter, and the two mid-call entries `entry_call_named_<slug>` (the prompt already
    wrote `=NAME>`) and `entry_call_name_<slug>` (the name is still to be written).
    """
    if not tools:
        return NO_TOOLS

    dispatch: list[str] = []
    blocks: list[str] = []
    for idx, tool in enumerate(tools):
        fn = tool.get("function", tool)
        name = fn["name"]
        slug = slugify(name, idx)
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        props: dict[str, Any] = params.get("properties") or {}
        wanted = set(params.get("required") or ())
        defs = params.get("$defs") or params.get("definitions") or {}

        required = [k for k in props if k in wanted]
        optional = [k for k in props if k not in wanted]
        order = required + optional
        n_req, n_all = len(required), len(order)

        dispatch.append(f"call_{slug}")
        lines = [f"// --- Tool: {name} ---"]
        head = lark_escape(f"{name}>\n")
        head_eq = lark_escape(f"={name}>\n")

        if n_all == 0:
            lines.append(f"call_{slug}: {head} params_{slug}")
            lines.append(f'params_{slug}: ""')
            params_ref = f"params_{slug}"
        elif n_all == 1:
            rule = f"param_{slug}_0"
            # The single parameter is optional exactly when it is not in `required`.
            lines.append(f"call_{slug}: {head} params_{slug}")
            lines.append(f"params_{slug}: {rule}{'' if n_req == 1 else '?'}")
            lines.append(compile_param_value_rule(rule, order[0], props[order[0]], defs))
            params_ref = f"params_{slug}"
        else:
            lines.append(f"call_{slug}: {head} params_{slug}::0x0")
            exit_cond = f" %if is_ones([0:{n_req}])" if n_req else ""
            branches = [f'""{exit_cond}']
            for bit, key in enumerate(order):
                rule = f"param_{slug}_{bit}"
                branches.append(f"{rule} params_{slug}::set_bit({bit}) %if bit_clear({bit})")
                lines.append(f"// bit {bit} = {key} ({'required' if bit < n_req else 'optional'})")
                lines.append(compile_param_value_rule(rule, key, props[key], defs))
            lines.append(f"params_{slug}::_ : " + "\n    | ".join(branches))
            params_ref = f"params_{slug}::0x0"

        tail = ' "</function>\\n" </tool_call> more_tool_calls <|im_end|>'
        lines.append(f"entry_call_named_{slug}: {params_ref}{tail}")
        lines.append(f"entry_call_name_{slug}: {head_eq} {params_ref}{tail}")
        blocks.append("\n".join(lines))

    return "tool_dispatch: " + "\n             | ".join(dispatch) + "\n\n" + "\n\n".join(blocks)
