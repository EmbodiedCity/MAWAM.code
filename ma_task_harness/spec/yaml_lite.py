"""Lightweight YAML / JSON serialization with zero mandatory third-party deps.

Uses PyYAML (`yaml.safe_load` / `yaml.safe_dump`) when installed; otherwise falls
back to a clean, deterministic block+flow YAML 1.2 subset parser and emitter that
supports nested mappings, sequences, inline flow dicts/lists, comments, strings,
booleans, ints, floats, and nulls.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, List, Tuple

try:
    import yaml as _pyyaml  # type: ignore
except ImportError:
    _pyyaml = None


def loads_yaml(text: str) -> Any:
    """Parse YAML (or JSON) text into Python data structures."""
    stripped = text.strip()
    if not stripped:
        return {}
    if _pyyaml is not None:
        return _pyyaml.safe_load(text)
    if stripped[0] in "{[":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    lines = _preprocess_lines(text)
    if not lines:
        return {}
    val, idx = _parse_block(lines, 0, lines[0][0])
    if idx != len(lines):
        raise ValueError(f"Unexpected trailing YAML content at line: {lines[idx][1]!r}")
    return val


def dumps_yaml(data: Any) -> str:
    """Serialize Python data structures into clean, human-readable YAML."""
    if _pyyaml is not None:
        return _pyyaml.safe_dump(
            data,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=100,
        )
    return "\n".join(_emit_block(data, 0)) + "\n"


def load_yaml_file(path: str | Path) -> Any:
    return loads_yaml(Path(path).read_text(encoding="utf-8"))


def save_yaml_file(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps_yaml(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Minimal zero-dependency YAML subset parser & emitter
# ---------------------------------------------------------------------------


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    escape = False
    for i, ch in enumerate(line):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_double:
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if i == 0 or line[i - 1].isspace():
                return line[:i].rstrip()
    return line.rstrip()


def _preprocess_lines(text: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        cleaned = _strip_comment(raw)
        if not cleaned.strip():
            continue
        if cleaned.strip() == "---":
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        out.append((indent, cleaned.strip()))
    return out


def _split_top_level(text: str, sep: str) -> List[str]:
    parts: List[str] = []
    depth_brace = 0
    depth_bracket = 0
    in_single = False
    in_double = False
    escape = False
    start = 0
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_double:
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace -= 1
            elif ch == "[":
                depth_bracket += 1
            elif ch == "]":
                depth_bracket -= 1
            elif ch == sep and depth_brace == 0 and depth_bracket == 0:
                parts.append(text[start:i])
                start = i + 1
    parts.append(text[start:])
    return parts


def _find_mapping_colon(content: str) -> int:
    depth_brace = 0
    depth_bracket = 0
    in_single = False
    in_double = False
    escape = False
    for i, ch in enumerate(content):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_double:
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace -= 1
            elif ch == "[":
                depth_bracket += 1
            elif ch == "]":
                depth_bracket -= 1
            elif ch == ":" and depth_brace == 0 and depth_bracket == 0:
                if i + 1 == len(content) or content[i + 1].isspace():
                    return i
    return -1


def _parse_scalar(token: str) -> Any:
    s = token.strip()
    if not s or s in ("null", "Null", "NULL", "~"):
        return None
    if s in ("true", "True", "TRUE"):
        return True
    if s in ("false", "False", "FALSE"):
        return False
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        if s.startswith("'"):
            return s[1:-1].replace("''", "'")
        return ast.literal_eval(s)
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(p) for p in _split_top_level(inner, ",") if p.strip()]
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1].strip()
        if not inner:
            return {}
        out = {}
        for item in _split_top_level(inner, ","):
            if not item.strip():
                continue
            cidx = _find_mapping_colon(item.strip())
            if cidx < 0:
                raise ValueError(f"Invalid inline mapping entry: {item!r}")
            k = _parse_scalar(item.strip()[:cidx])
            v = _parse_scalar(item.strip()[cidx + 1 :])
            out[str(k)] = v
        return out
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_block(lines: List[Tuple[int, str]], idx: int, base_indent: int) -> Tuple[Any, int]:
    if idx >= len(lines):
        return {}, idx
    _, first_text = lines[idx]
    if first_text == "-" or first_text.startswith("- "):
        return _parse_sequence(lines, idx, base_indent)
    if first_text.startswith("[") or first_text.startswith("{"):
        return _parse_scalar(first_text), idx + 1
    return _parse_mapping(lines, idx, base_indent)


def _parse_sequence(lines: List[Tuple[int, str]], idx: int, base_indent: int) -> Tuple[List[Any], int]:
    items: List[Any] = []
    while idx < len(lines):
        indent, content = lines[idx]
        if indent < base_indent:
            break
        if indent != base_indent or not (content == "-" or content.startswith("- ")):
            break
        rest = content[1:].strip()
        idx += 1
        if not rest:
            if idx < len(lines) and lines[idx][0] > base_indent:
                val, idx = _parse_block(lines, idx, lines[idx][0])
                items.append(val)
            else:
                items.append(None)
            continue
        cidx = _find_mapping_colon(rest)
        if cidx >= 0 and not (rest.startswith("{") or rest.startswith("[")):
            k = str(_parse_scalar(rest[:cidx]))
            v_str = rest[cidx + 1 :].strip()
            item_dict: dict[str, Any] = {}
            if v_str:
                item_dict[k] = _parse_scalar(v_str)
            else:
                if idx < len(lines) and lines[idx][0] > base_indent:
                    sub_val, idx = _parse_block(lines, idx, lines[idx][0])
                    item_dict[k] = sub_val
                else:
                    item_dict[k] = None
            if idx < len(lines) and lines[idx][0] > base_indent:
                extra_map, idx = _parse_mapping(lines, idx, lines[idx][0])
                item_dict.update(extra_map)
            items.append(item_dict)
        else:
            items.append(_parse_scalar(rest))
    return items, idx


def _parse_mapping(lines: List[Tuple[int, str]], idx: int, base_indent: int) -> Tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    while idx < len(lines):
        indent, content = lines[idx]
        if indent < base_indent:
            break
        if indent != base_indent:
            raise ValueError(f"Unexpected indentation {indent} != {base_indent} at: {content!r}")
        if content == "-" or content.startswith("- "):
            break
        cidx = _find_mapping_colon(content)
        if cidx < 0:
            raise ValueError(f"Expected key: value in YAML mapping at: {content!r}")
        key = str(_parse_scalar(content[:cidx]))
        val_str = content[cidx + 1 :].strip()
        idx += 1
        if val_str:
            mapping[key] = _parse_scalar(val_str)
        else:
            if idx < len(lines) and lines[idx][0] > base_indent:
                sub_val, idx = _parse_block(lines, idx, lines[idx][0])
                mapping[key] = sub_val
            else:
                mapping[key] = None
    return mapping, idx


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def _format_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    if (
        not s
        or s != s.strip()
        or any(c in s for c in ":#{}[],&*?|>!%@`\"\n")
        or s.lower() in ("true", "false", "null", "yes", "no", "on", "off", "~")
    ):
        return json.dumps(s, ensure_ascii=False)
    try:
        float(s)
        return json.dumps(s, ensure_ascii=False)
    except ValueError:
        return s


def _emit_block(value: Any, indent: int) -> List[str]:
    pad = " " * indent
    if _is_scalar(value):
        return [f"{pad}{_format_scalar(value)}"]
    if isinstance(value, list):
        if not value:
            return [f"{pad}[]"]
        if all(_is_scalar(x) for x in value) and sum(len(str(x)) for x in value) <= 60:
            inner = ", ".join(_format_scalar(x) for x in value)
            return [f"{pad}[{inner}]"]
        lines: List[str] = []
        for item in value:
            if _is_scalar(item):
                lines.append(f"{pad}- {_format_scalar(item)}")
            elif isinstance(item, dict) and item:
                it = list(item.items())
                k0, v0 = it[0]
                if _is_scalar(v0):
                    lines.append(f"{pad}- {k0}: {_format_scalar(v0)}")
                else:
                    lines.append(f"{pad}- {k0}:")
                    lines.extend(_emit_block(v0, indent + 4))
                for kn, vn in it[1:]:
                    if _is_scalar(vn):
                        lines.append(f"{pad}  {kn}: {_format_scalar(vn)}")
                    elif isinstance(vn, list) and all(_is_scalar(x) for x in vn) and len(vn) <= 6:
                        inner = ", ".join(_format_scalar(x) for x in vn)
                        lines.append(f"{pad}  {kn}: [{inner}]")
                    else:
                        lines.append(f"{pad}  {kn}:")
                        lines.extend(_emit_block(vn, indent + 4))
            elif isinstance(item, list) and all(_is_scalar(x) for x in item) and sum(len(str(x)) for x in item) <= 60:
                inner = ", ".join(_format_scalar(x) for x in item)
                lines.append(f"{pad}- [{inner}]")
            else:
                lines.append(f"{pad}-")
                lines.extend(_emit_block(item, indent + 2))
        return lines
    if isinstance(value, dict):
        if not value:
            return [f"{pad}{{}}"]
        lines = []
        for k, v in value.items():
            if _is_scalar(v):
                lines.append(f"{pad}{k}: {_format_scalar(v)}")
            elif isinstance(v, list) and all(_is_scalar(x) for x in v) and sum(len(str(x)) for x in v) <= 60:
                inner = ", ".join(_format_scalar(x) for x in v)
                lines.append(f"{pad}{k}: [{inner}]")
            else:
                lines.append(f"{pad}{k}:")
                lines.extend(_emit_block(v, indent + 2))
        return lines
    return [f"{pad}{_format_scalar(str(value))}"]
