"""Stateless helpers: model schemas, image encoding, executable lookup, streaming."""
import base64
import inspect
import json
import os
import shutil
import typing

from .errors import OllamaError

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def _schema(tool):
    """Python function -> tool in the API format. Dicts pass through."""
    if isinstance(tool, dict):
        return tool
    params = inspect.signature(tool).parameters
    hints = typing.get_type_hints(tool)
    # ponytail: Optional, Literal, classes etc. become "string"; need a richer schema? pass the API dict
    props = {n: {"type": _JSON_TYPES.get(typing.get_origin(hints.get(n)) or hints.get(n), "string")} for n in params}
    return {"type": "function", "function": {
        "name": tool.__name__,
        "description": inspect.getdoc(tool) or "",
        "parameters": {"type": "object", "properties": props,
                       "required": [n for n, p in params.items() if p.default is p.empty]},
    }}


def _img(x):
    """Image as bytes, file path or base64 -> base64."""
    if isinstance(x, (bytes, bytearray)):
        return base64.b64encode(x).decode()
    if os.path.isfile(x):
        with open(x, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return x  # already base64


def _exe():
    """Path to the Ollama executable, or None. The fixed paths cover the moment right after
    install(), when this process's PATH hasn't been updated yet."""
    for p in (
        shutil.which("ollama"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),  # Windows
        "/Applications/Ollama.app/Contents/Resources/ollama",  # macOS
    ):
        if p and os.path.isfile(p):
            return p


def _tag(model):
    """'llama3.2' -> 'llama3.2:latest', as Ollama lists it. Looks only at the last path
    segment so 'host:port/model' isn't mistaken for a tag."""
    return model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"


def _lines(resp, host):
    # separate from request() so connection errors raise at call time, not on the first next()
    with resp:
        try:
            for line in resp:
                if line.strip():
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise OllamaError(chunk["error"])
                    yield chunk
        except OSError as e:
            raise OllamaError(f"Network error talking to {host}: {e}") from None
