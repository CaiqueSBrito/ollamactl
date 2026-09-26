"""Generation: generate, chat, chat_tools, embed."""
import json

from . import util
from .errors import OllamaError


class GenerationMixin:
    def generate(self, model, prompt, stream=False, images=None, **kw):
        """r["response"] is the text; thinking, done_reason and metrics (eval_count,
        total_duration...) come along. stream=True iterates chunks in the same shape.
        images: file paths, bytes or base64."""
        if images:
            kw["images"] = [util._img(i) for i in images]
        return self.request("/api/generate", {"model": model, "prompt": prompt, "stream": stream, **kw}, stream)

    def chat(self, model, messages, stream=False, tools=None, **kw):
        """r["message"] holds content, thinking and tool_calls (append it straight to the
        history); the rest are metrics. stream=True iterates chunks in the same shape.
        tools: Python functions (schema built from type hints and docstring) or API dicts.
        "images" in messages: file paths, bytes or base64."""
        messages = [{**m, "images": [util._img(i) for i in m["images"]]} if m.get("images") else m for m in messages]
        if tools:
            kw["tools"] = [util._schema(t) for t in tools]
        return self.request("/api/chat", {"model": model, "messages": messages, "stream": stream, **kw}, stream)

    def chat_tools(self, model, messages, tools, max_turns=10, **kw):
        """chat that runs tool_calls by itself: calls the requested Python functions, sends
        the results back to the model and repeats until it answers without asking for a tool.
        Appends everything to `messages` and returns the final API response."""
        funcs = {t.__name__: t for t in tools if callable(t)}
        for _ in range(max_turns):
            r = self.chat(model, messages, tools=tools, **kw)
            messages.append(r["message"])
            if not r["message"].get("tool_calls"):
                return r
            for call in r["message"]["tool_calls"]:
                fn = call["function"]
                try:
                    out = funcs[fn["name"]](**fn["arguments"])
                except Exception as e:  # goes back to the model, which can fix its arguments
                    out = f"Error: {type(e).__name__}: {e}"
                content = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)
                messages.append({"role": "tool", "tool_name": fn["name"], "content": content})
        raise OllamaError(f"Model kept requesting tools after {max_turns} turns")

    def embed(self, model, input, **kw):
        """input: str or list of str. r["embeddings"] is the list of vectors."""
        return self.request("/api/embed", {"model": model, "input": input, **kw})
