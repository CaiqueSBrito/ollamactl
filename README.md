# ollamactl

Zero-dependency Python library to install, run and talk to [Ollama](https://ollama.com), with automatic tool calling.

- **Manages Ollama itself.** Install it, start and stop the server, pull, load and unload models.
- **Runs tool calls for you.** Pass plain Python functions and `chat_tools()` runs the whole call-and-answer loop.
- **Covers the API.** Generate, chat, embeddings, streaming (with `thinking`, `tool_calls` and metrics), images and model management.
- **Pure standard library.** No dependencies. Python 3.8+.

> Not affiliated with Ollama. If you only need an HTTP client, the Ollama team maintains [ollama-python](https://github.com/ollama/ollama-python). ollamactl talks to the same REST API and adds server lifecycle and tool execution on top.

## Install

```bash
pip install git+https://github.com/CaiqueSBrito/ollamactl.git
```

To pin a version in `requirements.txt`, add a tag or commit after `@`:

```text
ollamactl @ git+https://github.com/CaiqueSBrito/ollamactl.git@<tag-or-commit>
```

For development, from a clone:

```bash
pip install -e .
```

## Quickstart

```python
from ollamactl import Ollama

ai = Ollama()
ai.install()                    # no-op if Ollama is already installed

with ai:                        # starts `ollama serve` if needed, stops it on exit
    ai.load("qwen2.5:7b")       # pulls the model if missing, loads it into memory
    r = ai.generate("qwen2.5:7b", "Why is the sky blue?")
    print(r["response"])
```

Every method returns the API's JSON as-is, so metrics come along: `r["eval_count"]`, `r["total_duration"]`, `r["done_reason"]`.
Extra API parameters (`system`, `options`, `format`, `think`, `keep_alive`...) are passed as keyword arguments.

## Tool calling

```python
def weather(city: str) -> dict:
    """Current weather in a city."""
    return {"city": city, "temp_c": 31}

msgs = [{"role": "user", "content": "What's the weather in Recife?"}]
r = ai.chat_tools("qwen2.5:7b", msgs, tools=[weather])
print(r["message"]["content"])  # the whole conversation, tool results included, is now in msgs
```

- The tool schema is built from the function's type hints and docstring. For a richer schema (enums, optional or nested fields), pass the API's tool dict instead of a function.
- If a tool raises, the error goes back to the model so it can retry. The loop stops after `max_turns` (default 10).
- To run tools yourself, use `ai.chat(model, msgs, tools=[weather])` and read `r["message"]["tool_calls"]`.

## Streaming

```python
for chunk in ai.chat("qwen2.5:7b", msgs, stream=True):
    print(chunk["message"]["content"], end="", flush=True)
```

Chunks are the raw API JSON: `thinking` and `tool_calls` arrive in the chunks, and metrics arrive in the last one.

## Images

```python
ai.generate("llava", "Describe this picture", images=["photo.jpg"])  # paths, bytes or base64
```

In `chat`, put the same values in a message's `"images"` list.

## API

| Method | What it does |
|---|---|
| `install()` | Installs Ollama with the official script from ollama.com. Does nothing if it's already installed. |
| `start()` / `stop()` / `running()` | Starts, stops and checks the server. Also available as `with Ollama() as ai:`. |
| `load(model)` / `unload(model)` / `loaded()` | Puts a model in memory, takes it out, and lists what's in memory. `load` pulls the model first if it's missing. |
| `models()` / `show(model)` / `pull(model)` / `delete(model)` / `version()` | Lists installed models, shows one model's details, downloads (`stream=True` for progress), deletes, and returns the server version. |
| `create(model, from_=...)` / `copy(src, dst)` / `push(model)` | Creates a model from another one (with its own `system`, `parameters`, `template`...), copies one under a new name, and uploads one to ollama.com. |
| `generate()` / `chat()` / `chat_tools()` / `embed()` | Generates text, chats, chats with automatic tool execution, and creates embeddings. |
| `request(path, body, stream, method)` | Calls any endpoint without a dedicated method, such as ollama.com's `/api/web_search`. |

```python
ai.create("pirate", from_="qwen2.5:7b", system="Always answer in pirate speak.")
```

## Remote servers and ollama.com

```python
# a server behind an auth proxy or gateway: headers go on every request
ai = Ollama("https://my-server:11434", headers={"Authorization": "Bearer ..."})

# ollama.com (cloud models, web search): reads OLLAMA_API_KEY from the environment
cloud = Ollama("https://ollama.com")
cloud.request("/api/web_search", {"query": "what is ollama"})
```

`OLLAMA_API_KEY` is only sent automatically to ollama.com. Sent to any other host, it would leak to whoever runs that server. Headers you pass yourself go wherever you point the client.

## Errors

Every failure raises `OllamaError`: HTTP errors, network errors, timeouts and errors in the middle of a stream.
`e.status_code` holds the HTTP status when the server replied with an error.

```python
from ollamactl import OllamaError

try:
    ai.generate("no-such-model", "hi")
except OllamaError as e:
    if e.status_code == 404:
        ...
```

## Good to know

- **Only stops what it started.** `stop()`, or leaving the `with` block, only stops a server this instance started. An Ollama that was already running (the desktop app or a systemd service) is left alone, and so are its loaded models, which unload after `keep_alive` expires (5 minutes by default).
- **The server outlives your script** if you call `start()` without `with` or `stop()`.
- **`install()` runs a remote script.** It uses the official ollama.com script and installs the latest version. On Linux it asks for sudo and sets up a systemd service.
- **`load()` needs the full tag.** `load("qwen2.5")` means `qwen2.5:latest`, so use the exact tag to avoid pulling a model you already have under another tag.
- **`keep_alive` resets on every call.** Each `generate()` or `chat()` call without `keep_alive` resets the model's unload timer to the server default.
- **No timeout by default.** `timeout` defaults to `None` (wait as long as it takes), because non-streaming responses only arrive when generation ends.
- **The default host is `http://127.0.0.1:11434`.** Using `localhost` adds about 2 seconds per request on Windows.

## Examples

- [`examples/rag.py`](examples/rag.py): retrieval-augmented generation with `bge-m3` embeddings, in two modes. In classic mode the retrieved passages go into the prompt. In agentic mode the model calls a `search` tool itself.

## Tests

```bash
python test_ollamactl.py
```

## License

MIT
