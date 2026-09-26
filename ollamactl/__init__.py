"""Zero-dependency client for Ollama: manages the server and talks to its REST API.

    from ollamactl import Ollama

    def weather(city: str) -> str:
        "Current temperature in a city."
        return "25°C"

    with Ollama() as ai:              # starts `ollama serve` if it isn't running
        ai.load("qwen2.5:7b")         # pulls the model if missing and loads it into memory
        print(ai.generate("qwen2.5:7b", "Why is the sky blue?")["response"])

        msgs = [{"role": "user", "content": "What's the weather in Recife?"}]
        print(ai.chat_tools("qwen2.5:7b", msgs, tools=[weather])["message"]["content"])

        for c in ai.chat("qwen2.5:7b", [{"role": "user", "content": "Hi!"}], stream=True):
            print(c["message"]["content"], end="", flush=True)
    # on exit: unloads models and stops the server (only if this library started it)

Methods return the API's JSON as-is: text, thinking, tool_calls and metrics.
Extra API parameters (system, options, format, think, keep_alive...) pass through **kw.
ollama.com (cloud models, web search) with OLLAMA_API_KEY set in the environment:
    Ollama("https://ollama.com").request("/api/web_search", {"query": "ollama"})
"""
from .client import Ollama
from .errors import OllamaError
from .util import _img, _tag  # re-exported: handy in scripts/tests, not part of the stable API

__all__ = ["Ollama", "OllamaError"]
