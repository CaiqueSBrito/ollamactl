"""Model management: pull, push, create, copy, delete, load/unload."""
from . import util


class ModelsMixin:
    def load(self, model, **kw):
        """Pulls the model if missing and loads it into memory. kw: keep_alive (-1 = forever).
        Note: every generate/chat without keep_alive resets the timer to the server default (5m)."""
        if util._tag(model) not in [m["name"] for m in self.models()]:
            self.pull(model)
        if "completion" in self.show(model).get("capabilities", ["completion"]):
            self.request("/api/generate", {"model": model, **kw})
        else:  # embedding-only models reject /api/generate; an empty /api/embed loads them
            self.request("/api/embed", {"model": model, "input": [], **kw})

    def unload(self, model):
        """Removes the model from memory (it stays on disk)."""
        self.request("/api/generate", {"model": model, "keep_alive": 0})

    def loaded(self):
        """Models currently in memory (name, size_vram, expires_at...)."""
        return self.request("/api/ps")["models"]

    def models(self):
        """Installed models (name, size, modified_at, details with family and quantization)."""
        return self.request("/api/tags")["models"]

    def show(self, model):
        """Model details: capabilities (tools, vision, thinking...), model_info
        (context length etc.), template and parameters."""
        return self.request("/api/show", {"model": model})

    def pull(self, model, stream=False, **kw):
        """Downloads the model. stream=True iterates progress ({status, total, completed});
        otherwise blocks until done."""
        return self._progress("/api/pull", {"model": model, **kw}, stream)

    def push(self, model, stream=False, **kw):
        """Uploads the model to ollama.com. The name must be namespaced ("user/model") and
        this machine's Ollama key registered in your ollama.com account. stream as in pull()."""
        return self._progress("/api/push", {"model": model, **kw}, stream)

    def create(self, model, from_=None, stream=False, **kw):
        """Creates a model, usually from another one: create("pirate", from_="qwen2.5:7b",
        system="Talk like a pirate.", parameters={"temperature": 0.9}). kw: system, template,
        parameters, messages, license, quantize... stream as in pull()."""
        if from_:
            kw["from"] = from_  # `from` is a Python keyword
        return self._progress("/api/create", {"model": model, **kw}, stream)

    def copy(self, source, destination):
        """Copies a model under a new name."""
        self.request("/api/copy", {"source": source, "destination": destination})

    def delete(self, model):
        """Deletes the model from disk."""
        self.request("/api/delete", {"model": model}, method="DELETE")

    def _progress(self, path, body, stream):
        # always streamed on the wire: the timeout applies per chunk, not to the whole operation
        r = self.request(path, body, stream=True)
        if stream:
            return r
        for _ in r:
            pass

    def version(self):
        return self.request("/api/version")["version"]
