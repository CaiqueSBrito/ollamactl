"""Runs against a fake local Ollama: python test_ollamactl.py"""
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from ollamactl import Ollama, OllamaError, _img, _tag

installed, loaded, pulled, deleted, chat_bodies = {"llama3.2:latest"}, set(), [], [], []
headers_seen, copied, progress_bodies = [], [], []


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, *chunks):
        self.send_response(code)
        self.end_headers()
        self.wfile.write(b"".join(json.dumps(c).encode() + b"\n" for c in chunks))

    def do_GET(self):
        headers_seen.append(self.headers)
        if self.path == "/api/version":
            return self.reply(200, {"version": "0"})
        names = installed if self.path == "/api/tags" else loaded
        self.reply(200, {"models": [{"name": n} for n in sorted(names)]})

    def do_DELETE(self):
        deleted.append(json.loads(self.rfile.read(int(self.headers["Content-Length"])))["model"])
        self.send_response(200)
        self.end_headers()  # empty body, like Ollama

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        headers_seen.append(self.headers)
        if self.path == "/api/copy":
            copied.append((body["source"], body["destination"]))
            self.send_response(200)
            return self.end_headers()
        if self.path in ("/api/create", "/api/push"):
            progress_bodies.append(body)
            return self.reply(200, {"status": "working"}, {"status": "success"})
        if body["model"] == "slow":  # non-stream: late answer; stream: stalls mid-way
            if not body["stream"]:
                time.sleep(1)
            self.reply(200, {"response": "a"})
            self.wfile.flush()
            return time.sleep(1)
        if body["model"] == "missing":
            return self.reply(404, {"error": "model not found"})
        if self.path == "/api/generate" and "prompt" not in body:  # load/unload
            (loaded.discard if body.get("keep_alive") == 0 else loaded.add)(body["model"])
            return self.reply(200, {"done": True})
        if self.path == "/api/generate":
            return self.reply(200, {"response": f"echo: {body['prompt']}", "images": body.get("images"), "done": True})
        if self.path == "/api/chat":
            chat_bodies.append(body)
            last = body["messages"][-1]
            if body["stream"]:
                return self.reply(200, {"message": {"content": "", "thinking": "hmm"}}, {"message": {"content": "Hi"}},
                                  {"message": {"content": ""}, "done": True, "eval_count": 3})
            if body["model"] == "loop" or last["role"] == "user":  # the prompt names the tool to call
                call = {"function": {"name": last["content"], "arguments": {"a": 2, "b": 3}}}
                return self.reply(200, {"message": {"role": "assistant", "content": "", "tool_calls": [call]}})
            return self.reply(200, {"message": {"role": "assistant", "content": "result: " + last["content"]}, "eval_count": 7})
        if self.path == "/api/show":
            return self.reply(200, {"capabilities": ["embedding"] if body["model"] == "embedder" else ["completion", "tools"]})
        if self.path == "/api/embed" and body["input"] == []:  # how embedding models get loaded
            loaded.add(body["model"])
            return self.reply(200, {"embeddings": []})
        if self.path == "/api/embed":
            return self.reply(200, {"embeddings": [[0.1, 0.2]]})
        if self.path == "/api/pull":
            if body["model"] == "full":
                return self.reply(200, {"status": "pulling"}, {"error": "disk full"})
            pulled.append(body["model"])
            return self.reply(200, {"status": "pulling", "total": 10, "completed": 5}, {"status": "success"})


srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
srv.daemon_threads = True
srv.handle_error = lambda *a: None  # "slow" writes to a socket the client already closed
threading.Thread(target=srv.serve_forever, daemon=True).start()
ai = Ollama(f"http://127.0.0.1:{srv.server_port}")

# generation: the whole API JSON, streaming included (thinking and metrics aren't lost)
assert ai.generate("m", "hi")["response"] == "echo: hi"
chunks = list(ai.chat("m", [{"role": "user", "content": "hi"}], stream=True))
assert chunks[0]["message"]["thinking"] == "hmm"
assert "".join(c["message"]["content"] for c in chunks) == "Hi"
assert chunks[-1]["eval_count"] == 3
assert ai.embed("m", "hi")["embeddings"] == [[0.1, 0.2]]

# images: bytes, path or base64
with tempfile.NamedTemporaryFile(delete=False) as f:
    f.write(b"png")
assert _img(b"png") == _img(f.name) == _img("cG5n") == "cG5n"
assert ai.generate("m", "hi", images=[f.name])["images"] == ["cG5n"]
os.remove(f.name)


# tools
def add(a: int, b: int) -> int:
    "Adds two numbers."
    return a + b


def broken(a: int, b: int = 0):
    raise ValueError("oops")


msgs = [{"role": "user", "content": "add"}]
r = ai.chat_tools("m", msgs, tools=[add, broken])
assert r["message"]["content"] == "result: 5" and r["eval_count"] == 7
assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
assert chat_bodies[-1]["tools"][0]["function"] == {
    "name": "add", "description": "Adds two numbers.",
    "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]},
}
assert chat_bodies[-1]["tools"][1]["function"]["parameters"]["required"] == ["a"]
r = ai.chat_tools("m", [{"role": "user", "content": "broken"}], tools=[add, broken])
assert r["message"]["content"] == "result: Error: ValueError: oops"  # errors go back to the model
r = ai.chat_tools("m", [{"role": "user", "content": "made_up"}], tools=[add])
assert r["message"]["content"].startswith("result: Error: KeyError")  # tool that doesn't exist
try:
    ai.chat_tools("loop", [{"role": "user", "content": "add"}], tools=[add], max_turns=3)
    raise AssertionError("should have failed")
except OllamaError:
    pass

# models
assert [m["name"] for m in ai.models()] == ["llama3.2:latest"]
assert "tools" in ai.show("m")["capabilities"]
assert ai.version() == "0" == ai.request("/api/version")["version"]
ai.delete("old")
assert deleted == ["old"]
assert [p["status"] for p in ai.pull("x", stream=True)] == ["pulling", "success"]
assert (_tag("llama3.2"), _tag("qwen3:8b"), _tag("localhost:5000/m")) == ("llama3.2:latest", "qwen3:8b", "localhost:5000/m:latest")
pulled.clear()
ai.load("llama3.2")
ai.load("qwen3")
assert pulled == ["qwen3"]  # only pulls what's missing
assert [m["name"] for m in ai.loaded()] == ["llama3.2", "qwen3"]
ai.unload("qwen3")
assert [m["name"] for m in ai.loaded()] == ["llama3.2"]
ai.load("embedder")  # embedding-only: can't be loaded through /api/generate
assert [m["name"] for m in ai.loaded()] == ["embedder", "llama3.2"]
ai.create("pirate", from_="llama3.2", system="Arr.")
assert progress_bodies[-1] == {"model": "pirate", "from": "llama3.2", "system": "Arr."}
assert [p["status"] for p in ai.push("me/pirate", stream=True)] == ["working", "success"]
ai.copy("pirate", "pirate2")
assert copied == [("pirate", "pirate2")]

# headers: custom ones go on every request; OLLAMA_API_KEY only ever goes to ollama.com
authed = Ollama(ai.host, headers={"X-Token": "secret"})
assert authed.running() and headers_seen[-1].get("X-Token") == "secret"
authed.show("m")
assert headers_seen[-1].get("X-Token") == "secret"
with mock.patch.dict(os.environ, {"OLLAMA_API_KEY": "k"}):
    assert "Authorization" not in Ollama(ai.host).headers
    assert Ollama("https://ollama.com").headers == {"Authorization": "Bearer k"}
    assert Ollama("https://ollama.com", headers={"authorization": "Bearer mine"}).headers == {"authorization": "Bearer mine"}

# server: already running -> start doesn't spawn another, stop doesn't kill what isn't ours
assert ai.running()
ai.start()
assert ai._proc is None
ai.stop()
assert not Ollama("http://127.0.0.1:1").running()

# install (subprocess mocked; never runs the real installer)
with mock.patch("ollamactl.util._exe", return_value="ollama"), mock.patch("subprocess.run") as run:
    ai.install()
    assert not run.called  # already installed -> no-op
with mock.patch("ollamactl.util._exe", side_effect=[None, "ollama"]), mock.patch("subprocess.run") as run:
    run.return_value.returncode = 0
    ai.install()
    assert "https://ollama.com/install." in run.call_args[0][0][-1]
with mock.patch("ollamactl.util._exe", return_value=None), mock.patch("subprocess.run") as run:
    run.return_value.returncode = 0  # `curl | sh` "succeeded" but installed nothing
    try:
        ai.install()
        raise AssertionError("should have failed")
    except OllamaError:
        pass

# errors: everything becomes OllamaError, timeouts included; status_code when the server answered with an error
slow = Ollama(ai.host, timeout=0.3)
for call, msg, code in [
    (lambda: ai.generate("missing", "hi"), "HTTP 404: model not found", 404),
    (lambda: ai.pull("full"), "disk full", None),
    (lambda: Ollama("http://127.0.0.1:1", timeout=2).models(), "unreachable", None),
    (lambda: slow.generate("slow", "hi"), "timed out", None),
    (lambda: list(slow.generate("slow", "hi", stream=True)), "timed out", None),
]:
    try:
        call()
        raise AssertionError("should have failed")
    except OllamaError as e:
        assert msg in str(e) and e.status_code == code, (e, e.status_code)
print("ok")
