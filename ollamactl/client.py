"""Ollama client: combines server lifecycle, model management and generation, plus the
low-level request() used by all of them."""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from . import util
from .errors import OllamaError
from .generation import GenerationMixin
from .models import ModelsMixin
from .server import ServerMixin


class Ollama(ServerMixin, ModelsMixin, GenerationMixin):
    # 127.0.0.1, not localhost: on Windows localhost tries IPv6 first and every connection stalls ~2s
    def __init__(self, host="http://127.0.0.1:11434", timeout=None, headers=None):
        self.host = host.rstrip("/")
        # seconds per read (per chunk when streaming). None = wait as long as it takes:
        # without streaming Ollama only answers when done, and long generations take minutes
        self.timeout = timeout
        self.headers = dict(headers or {})  # sent on every request, e.g. auth for a remote server
        key = os.environ.get("OLLAMA_API_KEY")
        # only to ollama.com: sent anywhere else, the key would leak to whoever runs that server
        if key and urllib.parse.urlsplit(self.host).hostname == "ollama.com" and \
                not any(k.lower() == "authorization" for k in self.headers):
            self.headers["Authorization"] = f"Bearer {key}"
        self._proc = None  # `ollama serve` process started by this instance

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def request(self, path, body=None, stream=False, method=None):
        """Raw call to any endpoint. Returns the JSON, or an iterator of JSONs if stream=True."""
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json", **self.headers}
        req = urllib.request.Request(self.host + path, data=data, method=method, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
            if stream:
                return util._lines(resp, self.host)
            with resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read())["error"]
            except Exception:
                msg = e.reason
            raise OllamaError(f"HTTP {e.code}: {msg}", e.code) from None
        except urllib.error.URLError as e:
            raise OllamaError(f"Ollama unreachable at {self.host}: {e.reason}") from None
        except OSError as e:  # timeout waiting for the response, connection dropped mid-way...
            raise OllamaError(f"Network error talking to {self.host}: {e}") from None
        return json.loads(raw) if raw else {}  # delete answers 200 with an empty body
