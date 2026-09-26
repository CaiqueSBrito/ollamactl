"""Server lifecycle: install, start, stop `ollama serve`."""
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from . import util
from .errors import OllamaError


class ServerMixin:
    def running(self):
        """Is the server responding?"""
        try:
            urllib.request.urlopen(urllib.request.Request(self.host + "/api/version", headers=self.headers), timeout=2).close()
            return True
        except OSError:
            return False

    def install(self):
        """Installs Ollama with the official script from ollama.com, unless already installed.
        Windows: no admin, into %LOCALAPPDATA%\\Programs\\Ollama. Linux: asks for sudo and
        creates a systemd service (which starts the server). macOS: /Applications, opens the app."""
        if util._exe():
            return
        if sys.platform == "win32":
            cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "irm https://ollama.com/install.ps1 | iex"]
        else:
            cmd = ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]
        # check for the executable afterwards: `curl | sh` exits 0 even if the download fails
        if subprocess.run(cmd).returncode or not util._exe():
            raise OllamaError("Failed to install Ollama (see the output above)")

    def start(self, wait=30):
        """Starts `ollama serve` if it isn't running. Idempotent."""
        if self.running():
            return
        exe = util._exe()
        if not exe:
            raise OllamaError("Ollama is not installed. Call install() first.")
        env = {**os.environ, "OLLAMA_HOST": urllib.parse.urlsplit(self.host).netloc}
        # ponytail: logs go to DEVNULL (an unread PIPE would block the server); if startup fails, run `ollama serve` by hand to see why
        self._proc = subprocess.Popen(
            [exe, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + wait
        while not self.running():
            if self._proc.poll() is not None:
                self._proc = None
                raise OllamaError("`ollama serve` exited during startup. Run it in a terminal to see the error.")
            if time.monotonic() > deadline:
                self.stop()
                raise OllamaError(f"Ollama did not respond within {wait}s")
            time.sleep(0.2)

    def stop(self):
        """Stops the server, but only if this instance started it."""
        if not self._proc:
            return
        if self.running():
            for m in self.loaded():  # unload first: frees VRAM and leaves no orphan runner
                self.unload(m["name"])
        self._proc.terminate()
        self._proc.wait()
        self._proc = None
