"""Fake Ollama HTTP server for `check --contract` and tests: answers ok, returns 503, or hangs; plus a refusing URL.

Depends on: stdlib http.server and threading only; never touches the real Ollama at 127.0.0.1:11434.
"""

import json
import socket
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Literal, Self

Mode = Literal["ok", "503", "hang"]
MAX_POST_BYTES = 1024 * 1024


def _declared_size(header: str) -> int:
    """The body length a Content-Length header declares: 0 when it is not a plain number, over the cap when it is too long."""
    if not header.isdecimal():
        return 0
    return int(header) if len(header) <= 9 else MAX_POST_BYTES + 1


class FakeOllama:
    """A local HTTP server standing in for Ollama in one of three modes."""

    def __init__(
        self,
        mode: Mode,
        models: Mapping[str, int] | None = None,
        loaded: Mapping[str, int] | None = None,
        show: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        self.mode = mode
        self.models: dict[str, int] = dict(models or {})
        self.loaded: dict[str, int] = dict(loaded or {})
        self.show: dict[str, Mapping[str, object]] = dict(show or {})
        self.requests: list[str] = []
        self._release = threading.Event()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        fake = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                self._answer()

            def do_POST(self) -> None:
                self._answer()

            def _answer(self) -> None:
                fake.requests.append(self.path)

                if fake.mode == "hang":
                    fake._release.wait(3600)
                    return

                declared = self.headers.get("Content-Length", "")
                status, body = self._reply(_declared_size(declared))
                self.send_response(status)
                data = json.dumps(body).encode("utf-8")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _reply(self, size: int) -> tuple[int, object]:
                if size > MAX_POST_BYTES:
                    return 413, {"error": "fake 413"}
                raw = self.rfile.read(size)
                try:
                    sent = json.loads(raw) if raw else None
                except ValueError:
                    sent = None
                if fake.mode == "503":
                    return 503, {"error": "fake 503"}
                return 200, fake._ok_body(self.path, sent)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def _ok_body(self, path: str, sent: object) -> object:
        if path == "/api/tags":
            return {"models": [{"name": n, "model": n, "size": s} for n, s in self.models.items()]}
        if path == "/api/ps":
            return {
                "models": [
                    {"name": n, "model": n, "size": v, "size_vram": v}
                    for n, v in self.loaded.items()
                ]
            }
        if path == "/api/show":
            model = sent.get("model") if isinstance(sent, dict) else None
            return {"model_info": dict(self.show.get(model, {})) if isinstance(model, str) else {}}
        if path == "/api/version":
            return {"version": "0.0.0-fake"}
        return {}

    @property
    def url(self) -> str:
        """The server's base URL, for the job's `ollama_env` variable."""
        if self._server is None:
            raise RuntimeError("FakeOllama not started")
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._server is not None:
            self._release.set()
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def refused_url() -> str:
    """A URL on a just-freed local port, so a connection to it is refused."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}"
