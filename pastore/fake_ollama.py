"""Fake Ollama HTTP server for `check --contract` and tests: answers ok, returns 503, or hangs; plus a refusing URL.

Depends on: stdlib http.server and threading only; never touches the real Ollama at 127.0.0.1:11434.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Literal, Self

Mode = Literal["ok", "503", "hang"]


class FakeOllama:
    """A local HTTP server standing in for Ollama in one of three modes."""

    def __init__(self, mode: Mode) -> None:
        self.mode = mode
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

                body: object
                if fake.mode == "503":
                    self.send_response(503)
                    body = {"error": "fake 503"}
                else:  # "ok"
                    if self.path == "/api/tags":
                        body = {"models": []}
                    else:
                        body = {}
                    self.send_response(200)

                data = json.dumps(body).encode("utf-8")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

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
