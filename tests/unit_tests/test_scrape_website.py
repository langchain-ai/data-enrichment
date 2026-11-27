"""scrape_website reports fetch failures to the model instead of raising them.

Uses pages served on 127.0.0.1 and a stand-in summarizer, so no model API calls
or credentials are needed.
"""

import asyncio
import http.server
import importlib
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from enrichment_agent.state import State

tools_mod = importlib.import_module("enrichment_agent.tools")

STATE = State(
    topic="Example Corp",
    extraction_schema={"type": "object", "properties": {"founded": {"type": "string"}}},
)


class _Site(http.server.BaseHTTPRequestHandler):
    status = 200
    body = b"<p>Founded 1999.</p>"
    disconnect = False
    hang: threading.Event | None = None  # set: wait (without answering) until released

    def do_GET(self) -> None:  # noqa: N802
        if type(self).hang is not None:
            type(self).hang.wait(10)  # type: ignore[union-attr]
            return
        if type(self).disconnect:
            self.connection.close()  # drop the connection without answering
            return
        self.send_response(type(self).status)
        self.send_header("Content-Length", str(len(type(self).body)))
        self.end_headers()
        self.wfile.write(type(self).body)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def site() -> Iterator[type[_Site]]:
    handler = type("Site", (_Site,), {})  # per-test copy of the settings
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    handler.url = f"http://127.0.0.1:{server.server_address[1]}/about"  # type: ignore[attr-defined]
    try:
        yield handler
    finally:
        if handler.hang is not None:
            handler.hang.set()  # release a handler still waiting
        server.shutdown()
        server.server_close()


class Summarizer:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> AIMessage:
        self.prompts.append(prompt)
        return AIMessage(content="notes")


@pytest.fixture
def summarizer(monkeypatch: pytest.MonkeyPatch) -> Summarizer:
    model = Summarizer()
    monkeypatch.setattr(tools_mod, "init_model", lambda config=None: model)
    return model


def scrape(url: str) -> str:
    return asyncio.run(tools_mod.scrape_website(url, state=STATE, config={}))


def refused_url() -> str:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens on this port now
    return f"http://127.0.0.1:{port}/about"


def test_a_page_is_summarized(site: Any, summarizer: Summarizer) -> None:
    assert scrape(site.url) == "notes"
    assert len(summarizer.prompts) == 1


@pytest.mark.parametrize("status", [403, 404, 503])
def test_an_http_error_page_is_not_summarized(
    site: Any, summarizer: Summarizer, status: int
) -> None:
    site.status, site.body = status, f"<h1>{status} error page</h1>".encode()
    result = scrape(site.url)
    assert result == f"Could not scrape {site.url}: the site returned HTTP {status}."
    assert summarizer.prompts == []


def test_a_refused_connection_is_reported(summarizer: Summarizer) -> None:
    url = refused_url()
    result = scrape(url)
    assert result.startswith(f"Could not scrape {url}: ClientConnector")
    assert summarizer.prompts == []


def test_a_dropped_connection_is_reported(site: Any, summarizer: Summarizer) -> None:
    site.disconnect = True
    result = scrape(site.url)
    assert result.startswith(f"Could not scrape {site.url}:")
    assert summarizer.prompts == []


def test_a_page_that_never_answers_times_out(
    site: Any, summarizer: Summarizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools_mod, "SCRAPE_TIMEOUT_SECONDS", 0.5)
    site.hang = threading.Event()
    started = time.monotonic()
    result = scrape(site.url)
    assert time.monotonic() - started < 5
    assert result.startswith(f"Could not scrape {site.url}:")
    assert "Timeout" in result
    assert summarizer.prompts == []
