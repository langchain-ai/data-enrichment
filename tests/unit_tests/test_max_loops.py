"""max_loops bounds the research loop, whatever the model does.

Uses a scripted model and a page served on 127.0.0.1, so no model API calls or
credentials are needed.
"""

import asyncio
import http.server
import importlib
import threading
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from langchain_core.messages import AIMessage

# The package re-exports the compiled `graph`, which shadows the submodule name.
graph_mod = importlib.import_module("enrichment_agent.graph")
tools_mod = importlib.import_module("enrichment_agent.tools")

INPUT = {
    "topic": "Example Corp",
    "extraction_schema": {
        "type": "object",
        "properties": {"founded": {"type": "string"}},
    },
}


class _Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        data = b"<p>Founded 1999.</p>"
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def page_url() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/about"
    finally:
        server.shutdown()
        server.server_close()


class ScriptedModel:
    """Answers every agent call with ``reply()``; summarizes pages as "notes".

    Raises after ``budget`` agent calls, so a loop that max_loops fails to stop
    ends the test quickly instead of running to the recursion limit.
    """

    def __init__(self, reply: Callable[[], AIMessage], budget: int = 40) -> None:
        self.reply, self.budget = reply, budget
        self.agent_calls = 0
        self.summaries = 0
        self.checker_calls = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        outer = self

        class Bound:
            async def ainvoke(self, messages: Any) -> AIMessage:
                outer.agent_calls += 1
                if outer.agent_calls > outer.budget:
                    raise RuntimeError(
                        f"{outer.budget} agent calls: max_loops did not stop the run"
                    )
                return outer.reply()

        return Bound()

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:  # the checker
        outer = self

        class Checker:
            async def ainvoke(self, messages: Any) -> Any:
                outer.checker_calls += 1
                return schema(
                    reason=["complete", "sourced", "consistent"],
                    is_satisfactory=True,
                    improvement_instructions=None,
                )

        return Checker()

    async def ainvoke(self, prompt: Any) -> AIMessage:  # scrape_website's summarizer
        self.summaries += 1
        return AIMessage(content="notes")


@pytest.mark.parametrize("behaviour", ["keeps-researching", "never-calls-a-tool"])
def test_max_loops_bounds_the_research_loop(
    monkeypatch: pytest.MonkeyPatch, page_url: str, behaviour: str
) -> None:
    def scrape() -> AIMessage:
        call = {
            "name": "scrape_website",
            "args": {"url": page_url},
            "id": uuid.uuid4().hex,
        }
        return AIMessage(content="", tool_calls=[call])

    def text_only() -> AIMessage:
        return AIMessage(content="I believe it was founded in 1999.")

    model = ScriptedModel(scrape if behaviour == "keeps-researching" else text_only)
    monkeypatch.setattr(graph_mod, "init_model", lambda config=None: model)
    monkeypatch.setattr(tools_mod, "init_model", lambda config=None: model)

    asyncio.run(
        graph_mod.graph.ainvoke(INPUT, config={"configurable": {"max_loops": 2}})
    )

    # Execution order with max_loops=2: agent call 1 -> (keeps-researching: scrape
    # and one page summary) -> agent call 2 -> loop_step is 2, so the graph ends
    # without running the second call's tool.
    assert model.agent_calls == 2
    assert model.summaries == (1 if behaviour == "keeps-researching" else 0)


def test_info_on_the_last_allowed_step_still_reaches_the_checker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: max_loops must not cut off an Info submission."""
    replies = iter(
        [
            AIMessage(content="Let me think."),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "Info", "args": {"founded": "1999"}, "id": "info-1"}
                ],
            ),
        ]
    )
    model = ScriptedModel(lambda: next(replies))
    monkeypatch.setattr(graph_mod, "init_model", lambda config=None: model)
    monkeypatch.setattr(tools_mod, "init_model", lambda config=None: model)

    result = asyncio.run(
        graph_mod.graph.ainvoke(INPUT, config={"configurable": {"max_loops": 2}})
    )

    # Agent call 2 submits Info when loop_step reaches max_loops; it still goes to
    # the checker, which accepts it, so the run returns that info.
    assert model.agent_calls == 2
    assert model.checker_calls == 1
    assert result["info"] == {"founded": "1999"}
