import json
from typing import Any, Optional, cast
from typing_extensions import Annotated

import aiohttp
from enrichment_agent.configuration import Configuration
from enrichment_agent.utils import init_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import InjectedToolArg
from langgraph.prebuilt import InjectedState
from langchain_core.runnables import RunnableConfig
from enrichment_agent.state import State, InputState, OutputState, Info
from langchain_core.runnables import chain


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """A search engine optimized for comprehensive, accurate, and trusted results.
    Useful for when you need to answer questions about current events.
    Input should be a search query."""
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


_INFO_PROMPT = """You are doing web research on behalf of a user. You are trying to find out this information:

<info>
{info}
</info>

You just scraped the following website: {url}

Based on the website content below, jot down some notes about the website.

<Website content>
{content}
</Website content>"""

@chain
async def extract(state, config):

    raw_model = init_model(config)
    p = _INFO_PROMPT.format(
        info=state['info'],
        url=state['url'],
        content=state['content'],
    )
    result = await raw_model.ainvoke(p)
    return str(result.content)



async def scrape_website(
    url: str,
    *,
    state: Annotated[dict[str, Any], InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """Return a natural language response from the provided URL"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.text()

    template_schema = Info.schema_json()

    return await extract.ainvoke(
        {
        "info":json.dumps(template_schema, indent=2),
        "url":url,
        "content":content,},
        config=config
    )
