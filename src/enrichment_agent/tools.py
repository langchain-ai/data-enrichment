"""Tools for data enrichment.

This module contains functions that are directly exposed to the LLM as tools.
These tools can be used for tasks such as web searching and scraping.
Users can edit and extend these tools as needed.
"""

import asyncio
import json
from typing import Any, Optional, cast

import aiohttp
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langchain_tavily import TavilySearch
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated

from enrichment_agent.configuration import Configuration
from enrichment_agent.state import State
from enrichment_agent.utils import init_model


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Query a search engine.

    This function queries the web to fetch comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events. Provide as much context in the query as needed to ensure high recall.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearch(max_results=configuration.max_search_results)
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


SCRAPE_TIMEOUT_SECONDS = 30.0
"""Total time allowed for fetching one page (aiohttp's default is 5 minutes)."""


async def scrape_website(
    url: str,
    *,
    state: Annotated[State, InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """Scrape and summarize content from a given URL.

    Returns:
        str: A summary of the scraped content, tailored to the extraction schema.
    """
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=SCRAPE_TIMEOUT_SECONDS)) as session:
            async with session.get(url) as response:
                if response.status >= 400:
                    return f"Could not scrape {url}: the site returned HTTP {response.status}."
                content = await response.text()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        # Report the failure to the model instead of ending the whole run.
        return f"Could not scrape {url}: {type(e).__name__}: {e}"

    p = _INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        url=url,
        content=content[:40_000],
    )
    raw_model = init_model(config)
    result = await raw_model.ainvoke(p)
    return str(result.content)
