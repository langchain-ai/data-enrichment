"""Tools for data enrichment.

This module contains functions that are directly exposed to the LLM as tools.
These tools can be used for tasks such as web searching and scraping.
Users can edit and extend these tools as needed.
"""

import json
from typing import Any, Optional, cast

import aiohttp
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated

from enrichment_agent.configuration import Configuration
from enrichment_agent.state import State
from enrichment_agent.utils import init_model


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Perform a general web search using the Tavily search engine.

    This asynchronous function executes the following steps:
    1. Extracts configuration from the provided RunnableConfig.
    2. Initializes a TavilySearchResults object with a maximum number of results.
    3. Invokes the Tavily search with the given query.
    4. Returns the search results as a list of dictionaries.

    Args:
        query (str): The search query string.
        config (RunnableConfig): Configuration object containing search parameters.

    Returns:
        Optional[list[dict[str, Any]]]: A list of search result dictionaries, or None if the search fails.
        Each dictionary typically contains information like title, url, content snippet, etc.

    Note:
        This function uses the Tavily search engine, which is designed for comprehensive
        and accurate results, particularly useful for current events and factual queries.
        The maximum number of results is determined by the configuration.
    """
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


async def scrape_website(
    url: str,
    *,
    state: Annotated[State, InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """Scrape and summarize content from a given URL.

    This asynchronous function performs the following steps:
    1. Fetches the content of the specified URL.
    2. Formats a prompt using the fetched content and the extraction schema from the state.
    3. Initializes a language model using the provided configuration.
    4. Invokes the model with the formatted prompt to summarize the content.

    Args:
        url (str): The URL of the website to scrape.
        state (State): Injected state containing the extraction schema.
        config (RunnableConfig): Configuration for initializing the language model.

    Returns:
        str: A summary of the scraped content, tailored to the extraction schema.

    Note:
        The function uses aiohttp for asynchronous HTTP requests and assumes the
        existence of a _INFO_PROMPT template and an init_model function.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.text()

    p = _INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        url=url,
        content=content,
    )
    raw_model = init_model(config)
    result = await raw_model.ainvoke(p)
    return str(result.content)
