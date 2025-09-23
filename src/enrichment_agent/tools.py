"""Tools for data enrichment.

This module contains functions that are directly exposed to the LLM as tools.
These tools can be used for tasks such as web searching and scraping.
Users can edit and extend these tools as needed.
"""

import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated, List

from pydantic import BaseModel, Field

from enrichment_agent.configuration import Configuration
from enrichment_agent.state import State
from enrichment_agent.utils import init_model, deduplicate_and_format_sources

from tavily import AsyncTavilyClient

# Schemas
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )   

# Instructions
query_writer_instructions = """You are a search query generator tasked with creating diverse but related search queries based on an initial query.

For the initial query: {initial_query}

Generate distinct search queries that:
1. Cover different aspects or angles of the topic
2. Are mutually exclusive to avoid redundant results
3. Help gather comprehensive information about the subject
4. Use different phrasings and keywords to maximize coverage
5. Maintain relevance to the original query intent

Each query should focus on a unique aspect while staying connected to the main topic."""

_INFO_PROMPT = """You are doing web research on behalf of a user. You need to extract specific information based on this schema:

<schema>
{info}
</schema>

You have just scraped website content. Review the content below and take detailed notes that align with the extraction schema above. 

Focus only on information that matches the schema requirements.

<Website contents>
{content}
</Website contents>

Please provide well structured notes that:
1. Map directly to the schema fields
2. Include only relevant information from the content
3. Maintain the original facts and data points
4. Note any missing schema fields that weren't found in the content"""

async def perform_web_research(
    query: str, 
    *, 
    state: Annotated[State, InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Execute a multi-step web search and information extraction process.

        This function performs the following steps:
        1. Generates multiple search queries based on the input query
        2. Executes concurrent web searches using the Tavily API
        3. Deduplicates and formats the search results
        4. Extracts structured information based on the provided schema

        Args:
            query: The initial search query string
            state: Injected application state containing the extraction schema
            config: Runtime configuration for the search process

        Returns:
            str: Structured notes from the search results that are
             relevant to the extraction schema in state.extraction_schema

        Note:
            The function uses concurrent execution for multiple search queries to improve
            performance and combines results from various sources for comprehensive coverage.
    """
    configuration = Configuration.from_runnable_config(config)

    # Generate search queries
    raw_model = init_model(config)
    structured_llm = raw_model.with_structured_output(Queries)
    
    # Format system instructions
    query_instructions = query_writer_instructions.format(initial_query=query)

    # Generate queries  
    results = structured_llm.invoke([SystemMessage(content=query_instructions)]+[HumanMessage(content=f"Please generate {configuration.num_search_queries} search queries.")])

    # Search client
    tavily_async_client = AsyncTavilyClient()

    # Web search
    query_list = [query.search_query for query in results.queries]
    search_tasks = []
    for query in query_list:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=configuration.max_search_results,
                    include_raw_content=True,
                    topic="general"
                )
            )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(search_docs, max_tokens_per_source=1000, include_raw_content=True)

    # Generate structured notes relevant to the extraction schema
    p = _INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        content=source_str,
    )
    result = await raw_model.ainvoke(p)
    return str(result.content)