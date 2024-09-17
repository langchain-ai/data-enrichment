from typing import Any, Dict

import pytest
from langsmith import unit

from enrichment_agent import graph


@pytest.fixture(scope="function")
def extraction_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "founder": {
                "type": "string",
                "description": "The name of the company founder.",
            },
            "websiteUrl": {
                "type": "string",
                "description": "Website URL of the company, e.g.: https://openai.com/, or https://microsoft.com",
            },
            "products_sold": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of products sold by the company.",
            },
        },
        "required": ["founder", "websiteUrl", "products_sold"],
    }


@pytest.mark.asyncio
@unit
async def test_researcher_simple_runthrough(extraction_schema: Dict[str, Any]) -> None:
    res = await graph.ainvoke(
        {
            "topic": "LangChain",
            "extraction_schema": extraction_schema,
        }
    )

    assert res["info"] is not None
    assert "harrison" in res["info"]["founder"].lower()


@pytest.fixture(scope="function")
def array_extraction_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "providers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Company name"},
                        "technology_summary": {
                            "type": "string",
                            "description": "Brief summary of their chip technology for LLM training",
                        },
                        "current_market_share": {
                            "type": "string",
                            "description": "Estimated current market share percentage or position",
                        },
                        "future_outlook": {
                            "type": "string",
                            "description": "Brief paragraph on future prospects and developments",
                        },
                    },
                    "required": [
                        "name",
                        "technology_summary",
                        "current_market_share",
                        "future_outlook",
                    ],
                },
                "description": "List of top chip providers for LLM Training",
            },
            "overall_market_trends": {
                "type": "string",
                "description": "Brief paragraph on general trends in the LLM chip market",
            },
        },
        "required": ["providers", "overall_market_trends"],
    }


@pytest.mark.asyncio
@unit
async def test_researcher_list_type(array_extraction_schema: Dict[str, Any]) -> None:
    res = await graph.ainvoke(
        {
            "topic": "Top 5 chip providers for LLM training",
            "extraction_schema": array_extraction_schema,
        }
    )
    # Check that nvidia is amongst them lol
    info = res["info"]
    assert "providers" in info
    assert isinstance(res["info"]["providers"], list)
    assert len(info["providers"]) == 5  # Ensure we got exactly 5 providers

    # Check for NVIDIA's presence
    nvidia_present = any(
        provider["name"].lower().strip() == "nvidia" for provider in info["providers"]
    )
    assert (
        nvidia_present
    ), "NVIDIA should be among the top 5 chip providers for LLM training"

    # Validate structure of each provider
    for provider in info["providers"]:
        assert "name" in provider
        assert "technology_summary" in provider
        assert "current_market_share" in provider
        assert "future_outlook" in provider

    # Check for overall market trends
    assert "overall_market_trends" in info
    assert isinstance(info["overall_market_trends"], str)
    assert len(info["overall_market_trends"]) > 0
