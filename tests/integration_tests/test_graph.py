import pytest

from enrichment_agent import graph

template_schema = {
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
async def test_researcher_simple_runthrough() -> None:
    res = await graph.ainvoke(
        {
            "topic": "LangChain",
            "template_schema": template_schema,
        }
    )

    assert res["info"] is not None
    assert "harrison" in res["info"]["founder"].lower()
