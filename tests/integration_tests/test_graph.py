import pytest
from enrichment_agent import graph
from pydantic import BaseModel, Field


class EnrichmentSchema(BaseModel):
    founder: str = Field(description="The name of the company founder.")
    websiteUrl: str = Field(
        description="Website URL of the company, e.g.: https://openai.com/, or https://microsoft.com"
    )


@pytest.mark.asyncio
async def test_researcher_simple_runthrough():
    res = await graph.ainvoke(
        {
            "topic": "LangChain",
            "template_schema": EnrichmentSchema.model_json_schema(),
        }
    )

    assert res["info"] is not None
    assert "harrison" in res["info"]["founder"].lower()
