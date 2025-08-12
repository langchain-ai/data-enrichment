"""Define the runtime context information for the agent."""

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from enrichment_agent import prompts


@dataclass(kw_only=True)
class Context:
    """The runtime context for the enrichment agent."""

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Should be in the form: provider/model-name."
        },
    )

    prompt: str = field(
        default=prompts.MAIN_PROMPT,
        metadata={
            "description": "The main prompt template to use for the agent's interactions. "
            "Expects two f-string arguments: {info} and {topic}."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    max_info_tool_calls: int = field(
        default=3,
        metadata={
            "description": "The maximum number of times the Info tool can be called during a single interaction."
        },
    )

    max_loops: int = field(
        default=6,
        metadata={
            "description": "The maximum number of interaction loops allowed before the agent terminates."
        },
    )

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))
