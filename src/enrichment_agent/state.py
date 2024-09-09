from enum import Enum
from typing import Any, List, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class InputState(TypedDict):
    topic: Optional[str]
    info: Optional[dict[str, Any]]  # This is primarily populated by the agent
    "The info state tracks the current extracted data for the given topic, conforming to the provided schema."


class InfoLocation(BaseModel):
    city: str = Field(description="city in which the company is located")
    state: str = Field(description="state in which the company is located")
    country: str = Field(description="country in which the company is located")

class OrganizationType(str, Enum):
    private = "private"
    public = "public"
    non_profit = "non-profit"
    government = "government"
    educational = "educational"
    other = "other"


class QualityInfo(BaseModel):
    quality_reasoning: str = Field(description="Reasoning about the quality of the company as a partner. Base this on how legitimate the company is, how much usage they have, how they are viewed by peers, how fast-moving they are, and how cutting-edge their technology is. This should be a subjective score that would inform a rising startup how much they should consider partnering with this company.")
    quality_score: int = Field(description="quality of the company as a partner on a scale of 1-10. Base this on how legitimate the company is, how much usage they have, how they are viewed by peers, how fast-moving they are, and how cutting-edge their technology is. This should be a subjective score that would inform a rising startup how much they should consider partnering with this company.")


class Info(BaseModel):
    """Information to """
    ceo: str = Field(description="CEO of the company")
    organization_type: OrganizationType = Field(description="type of organization - public, private, educational, etc.")
    revenue: int | None = Field(description="revenue in millions, or null if this is not public information")
    name: str = Field(description="Full name of the company")
    description: str = Field(description="description of the company - what they do")
    industry: str = Field(description="industry the company is in")
    size: int = Field(description="number of employees")
    funding: int | None = Field(description="Total raised funding in millions, if the company is private")
    location: InfoLocation = Field(description="location of the company")
    investors: List[str] | None = Field(description="list of investors: for example, VC firms or angels")
    quality_info: QualityInfo = Field(description="quality of the company as a partner on a scale of 1-10. Base this on how legitimate the company is, how much usage they have, how they are viewed by peers, how fast-moving they are, and how cutting-edge their technology is. This should be a subjective score that would inform a rising startup how much they should consider partnering with this company.")



class State(InputState):
    """
    A graph's STate defines three main things:
    1. The structure of the data to be passed between nodes (which "channels" to read from/write to and their types)
    2. Default values for each field
    3. Reducers for the state's fields. Reducers are functions that determine how to apply updates to the state.
    See [Reducers](https://langchain-ai.github.io/langgraphjs/concepts/low_level/#reducers) for more information.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    """
    Messages track the primary execution state of the agent.

    Typically accumulates a pattern of:

    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
        information
    3. ToolMessage(s) - the responses (or errors) from the executed tools

        (... repeat steps 2 and 3 as needed ...)
    4. AIMessage without .tool_calls - agent responding in unstructured
        format to the user.

    5. HumanMessage - user responds with the next conversational turn.

        (... repeat steps 2-5 as needed ... )

    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`.
        """

    # Feel free to add additional attributes to your state as needed.
    # Common examples include retrieved documents, extracted entities, API connections, etc.


class OutputState(TypedDict):
    """
    Represents the subset of the graph's state that is returned to the end user.

    This class defines the structure of the output that will be provided
    to the user after the graph's execution is complete.
    """

    info: dict[str, Any]
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """
