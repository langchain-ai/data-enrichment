import json
from typing import List, Literal, Optional, cast, Dict, Any

from enrichment_agent.configuration import Configuration
from enrichment_agent.state import State, InputState, OutputState, Info
from enrichment_agent.utils import init_model
from enrichment_agent.tools import scrape_website, search
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

main_prompt = """You are doing web research on behalf of a user. You are trying to figure out this information:

<info>
{info}
</info>

You have access to the following tools:

- `Search`: call a search tool and get back some results
- `ScrapeWebsite`: scrape a website and get relevant notes about the given request. This will update the notes above.
- `Info`: call this when you are done and have gathered all the relevant info

Here is the information you have about the topic you are researching:

Topic: {topic}"""


# Define the nodes


async def call_model(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    template_schema = Info.schema_json()

    p = main_prompt.format(
        info=json.dumps(template_schema, indent=2), topic=state["topic"]
    )
    messages = [HumanMessage(content=p)] + state["messages"]
    raw_model = init_model(config)

    model = raw_model.bind_tools([scrape_website, search, Info])
    response = cast(AIMessage, await model.ainvoke(messages))

    info = None
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "Info":
                info = tool_call["args"]
                break

    return {"messages": [response], "info": info}


class InfoIsSatisfactory(BaseModel):
    reason: List[str] = Field(
        description="First, provide reasoning for why this is either good or bad as a final result. Must include at least 3 reasons."
    )
    is_satisfactory: bool = Field(
        description="After providing your reasoning, provide a value indicating whether the result is satisfactory. If not, you will continue researching."
    )


async def call_checker(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:

    template_schema = Info.schema_json()
    p = main_prompt.format(
        info=json.dumps(template_schema, indent=2), topic=state["topic"]
    )
    messages = [HumanMessage(content=p)] + state["messages"][:-1]
    presumed_info = state["info"]
    checker_prompt = """I am thinking of calling the info tool with the info below. \
Is this good? Give your reasoning as well. \
You can encourage the Assistant to look at specific URLs if that seems relevant, or do more searches.
If you don't think it is good, you should be very specific about what could be improved.

{presumed_info}"""
    p1 = checker_prompt.format(presumed_info=json.dumps(presumed_info or {}, indent=2))
    messages.append(HumanMessage(content=p1))
    raw_model = init_model(config)
    bound_model = raw_model.with_structured_output(InfoIsSatisfactory)
    response = cast(InfoIsSatisfactory, await bound_model.ainvoke(messages))
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"{call_checker.__name__} expects the last message in the state to be an AI message with tool calls."
            f" Got: {type(last_message)}"
        )

    if response.is_satisfactory:
        try:
            return {"info": presumed_info}
        except Exception as e:
            return {
                "messages": [
                    ToolMessage(
                        tool_call_id=last_message.tool_calls[0]["id"],
                        content=f"Invalid response: {e}",
                        name="Info",
                    )
                ]
            }
    else:
        return {
            "messages": [
                ToolMessage(
                    tool_call_id=last_message.tool_calls[0]["id"],
                    content=str(response),
                    name="Info",
                    additional_kwargs={"artifact": response.dict()},
                )
            ]
        }


def create_correction_response(state: State) -> Dict[str, List[BaseMessage]]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return {
            "messages": [
                ToolMessage(
                    tool_call_id=call["id"],
                    content="You must call one, and only one, tool!",
                    name=call["name"],
                )
                for call in last_message.tool_calls
            ]
        }
    return {
        "messages": [
            HumanMessage(
                content="You must call one, and only one, tool! You can call the `Info` tool to finish the task."
            )
        ]
    }


def route_after_agent(
    state: State,
) -> Literal["create_correction_response", "call_checker", "tools", "__end__"]:
    last_message = state["messages"][-1]
    num_rounds = sum(
        1 for m in state["messages"] if isinstance(m, ToolMessage) and m.name == "Info"
    )

    if (
        not isinstance(last_message, AIMessage)
        or not last_message.tool_calls
        or (
            len(last_message.tool_calls) != 1
            and any(tc["name"] == "Info" for tc in last_message.tool_calls)
        )
    ):
        return "create_correction_response"
    elif last_message.tool_calls[0]["name"] == "Info":
        if num_rounds > 2:
            return "__end__"
        return "call_checker"
    else:
        return "tools"


def route_after_checker(state: State) -> Literal["__end__", "call_model"]:
    if state["info"]:
        return "__end__"
    return "call_model"


# Create the graph
workflow = StateGraph(
    State, input=InputState, output=OutputState, config_schema=Configuration
)
workflow.add_node(call_model)
workflow.add_node(call_checker)
workflow.add_node(create_correction_response)
workflow.add_node(ToolNode([search, scrape_website]))
workflow.add_edge("__start__", "call_model")
workflow.add_conditional_edges("call_model", route_after_agent)
workflow.add_edge("tools", "call_model")
workflow.add_conditional_edges("call_checker", route_after_checker)
workflow.add_edge("create_correction_response", "call_model")

graph = workflow.compile()
graph.name = "ResearchTopic"
