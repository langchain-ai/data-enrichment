"""Define a data enrichment agent.

Works with a chat model with tool calling support.
"""

import json
from typing import Any, Dict, List, Literal, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from enrichment_agent import prompts
from enrichment_agent.configuration import Configuration
from enrichment_agent.state import InputState, OutputState, State
from enrichment_agent.tools import scrape_website, search
from enrichment_agent.utils import init_model

# Define the nodes
async def call_agent_model(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Call the primary Language Model (LLM) to decide on the next research action.

    This asynchronous function performs the following steps:
    1. Initializes configuration and sets up the 'Info' tool, which is the user-defined extraction schema.
    2. Prepares the prompt and message history for the LLM.
    3. Initializes and configures the LLM with available tools.
    4. Invokes the LLM and processes its response.
    5. Handles the LLM's decision to either continue research or submit final info.

    Args:
        state (State): The current state of the research process, including topic and extraction schema.
        config (Optional[RunnableConfig]): Configuration for the LLM, if provided.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'messages': List of response messages from the LLM.
            - 'info': Extracted information if the LLM decided to submit final info, else None.
            - 'loop_step': Incremented step count for the research loop.

    Note:
        - The function uses three tools: scrape_website, search, and a dynamic 'Info' tool.
        - If the LLM calls the 'Info' tool, it's considered as submitting the final answer.
        - If the LLM doesn't call any tool, a prompt to use a tool is appended to the messages.
    """

    # Load configuration from the provided RunnableConfig
    configuration = Configuration.from_runnable_config(config)

    # Define the 'Info' tool, which is the user-defined extraction schema
    info_tool = {
        "name": "Info",
        "description": "Call this when you have gathered all the relevant info",
        "parameters": state.extraction_schema,
    }

    # Format the prompt defined in prompts.py with the extraction schema and topic
    p = configuration.prompt.format(
        info=json.dumps(state.extraction_schema, indent=2), topic=state.topic
    )

    # Create the messages list with the formatted prompt and the previous messages
    messages = [HumanMessage(content=p)] + state.messages

    # Initialize the raw model with the provided configuration and bind the tools
    raw_model = init_model(config)
    model = raw_model.bind_tools([scrape_website, search, info_tool], tool_choice="any")
    response = cast(AIMessage, await model.ainvoke(messages))

    # Initialize info to None
    info = None

    # Check if the response has tool calls
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "Info":
                info = tool_call["args"]
                break
    if info is not None:
        # The agent is submitting their answer;
        # ensure it isnt' erroneously attempting to simultaneously perform research
        response.tool_calls = [
            next(tc for tc in response.tool_calls if tc["name"] == "Info")
        ]
    response_messages: List[BaseMessage] = [response]
    if not response.tool_calls:  # If LLM didn't respect the tool_choice
        response_messages.append(
            HumanMessage(content="Please respond by calling one of the provided tools.")
        )
    return {
        "messages": response_messages,
        "info": info,
        # Add 1 to the step count
        "loop_step": 1,
    }

class InfoIsSatisfactory(BaseModel):
    """Validate whether the current extracted info is satisfactory and complete."""

    reason: List[str] = Field(
        description="First, provide reasoning for why this is either good or bad as a final result. Must include at least 3 reasons."
    )
    is_satisfactory: bool = Field(
        description="After providing your reasoning, provide a value indicating whether the result is satisfactory. If not, you will continue researching."
    )


async def reflect(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Validate the quality of the data enrichment agent's output.

    This asynchronous function performs the following steps:
    1. Prepares the initial prompt using the main prompt template.
    2. Constructs a message history for the model.
    3. Prepares a checker prompt to evaluate the presumed info.
    4. Initializes and configures a language model with structured output.
    5. Invokes the model to assess the quality of the gathered information.
    6. Processes the model's response and determines if the info is satisfactory.

    Args:
        state (State): The current state of the research process, including topic,
                       extraction schema, and gathered information.
        config (Optional[RunnableConfig]): Configuration for the language model, if provided.

    Returns:
        Dict[str, Any]: A dictionary containing either:
            - 'info': The presumed info if it's deemed satisfactory.
            - 'messages': A list with a ToolMessage indicating an error or unsatisfactory result.

    Raises:
        ValueError: If the last message in the state is not an AIMessage with tool calls.

    Note:
        - This function acts as a quality check for the information gathered by the agent.
        - It uses a separate language model invocation to critique the gathered info.
        - The function can either approve the gathered info or request further research.
    """
    p = prompts.MAIN_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), topic=state.topic
    )
    messages = [HumanMessage(content=p)] + state.messages[:-1]
    presumed_info = state.info
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
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"{reflect.__name__} expects the last message in the state to be an AI message with tool calls."
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
                        status="error",
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
                    status="error",
                )
            ]
        }


def route_after_agent(
    state: State,
) -> Literal["reflect", "tools", "call_agent_model", "__end__"]:
    """
    Schedule the next node after the agent's action.

    This function determines the next step in the research process based on the
    last message in the state. It handles three main scenarios:

    1. Error recovery: If the last message is unexpectedly not an AIMessage.
    2. Info submission: If the agent has called the "Info" tool to submit findings.
    3. Continued research: If the agent has called any other tool.

    Args:
        state (State): The current state of the research process, including
                       the message history.

    Returns:
        Literal["reflect", "tools", "call_agent_model", "__end__"]: 
            - "reflect": If the agent has submitted info for review.
            - "tools": If the agent has called a tool other than "Info".
            - "call_agent_model": If an unexpected message type is encountered.

    Note:
        - The function assumes that normally, the last message should be an AIMessage.
        - The "Info" tool call indicates that the agent believes it has gathered
          sufficient information to answer the query.
        - Any other tool call indicates that the agent needs to continue research.
        - The error recovery path (returning "call_agent_model" for non-AIMessages)
          serves as a safeguard against unexpected states.
    """
    last_message = state.messages[-1]

    # "If for some reason the last message is not an AIMessage (due to a bug or unexpected behavior elsewhere in the code), 
    # it ensures the system doesn't crash but instead tries to recover by calling the agent model again.
    if not isinstance(last_message, AIMessage):
        return "call_agent_model"
    # If the "Into" tool was called, then the model provided its extraction output. Reflect on the result
    if last_message.tool_calls and last_message.tool_calls[0]["name"] == "Info":
        return "reflect"
    # The last message is a tool call that is not "Info" (extraction output)
    else:
        return "tools"

def route_after_checker(
    state: State, config: RunnableConfig
) -> Literal["__end__", "call_agent_model"]:
    """
    Schedule the next node after the checker's evaluation.

    This function determines whether to continue the research process or end it
    based on the checker's evaluation and the current state of the research.

    Args:
        state (State): The current state of the research process, including
                       the message history, info gathered, and loop step count.
        config (RunnableConfig): Configuration object containing settings like
                                 the maximum number of allowed loops.

    Returns:
        Literal["__end__", "call_agent_model"]: 
            - "__end__": If the research process should terminate.
            - "call_agent_model": If further research is needed.

    The function makes decisions based on the following criteria:
    1. If the maximum number of loops has been reached, it ends the process.
    2. If no info has been gathered yet, it continues the research.
    3. If the last message indicates an error or unsatisfactory result, it continues the research.
    4. If none of the above conditions are met, it assumes the result is satisfactory and ends the process.

    Note:
        - The function relies on a Configuration object derived from the RunnableConfig.
        - It checks the loop_step against max_loops to prevent infinite loops.
        - The presence of info and its quality (determined by the checker) influence the decision.
        - An error status in the last message triggers additional research.
    """
    configurable = Configuration.from_runnable_config(config)
    last_message = state.messages

    if state.loop_step < configurable.max_loops:
        if not state.info:
            return "call_agent_model"
        if isinstance(last_message, ToolMessage) and last_message.status == "error":
            # Research deemed unsatisfactory
            return "call_agent_model"
        # It's great!
        return "__end__"
    else:
        return "__end__"

# Create the graph
workflow = StateGraph(
    State, input=InputState, output=OutputState, config_schema=Configuration
)
workflow.add_node(call_agent_model)
workflow.add_node(reflect)
workflow.add_node("tools", ToolNode([search, scrape_website]))
workflow.add_edge("__start__", "call_agent_model")
workflow.add_conditional_edges("call_agent_model", route_after_agent)
workflow.add_edge("tools", "call_agent_model")
workflow.add_conditional_edges("reflect", route_after_checker)

graph = workflow.compile()
graph.name = "ResearchTopic"
