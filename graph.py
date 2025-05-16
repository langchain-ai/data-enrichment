import json
import pathlib
import argparse # For parsing command-line arguments
from typing import List
from jinja2 import Environment, FileSystemLoader

from langchain_core.messages import HumanMessage # Add this import
from langchain_openai import ChatOpenAI # Add this import
from dotenv import load_dotenv # Add this import
import pypdfium2 as pdfium # Add this import
import uuid # Add this import

from langgraph.graph import StateGraph, END
from dgh_state import DGHState
from refinery_agent.load_source import load_source
from refinery_agent.nodes_stub import (
    extract_data as extract_data_node, # USE the one from nodes_stub.py
    diff_roles as diff_roles_node,           # Corrected import
    save_gold_example as save_gold_example_node, # Corrected import
)
from refinery_agent.nodes_validate import validate_schema_node

# Load settings and schema
SETTINGS = json.load(open("config/refinery.json"))
SCHEMA = json.load(open(SETTINGS["schema_path"]))
load_dotenv(dotenv_path=pathlib.Path(".devcontainer/.env")) # Load environment variables from specific path

# --- Define Nodes ---
# Placeholder for validate_schema node - THIS IS REMOVED as we import the real one
# def validate_schema_node(state: DGHState) -> DGHState:
#     # In a real scenario, this node would validate input_docs against the schema
#     # or prepare data for extraction based on the schema.
#     print("Executing validate_schema_node (placeholder)")
#     # For now, just pass the state through
#     return state

# --- Graph Definition ---
workflow = StateGraph(DGHState)

# Add nodes to the graph
workflow.add_node("load_source", load_source)
workflow.add_node("extract_data", extract_data_node)
workflow.add_node("validate_output", validate_schema_node)
workflow.add_node("diff_roles", diff_roles_node) # Uses imported and aliased diff_roles
workflow.add_node("save_gold_example", save_gold_example_node) # Uses imported and aliased save_gold_example
# Potentially add chunk_document_node and final_result_node if they are part of the flow
# workflow.add_node("chunk_document", chunk_document_node)
# workflow.add_node("final_result", final_result_node)

# --- Wire the graph ---
# Set the entry point
# Assuming the flow starts with extract_data now, or a new entry point if chunking is first.
# For this specific set of changes, the prompt implies extract_data is effectively the start after initial setup.
# If there's an initial validation or chunking step that should be the entry point, adjust this.
# The original prompt had validate_schema -> diff_roles.
# The new prompt has extract_data -> validate_output -> diff_roles.
# We need an entry point. If there's no explicit "load" or "start" node that feeds `extract_data`,
# then `extract_data` might be considered the first processing step after state initialization.
# Let's assume for now the entry point should be extract_data as per the new flow.
# If a different entry point is needed (e.g. a node that loads data into input_docs), that should be specified.
# The previous `workflow.set_entry_point("validate_schema")` and `workflow.add_edge("validate_schema", "extract_data")`
# suggest there was an initial step. Given the removal of the placeholder `validate_schema_node` that was the entry point,
# and introduction of `validate_output` *after* `extract_data`, we will set `extract_data` as entry point.

workflow.set_entry_point("load_source") # CHANGE THIS LINE

# Add edges as per instructions
# The old edge was: graph.add_edge("validate_schema", "diff_roles")
# The new edges are:
# graph.add_edge("extract_data", "validate_output")
# graph.add_edge("validate_output", "diff_roles")

# Remove old edge if it existed with the placeholder name
# workflow.remove_edge("validate_schema", "diff_roles") # LangGraph does not have a remove_edge

# Add new edges
workflow.add_edge("load_source", "extract_data")
workflow.add_edge("extract_data", "validate_output")
workflow.add_edge("validate_output", "diff_roles")
workflow.add_edge("diff_roles", "save_gold_example")
workflow.add_edge("save_gold_example", END)

# Compile the graph
app = workflow.compile()

# --- Main execution block ---
def main(files: List[str]):
    # Initial state sets:
    initial_state = DGHState(
        input_docs=files,
        extraction_schema=SCHEMA,
        extracted_json=None,
        diff_json=None,
        approved=None,
        saved_location=None,
        status="to-parse",
    )

    print(f"Initial state: {initial_state}")
    result = app.invoke(initial_state)
    payload = json.dumps(result, indent=2)
    print("GRAPH_JSON_START")
    print(payload)
    print("GRAPH_JSON_END")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Refinery graph.")
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="List of input document paths to process."
    )
    args = parser.parse_args()
    main(args.files) 