from langchain_core.tools import tool
from uuid import uuid4
from dgh_state import DGHState
from langsmith import Client as LSClient
from langsmith.utils import LangSmithNotFoundError
# from langsmith import LangSmithNotFoundError # Previous attempt
# import requests # Import requests to catch HTTPError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from jsonschema import validate, ValidationError
import json # Added import

def _dict_diff(a: dict, b: dict, path="") -> list[dict]:
    diffs = []
    for k in a.keys() | b.keys():
        pa = f"{path}.{k}" if path else k
        if k not in b:
            diffs.append({"field": pa, "change": "removed", "old": a[k]})
        elif k not in a:
            diffs.append({"field": pa, "change": "added", "new": b[k]})
        elif a[k] != b[k]:
            if isinstance(a[k], dict) and isinstance(b[k], dict):
                diffs += _dict_diff(a[k], b[k], pa)
            else:
                diffs.append({"field": pa, "change": "modified",
                              "old": a[k], "new": b[k]})
    return diffs

def diff_roles(state: DGHState) -> DGHState:
    baseline = {}   # later: fetch from DB; today just empty
    current  = state.get("extracted_json", {})
    state["diff_json"] = _dict_diff(baseline, current)
    return state

def save_gold_example(state: DGHState) -> DGHState:
    client = LSClient()
    dataset_name = "refinery-gold-v1"
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Found existing dataset: {dataset_name}")
    except LangSmithNotFoundError:
        print(f"Dataset {dataset_name} not found, creating new one.")
        dataset = client.create_dataset(
            dataset_name=dataset_name, 
            description="Validated role extractions"
        )

    example = client.create_example(
        dataset_id = dataset.id,
        inputs     = {"raw_text": state["input_docs"][0]},
        outputs    = state["extracted_json"],
    )
    state["saved_location"] = f"langsmith://{example.id}"
    state["approved"] = True
    state["status"]   = "approved"
    return state 

# --- extraction with one repair attempt ---
def extract_data(state: DGHState) -> DGHState:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
    prompt = ChatPromptTemplate.from_file("prompts/extract_roles.jinja")
    raw = "\n".join(state["input_docs"])
    messages = prompt.format_messages(schema=state["extraction_schema"], input=raw)
    response = llm.invoke(messages).content

    try:
        data = json.loads(response)
        validate(data, state["extraction_schema"])
    except (json.JSONDecodeError, ValidationError):
        # one repair attempt
        repair_msg = [{"role":"system","content":"Fix JSON so it passes schema"},{"role":"user","content":response}]
        fixed = llm.invoke(repair_msg).content
        try:
            data = json.loads(fixed)
            validate(data, state["extraction_schema"])
        except Exception:
            data = {}

    state["extracted_json"] = data
    return state 