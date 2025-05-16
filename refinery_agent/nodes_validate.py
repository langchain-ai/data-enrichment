from jsonschema import validate, Draft7Validator, FormatChecker
from jsonschema.exceptions import ValidationError
from dgh_state import DGHState

def validate_schema_node(state: DGHState) -> DGHState:
    """Ensure extracted_json conforms; if not, flag error."""
    schema = state["extraction_schema"]
    data   = state.get("extracted_json", {})
    try:
        validate(instance=data, schema=schema,
                 cls=Draft7Validator, format_checker=FormatChecker())
    except ValidationError as e:
        state["approved"] = False
        state["diff_json"] = {"schema_error": str(e)}
    return state 