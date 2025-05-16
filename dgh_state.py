from typing import TypedDict, Literal

class DGHState(TypedDict):
    input_docs: list[str]
    extraction_schema: dict
    extracted_json: dict | None
    diff_json: dict | None
    approved: bool | None
    saved_location: str | None
    status: Literal["to-parse", "approved"] 