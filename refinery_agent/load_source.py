import pdfplumber, pathlib
from dgh_state import DGHState

def load_source(state: DGHState) -> DGHState:
    paths = state.get("input_docs", [])
    texts = []
    for p in paths:
        pdf_path = pathlib.Path(p)
        if pdf_path.suffix.lower() == ".pdf" and pdf_path.exists():
            with pdfplumber.open(pdf_path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            texts.append("\n".join(pages))
        else:
            texts.append("")
    state["input_docs"] = texts
    return state 