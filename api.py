from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dgh_state import DGHState
import sqlite3, json, os, pathlib, uuid, subprocess

app = FastAPI()

# allow browser calls from Vite dev-server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = pathlib.Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------- DB bootstrap ----------
DB_FILE = "refinery.db"
CREATE_SQL = """
CREATE TABLE IF NOT EXISTS dgh_runs (
  run_id      TEXT PRIMARY KEY,
  file_name   TEXT,
  status      TEXT,
  diff_json   TEXT,
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
)"""
con = sqlite3.connect(DB_FILE)
con.execute(CREATE_SQL)
con.commit()
con.close()

@app.get("/runs")
def list_runs():
    # naive—reads latest 20 rows from the sqlite table created earlier
    con=sqlite3.connect("refinery.db")
    rows=con.execute("SELECT run_id,file_name,status,diff_json FROM dgh_runs ORDER BY created_at DESC LIMIT 20")
    return [dict(zip(("run_id","file_name","status","diff_json"),r)) for r in rows] 

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")
    run_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{run_id}.pdf"
    contents = await file.read()
    dest.write_bytes(contents)

    # Call the graph synchronously (could be async in prod)
    cp = subprocess.run(
        ["poetry","run","python","graph.py","--files",str(dest)],
        capture_output=True, text=True
    )
    if cp.returncode != 0:
        raise HTTPException(status_code=500, detail=cp.stderr)

    std = cp.stdout
    try:
        start = std.index("GRAPH_JSON_START") + len("GRAPH_JSON_START")
        end   = std.index("GRAPH_JSON_END", start)
        payload = std[start:end].strip()
        result = json.loads(payload)
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse graph output: {e}")

    # Insert row for Inbox UI
    con = sqlite3.connect("refinery.db")
    con.execute(
        "INSERT INTO dgh_runs(run_id,file_name,status,diff_json) VALUES (?,?,?,?)",
        (run_id, file.filename, result["status"], json.dumps(result["diff_json"]))
    )
    con.commit()
    # Note: The provided patch does not close the sqlite3 connection.
    # Replicating as is, but this could be a resource leak.
    return {"run_id": run_id, "status": result["status"]} 