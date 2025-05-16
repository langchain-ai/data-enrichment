import sqlite3

DB_NAME = "refinery.db"

def create_table():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dgh_runs (
        run_id TEXT PRIMARY KEY,
        file_name TEXT,
        status TEXT,
        diff_json TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()
    print(f"Table 'dgh_runs' created or already exists in {DB_NAME}")

if __name__ == "__main__":
    create_table() 