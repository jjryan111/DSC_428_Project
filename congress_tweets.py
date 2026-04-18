import json
import os
import glob
import sqlite3
import json
import os


def load_congress_tweets(db_path: str, json_dir: str, target_rows: int = 56500):
    conn = sqlite3.connect(db_path)
    conn.execute("""DROP TABLE IF  EXISTS congress_tweets""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS congress_tweets (
            textID  INTEGER PRIMARY KEY,
            text    TEXT,
            num_idx TEXT
        )
    """)
    conn.commit()

    count = 0
    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue
        if count >= target_rows:
            break

        filepath = os.path.join(json_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping malformed file: {filename}")
                continue

        # Handle both a single dict and a list of dicts per file
        records = data if isinstance(data, list) else [data]

        for record in records:
            if count >= target_rows:
                break
            text = record.get("text", "").strip()
            if not text:
                continue

            conn.execute(
                "INSERT INTO congress_tweets (textID, text, num_idx) VALUES (?, ?, ?)",
                (count, text, str(count))
            )
            count += 1

        conn.commit()
        print(f"  {filename}: running total {count:,}")

    conn.close()
    print(f"\nDone. {count:,} rows inserted into congress_tweets.")
