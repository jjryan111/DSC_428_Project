import sqlite3
import csv

def create_database(conn, db_name):
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    db_spok_params = {'textID': 'TEXT', 'ID': 'TEXT', 'wordID': 'TEXT'}
    create_table(conn, 'db_spok', db_spok_params, None)
    lexicon_params = {'wordID': 'TEXT', 'word': 'TEXT', 'lemma': 'TEXT', 'pos': 'TEXT'}
    create_table(conn, 'lexicon', lexicon_params, None)

def create_table(conn, table_name, table_params, primary_key):
    create_query = f'CREATE TABLE IF NOT EXISTS {table_name} ('
    for key, key_type in table_params.items():
        create_query += f'{key} {key_type}, '
    if primary_key:
        create_query += f'PRIMARY KEY ({primary_key}))'  # <-- added closing )
    else:
        create_query = create_query.rstrip(', ') + ')'  # <-- fixed, strip trailing comma
    drop_query = f"DROP TABLE IF EXISTS {table_name}"
    conn.execute(drop_query)
    conn.execute(create_query)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.commit()

def db_summary(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cursor.fetchone()[0]

        print(f"\n{table} ({row_count} rows)")
        print("-" * 40)
        for col in columns:
            cid, name, dtype, notnull, default, pk = col
            pk_marker  = " PK" if pk else ""
            nn_marker  = " NOT NULL" if notnull else ""
            print(f"  {name:30} {dtype:10}{pk_marker}{nn_marker}")

def load_table(cursor, filename, sql, num_cols, batch_size=1000):
    batch = []
    skipped = 0
    with open(filename, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="\x00")
        for i, row in enumerate(reader):
            if len(row) != num_cols:
                print(row)
                skipped += 1
                continue
            batch.append(row)
            if len(batch) >= batch_size:
                cursor.executemany(sql, batch)
                batch.clear()
    if batch:
        cursor.executemany(sql, batch)  # insert remaining rows
    print(f"{filename}: skipped {skipped} malformed rows")
