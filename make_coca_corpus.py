from db_functions_and_helpers import *

def sanity_check(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM db_spok")
    print(f"db_spok rows: {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM db_spok AS sp JOIN lexicon AS lex ON sp.wordID = lex.wordID")
    print(f"db_spok rows: {cursor.fetchone()[0]}")

def create_coca_database(conn, db_name):
    load_table(conn, "db_spok.txt", "INSERT INTO db_spok VALUES (?, ?, ?)", 3)
    load_table(conn, "lexicon.txt", "INSERT INTO lexicon VALUES (?, ?, ?, ?)", 4)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM db_spok")
    print(f"db_spok rows: {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM lexicon")
    print(f"lexicon rows: {cursor.fetchone()[0]}")
    print(f"Done. Database saved as '{db_name}'")


def make_coca_wordbag(conn, transcript_type):
    if transcript_type == 'text':
        word_or_lemma = 'word'
    else:
        word_or_lemma = 'lemma'
    cursor = conn.cursor()
    wordbag_table = f'coca_{transcript_type}_wordbag'
    wordbag_table_params = {'word': 'TEXT', 'wordID': 'INT' }
    wordbag_table_pkey = 'word'
    create_table(conn, wordbag_table, wordbag_table_params, wordbag_table_pkey)

    select_query = f"""
        SELECT DISTINCT(LOWER(lex.{word_or_lemma})) as {word_or_lemma}
        FROM lexicon AS lex 
    """
    cursor.execute(select_query)

    batch = []
    for row in cursor.fetchall():
        batch.append((row[0], 0))
        if len(batch) >= 10000:
            conn.executemany(f"INSERT INTO {wordbag_table} VALUES (?, ?)", batch)
            batch.clear()
    if batch:
        conn.executemany(f"INSERT INTO {wordbag_table} VALUES (?, ?)", batch)
    conn.commit()

def reconstruct_coca_texts(conn, transcript_type):
    if transcript_type == 'text':
        word_or_lemma = 'word'
    else:
        word_or_lemma = 'lemma'
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT sp.textID, lex.{word_or_lemma} 
        FROM db_spok AS sp
        JOIN lexicon AS lex ON sp.wordID = lex.wordID
        ORDER BY textID
    """)

    texts = {}
    for textID, word in cursor:
        if textID not in texts:
            texts[textID] = []
        texts[textID].append(word)
    return texts

def make_coca_transcript_table(conn, transcript_type, texts, wordbag):
    transcript_table = f'coca_{transcript_type}'
    transcript_table_params = {'textID': 'INT', 'text': 'TEXT', 'num_idx': 'TEXT'}
    create_table(conn, transcript_table, transcript_table_params, None )
    batch = []
    for textID, words in texts.items():
        batch.append((textID, ','.join(words), ','.join(str(i) for i in convert_texts_to_int(' '.join(words), wordbag))))
        if len(batch) >= 1000:
            conn.executemany(f"INSERT INTO {transcript_table} VALUES (?, ?, ?)", batch)
            batch.clear()
    if batch:
        conn.executemany(f"INSERT INTO {transcript_table} VALUES (?, ?, ?)", batch)

    conn.commit()
    print(f"Inserted {len(texts)} texts into {transcript_table}")

def create_coca_corpus_wordbag(conn, db_name, transcript_types):
    create_database(conn, db_name)
    create_coca_database(conn, db_name)
    sanity_check(conn)
    for transcript_type in transcript_types:
        make_coca_wordbag(conn, transcript_type)

def create_coca_transcripts(conn, master_wordbag, transcript_type):
    texts = reconstruct_coca_texts(conn, transcript_type)
    make_coca_transcript_table(conn, transcript_type, texts, master_wordbag)
