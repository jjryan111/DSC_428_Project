from db_functions_and_helpers import *

def sanity_check(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM db_spok")
    print(f"db_spok rows: {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM db_spok AS sp JOIN lexicon AS lex ON sp.wordID = lex.wordID")
    print(f"db_spok rows: {cursor.fetchone()[0]}")

def create_coca_database(conn, db_name):
    create_database(conn, db_name)
    cursor = conn.cursor()
    load_table(cursor, "db_spok.txt", "INSERT INTO db_spok VALUES (?, ?, ?)", 3)
    load_table(cursor, "lexicon.txt", "INSERT INTO lexicon VALUES (?, ?, ?, ?)", 4)
    conn.commit()

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
    wordbag_table_params = {'wordID': 'INT', f'{word_or_lemma}': 'TEXT', 'count': 'INT'}
    create_table(conn, wordbag_table, wordbag_table_params, None)

    select_query = f"""
        SELECT MIN(lex.wordID), LOWER(lex.{word_or_lemma}) as {word_or_lemma}, COUNT(sp.textID) as count
        FROM db_spok AS sp
        JOIN lexicon AS lex ON sp.wordID = lex.wordID
        GROUP BY LOWER(lex.{word_or_lemma})
        ORDER BY COUNT(sp.textID) DESC
    """
    cursor.execute(select_query)

    batch = []
    for row in cursor.fetchall():
        batch.append(row)
        if len(batch) >= 10000:
            conn.executemany(f"INSERT INTO {wordbag_table} VALUES (?, ?, ?)", batch)
            batch.clear()
    if batch:
        conn.executemany(f"INSERT INTO {wordbag_table} VALUES (?, ?, ?)", batch)
    conn.commit()
    _, _, _ = get_coca_stats(conn, wordbag_table)

def get_coca_stats(conn, wordbag_table):
    cursor = conn.cursor()
    cursor.execute(f"SELECT SUM(count) FROM {wordbag_table}")
    words_in_corpus = cursor.fetchone()[0]
    print(f'Number of words in the corpus: {words_in_corpus}')

    cursor.execute(f"SELECT COUNT(DISTINCT wordID) FROM {wordbag_table}")
    words_in_vocab = cursor.fetchone()[0]
    print(f'Number of words in the corpus vocabulary: {words_in_vocab}')

    cursor.execute("SELECT COUNT(DISTINCT textID) FROM db_spok")
    count_of_texts = cursor.fetchone()[0]
    print(f'The number of texts in the corpus: {count_of_texts}')
    return words_in_corpus, words_in_vocab, count_of_texts


def reconstruct_coca_texts(conn, transcript_type):
    if transcript_type == 'text':
        word_or_lemma = 'word'
    else:
        word_or_lemma = 'lemma'
    cursor = conn.cursor()
    cursor.execute("""
        SELECT textID, ID, wordID
        FROM db_spok
        ORDER BY textID, CAST(ID AS INTEGER)
    """)

    texts = {}
    for textID, ID, wordID in cursor:
        if textID not in texts:
            texts[textID] = []
        texts[textID].append(wordID)

    # for textID, words in list(texts.items())[:10]:
    #     print(f"Text {textID}: {','.join(words)}")
    return texts


def make_coca_transcript_table(conn, transcript_type, texts):
    transcript_table = f'coca_{transcript_type}'
    transcript_table_params = {'textID': 'INT', 'text': 'TEXT'}
    create_table(conn, transcript_table, transcript_table_params, None )
    batch = []
    for textID, words in texts.items():
        batch.append((textID, ','.join(words)))
        if len(batch) >= 1000:
            conn.executemany(f"INSERT INTO {transcript_table} VALUES (?, ?)", batch)
            batch.clear()
    if batch:
        conn.executemany(f"INSERT INTO {transcript_table} VALUES (?, ?)", batch)

    conn.commit()
    print(f"Inserted {len(texts)} texts into {transcript_table}")

def create_coca_corpus(conn, db_name, transcript_types):
    create_coca_database(conn, db_name)
    sanity_check(conn)
    for transcript_type in transcript_types:
        make_coca_wordbag(conn, transcript_type)
        texts = reconstruct_coca_texts(conn, transcript_type)
        make_coca_transcript_table(conn, transcript_type, texts)
