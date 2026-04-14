from collections import defaultdict
from db_functions_and_helpers import create_table
from itertools import product

def get_ngrams_and_skipgrams(texts, n=2, k=0):
    """
    texts : dict of {textID: [wordID, ...]}
    n     : size of the gram (1-4)
    k     : number of words to skip between each token (0 = standard ngram, ignored for unigrams)
    """
    gram_counts = defaultdict(int)

    for text in texts:
        words = [int(i) for i in text.split(',') if i]
        if n == 1:
            for word in words:
                gram_counts[(word,)] += 1
        else:
            step = k + 1
            window = (n - 1) * step + 1
            for i in range(len(words) - window + 1):
                gram = tuple(words[i + j * step] for j in range(n))
                gram_counts[gram] += 1

    return dict(gram_counts)


def insert_grams(conn, table_name, gram_dict, n, k, batch_size=10000):
    cursor = conn.cursor()
    batch = []
    for gram, count in gram_dict.items():
        batch.append(("|".join(str(i) for i in gram), n, k, count))
        if len(batch) >= batch_size:
            insert_query = f"""
                INSERT INTO {table_name} (gram, n, k, count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(gram, n, k) DO UPDATE SET count = count + excluded.count
            """
            cursor.executemany(insert_query, batch)
            batch.clear()
    if batch:
        batch_query = f"""
            INSERT INTO {table_name} (gram, n, k, count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(gram, n, k) DO UPDATE SET count = count + excluded.count
        """
        cursor.executemany(batch_query, batch)
    conn.commit()


def build_all_grams(transcript_type, corpus, conn, max_n=4, max_k=3):
    gram_table_name = f'{corpus}_n_and_skipgrams_{transcript_type}'
    gram_table_params = { 'gram': 'TEXT', 'n': 'INTEGER', 'k': 'INTEGER', 'count': 'INTEGER'}
    gram_pkey = 'gram, n, k'
    create_table(conn, gram_table_name, gram_table_params, gram_pkey)
    source_table = f'{corpus}_{transcript_type}'
    texts = get_texts_from_sqlite(conn, source_table)
    for n, k in product(range(1, max_n + 1), range(0, max_k + 1)):
        if n == 1 and k > 0:
            continue
        print(f"Generating n={n}, k={k}...")
        gram_dict = get_ngrams_and_skipgrams(texts, n=n, k=k)
        insert_grams(conn, gram_table_name, gram_dict, n, k)
        print(f"  -> {len(gram_dict)} grams inserted")

def get_texts_from_sqlite(conn, source_table_name):
    cursor = conn.cursor()
    cursor.execute(f'SELECT text FROM {source_table_name}')
    return [row[0] for row in cursor.fetchall()]

def find_significant_words_n_skip_grams(corpus, transcript_type):
    pass
