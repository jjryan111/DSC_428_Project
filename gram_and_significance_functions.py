from collections import defaultdict
from db_functions_and_helpers import create_table
from itertools import product
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import scipy.stats

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
    cursor.execute(f'SELECT num_idx  FROM {source_table_name}')
    return [row[0] for row in cursor.fetchall()]



def decode_gram(gram_str, idx_to_word):
    """Convert '123|456|789' back to readable words"""
    indices = gram_str.split('|')
    return ' '.join(idx_to_word.get(int(i), f'<UNK:{i}>') for i in indices)

def load_master_wordbag(conn, transcript_type):
    """Returns both word->idx and idx->word mappings"""
    table = f'master_{transcript_type}_wordbag'
    cursor = conn.cursor()
    cursor.execute(f'SELECT word, idx FROM {table}')
    word_to_idx = {}
    idx_to_word = {}
    for word, idx in cursor.fetchall():
        word_to_idx[word] = idx
        idx_to_word[idx] = word
    return word_to_idx, idx_to_word

def get_corpus_total(conn, corpus, transcript_type):
    table = f'{corpus}_n_and_skipgrams_{transcript_type}'
    cursor = conn.cursor()
    cursor.execute(f'SELECT SUM(count) FROM {table} WHERE n=1 AND k=0')
    return cursor.fetchone()[0]

def run_significance_for_nk(conn, corpus_a, corpus_b, transcript_type,
                             n, k, total_a, total_b,
                             min_count_a=100, alpha=0.05, min_log_or=0.5):
    table_a = f'{corpus_a}_n_and_skipgrams_{transcript_type}'
    table_b = f'{corpus_b}_n_and_skipgrams_{transcript_type}'

    df_a = pd.read_sql_query(f'SELECT gram, count FROM {table_a} WHERE n=? AND k=?',
                             conn, params=(n, k))
    df_b = pd.read_sql_query(f'SELECT gram, count FROM {table_b} WHERE n=? AND k=?',
                             conn, params=(n, k))

    df_a = df_a[df_a['count'] >= min_count_a]
    if df_a.empty:
        return pd.DataFrame()

    df = df_a.merge(df_b, on='gram', how='left', suffixes=('_a', '_b'))
    df['count_b'] = df['count_b'].fillna(0).astype(np.int64)
    df['count_a'] = df['count_a'].astype(np.int64)

    a = df['count_a'].values
    b = df['count_b'].values
    c = (total_a - a).clip(min=0)
    d = (total_b - b).clip(min=0)
    n_total = a + b + c + d

    expected_a = (a + b) * (a + c) / n_total
    valid = expected_a >= 5
    if not valid.any():
        return pd.DataFrame()

    df, a, b, c, d, n_total = (x[valid] if isinstance(x, np.ndarray) else x
                                for x in [df, a, b, c, d, n_total])

    exp = np.array([(a+b)*(a+c)/n_total, (a+b)*(b+d)/n_total,
                    (c+d)*(a+c)/n_total, (c+d)*(b+d)/n_total])
    obs = np.array([a, b, c, d])
    chi2 = np.sum((obs - exp)**2 / exp, axis=0)
    p_values = scipy.stats.chi2.sf(chi2, df=1)  # sf = 1 - cdf, more numerically stable

    rejected, p_adj, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    log_or = np.log((a + 0.5) / (c + 0.5)) - np.log((b + 0.5) / (d + 0.5))

    df = df.copy()
    df['n'] = n
    df['k'] = k
    df['chi2'] = chi2
    df['p_raw'] = p_values
    df['p_adj'] = p_adj
    df['log_odds_ratio'] = log_or
    df['rejected'] = rejected

    # filter to significant AND meaningful effect size, positive direction only
    return df[(df['rejected']) & (df['log_odds_ratio'] >= min_log_or)]


def find_significant_grams_all(conn, transcript_type, corpus_a='trump', corpus_b='coca',
                                min_count_a=100, alpha=0.05, min_log_or=0.5,
                                target_n=3000, max_n=4, max_k=3):

    _, idx_to_word = load_master_wordbag(conn, transcript_type)
    total_a = get_corpus_total(conn, corpus_a, transcript_type)
    total_b = get_corpus_total(conn, corpus_b, transcript_type)

    all_results = []
    nk_combos = [(n, k) for n, k in product(range(1, max_n+1), range(0, max_k+1))
                 if not (n == 1 and k > 0)]

    for n, k in nk_combos:
        print(f'Processing n={n}, k={k}...')
        result = run_significance_for_nk(
            conn, corpus_a, corpus_b, transcript_type,
            n, k, total_a, total_b,
            min_count_a=min_count_a, alpha=alpha, min_log_or=min_log_or
        )
        if not result.empty:
            all_results.append(result)
            print(f'  -> {len(result)} significant grams')

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    combined['gram_decoded'] = combined['gram'].apply(
        lambda g: decode_gram(g, idx_to_word)
    )

    combined = combined.sort_values('log_odds_ratio', ascending=False)

    # self-titrating: if over target, tighten min_log_or until ~target_n remain
    if len(combined) > target_n:
        # find the log_or threshold at the target_n-th row
        threshold = combined.iloc[target_n - 1]['log_odds_ratio']
        combined = combined[combined['log_odds_ratio'] >= threshold]
        print(f'Titrated to {len(combined)} grams at log_OR >= {threshold:.3f}')

    return combined[['gram', 'gram_decoded', 'n', 'k',
                      'count_a', 'count_b', 'chi2',
                      'p_raw', 'p_adj', 'log_odds_ratio']]