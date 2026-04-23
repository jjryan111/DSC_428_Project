import json
import re
from db_functions_and_helpers import *

no_urls   = re.compile(r'https?://\S+')
no_hashes = re.compile(r'#\w+')
no_ats    = re.compile(r'@\w+')

def make_wordbag(conn, texts, transcript_type, batch_size=1000):
    wordbag_table_name = f'congress_{transcript_type}_wordbag'
    wordbag_table_params = {'word': 'TEXT', 'wordID': 'INTEGER'}
    wordbag_pkey = 'word'
    create_table(conn, wordbag_table_name, wordbag_table_params, wordbag_pkey)
    ctr = 0
    wordbag = {}
    for text in texts:
        for word in text.split():
           if word not in wordbag:
               wordbag[word] = ctr
               ctr +=1
    cursor = conn.cursor()
    batch = []
    for word, idx in wordbag.items():
        batch.append((word, 0))
        if len(batch) >= batch_size:
            cursor.executemany(f"INSERT INTO {wordbag_table_name} VALUES (?, ?)", batch)
            conn.commit()
            batch.clear()
    if batch:
        cursor.executemany(f"INSERT INTO {wordbag_table_name} VALUES (?, ?)", batch)
        conn.commit()

def lemmatize_tweet(tweet, nlp):
    doc = nlp(tweet)
    return ' '.join(token.lemma_ for token in doc if not token.is_space)


def lemmatize_all(tweets, nlp):
    """
    tweets: dict of {textID: tweet_text}
    returns: dict of {textID: lemmatized_tweet_text}
    """
    return {textID: lemmatize_tweet(tweet, nlp) for textID, tweet in tweets.items()}

def clean_tweets(tweets):
    cleaned = {}
    for tweet in tweets:
        textID = tweet['id']
        clean = no_urls.sub('', tweet['text'])
        clean = no_hashes.sub('', clean)
        clean = no_ats.sub('', clean)
        cleaned[textID] = clean.strip()
    return cleaned

def save_first_pass_tweets(conn, tweets, transcript_type, batch_size=1000):
    tweet_table_name = f'congress_{transcript_type}'
    tweet_table_params = {'textID': 'INT', 'text': 'TEXT', 'num_idx': 'TEXT'}
    tweet_table_key = 'textID'
    create_table(conn, tweet_table_name, tweet_table_params, tweet_table_key)
    cursor = conn.cursor()
    batch = []
    for textID, raw_text in tweets.items():
        batch.append((textID, raw_text, ''))
        if len(batch) >= batch_size:
            cursor.executemany(f"INSERT INTO {tweet_table_name} VALUES (?, ?, ?)", batch)
            batch.clear()
    if batch:
        cursor.executemany(f"INSERT INTO {tweet_table_name} VALUES (?, ?, ?)", batch)
    print(f"Inserted {len(tweets)} {transcript_type} tweets")
    conn.commit()

def update_tweet_tables(conn, transcript_type, wordbag, batch_size=1000):
    table_name = f'congress_{transcript_type}'
    table_query = f'SELECT * FROM {table_name}'
    texts = get_table_contents(conn, table_name, table_query)
    batch = []
    cursor = conn.cursor()
    for textID, text, _ in texts:
        idx_text = convert_texts_to_int(text, wordbag)
        insert_idx_text = ','.join(str(i) for i in idx_text)
        batch.append((insert_idx_text, textID))
        if len(batch) >= batch_size:
            cursor.executemany(f"UPDATE {table_name} SET  num_idx = (?) WHERE textID = (?)", batch)
            batch.clear()
    if batch:
        cursor.executemany(f"UPDATE {table_name} SET  num_idx = (?) WHERE textID = (?)", batch)
    conn.commit()

def create_congress_corpus_wordbag(conn, nlp):
    cursor = conn.cursor()

    raw_congress_tweets = cursor.execute('SELECT * FROM congress_tweets').fetchall()
    parsed_congress_tweets = [{'id': id, 'text': text, 'num_idx': num_idx} for id, text, num_idx in raw_congress_tweets]
    print('congress data acquired')
    cleaned_tweets = clean_tweets(parsed_congress_tweets)
    save_first_pass_tweets(conn, cleaned_tweets, 'text', batch_size=1000)
    make_wordbag(conn, cleaned_tweets.values(), 'text')
    print('Cleaned')
    lemmatized_tweets = lemmatize_all(cleaned_tweets, nlp)
    make_wordbag(conn, lemmatized_tweets.values(), 'lemmatized')
    save_first_pass_tweets(conn, lemmatized_tweets, 'lemmatized', batch_size=1000)
    print('Lemmatized')

def create_congress_transcripts(conn, wordbag, transcript_type):
    update_tweet_tables(conn, transcript_type, wordbag)