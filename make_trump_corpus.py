import json
import re
from db_functions_and_helpers import *

no_urls   = re.compile(r'https?://\S+')
no_hashes = re.compile(r'#\w+')
no_ats    = re.compile(r'@\w+')

def int_index_word(wordbag, word):
    return wordbag.get(word)

def make_wordbag(conn, texts, transcript_type, batch_size=1000):
    wordbag_table_name = f'trump_{transcript_type}_wordbag'
    wordbag_table_params = {'word': 'TEXT', 'idx': 'INT'}
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
        batch.append((word, idx))
        if len(batch) >= batch_size:
            cursor.executemany(f"INSERT INTO {wordbag_table_name} VALUES (?, ?)", batch)
            conn.commit()
            batch.clear()
    if batch:
        cursor.executemany(f"INSERT INTO {wordbag_table_name} VALUES (?, ?)", batch)
        conn.commit()
    return wordbag


def convert_texts_to_int(text, wordbag):
    return [int_index_word(wordbag, word) for word in text.split() if int_index_word(wordbag, word) is not None]

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

def create_tweet_tables(conn, transcript_type, texts, wordbag, batch_size=1000):
    tweet_table_name = f'trump_{transcript_type}'
    tweet_table_params = {'textID': 'INT', 'text': 'TEXT'}
    create_table(conn, tweet_table_name, tweet_table_params, None)
    cursor = conn.cursor()
    batch = []
    for textID, raw_text in texts.items():
        text = ','.join(str(i) for i in convert_texts_to_int(raw_text, wordbag))
        batch.append((textID, text))
        if len(batch) >= batch_size:
            cursor.executemany(f"INSERT INTO {tweet_table_name} VALUES (?, ?)", batch)
            batch.clear()
    if batch:
        cursor.executemany(f"INSERT INTO {tweet_table_name} VALUES (?, ?)", batch)
    print(f"Inserted {len(texts)} {transcript_type} tweets")
    conn.commit()

def create_trump_corpus(conn, nlp):
    with open('trump_tweets_01-08-2021.json', 'r') as f:
        raw_trump_tweets = json.load(f)
    print('Trump data acquired')
    cleaned_tweets = clean_tweets(raw_trump_tweets)
    text_wordbag = make_wordbag(conn, cleaned_tweets.values(), 'text')
    print('Cleaned')
    lemmatized_tweets = lemmatize_all(cleaned_tweets, nlp)
    lem_wordbag = make_wordbag(conn, lemmatized_tweets.values(), 'lemmatized')
    print('Lemmatized')
    create_tweet_tables(conn, 'text',  cleaned_tweets, text_wordbag)
    create_tweet_tables(conn, 'lemmatized',  lemmatized_tweets, lem_wordbag)