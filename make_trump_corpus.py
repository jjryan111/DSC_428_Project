import json
import re
from db_functions_and_helpers import *

no_urls   = re.compile(r'https?://\S+')
no_hashes = re.compile(r'#\w+')
no_ats    = re.compile(r'@\w+')

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

def create_tweet_tables(conn, transcript_type, texts, batch_size=1000):
    tweet_table_name = f'trump_{transcript_type}'
    tweet_table_params = {'textID': 'INT', 'text': 'TEXT'}
    create_table(conn, tweet_table_name, tweet_table_params, None)
    cursor = conn.cursor()
    batch = []
    for textID, text in texts.items():
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
    print('Cleaned')
    lemmatized_tweets = lemmatize_all(cleaned_tweets, nlp)
    print('Lemmatized')
    create_tweet_tables(conn, 'text',  cleaned_tweets)
    create_tweet_tables(conn, 'lemmatized',  lemmatized_tweets)