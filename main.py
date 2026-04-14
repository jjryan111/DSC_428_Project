import spacy
import sqlite3
from make_coca_corpus import create_coca_corpus
from make_trump_corpus import create_trump_corpus
from gram_and_significance_functions import build_all_grams
from db_functions_and_helpers import db_summary
nlp = spacy.load("en_core_web_sm")
TRANSCRIPT_TYPES= ['text','lemmatized']
CORPORA = ['trump', 'coca']
DB_NAME = "coca_corpus.db"

conn = sqlite3.connect(DB_NAME)
print('Starting COCA')
create_coca_corpus(conn, DB_NAME, TRANSCRIPT_TYPES)
print('Starting Trump')
create_trump_corpus(conn, nlp)

print('Done')
db_summary(DB_NAME)
for transcript_type in TRANSCRIPT_TYPES:
    for corpus in CORPORA:
        build_all_grams(transcript_type, corpus, conn)

conn.close()