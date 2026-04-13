import spacy
import sqlite3
from make_coca_corpus import create_coca_corpus
from make_trump_corpus import create_trump_corpus

nlp = spacy.load("en_core_web_sm")
TRANSCRIPT_TYPES= ['text','lemmatized']
CORPORA = ['trump', 'coca']
DB_NAME = "coca_corpus.db"

conn = sqlite3.connect(DB_NAME)
print('Starting COCA')
create_coca_corpus(conn, DB_NAME, TRANSCRIPT_TYPES)
print('Starting Trump')
create_trump_corpus(conn, nlp)
conn.close()
print('Done')