import spacy
from make_coca_corpus import create_coca_corpus_wordbag, create_coca_transcripts
from make_trump_corpus import create_trump_corpus_wordbag, create_trump_transcripts
from gram_and_significance_functions import build_all_grams
from db_functions_and_helpers import *
from gram_and_significance_functions import find_significant_grams_all
from distilbert import DistilBertAuthorClassifier
import pandas as pd
from eval import predict
from distilbert import DistilBertAuthorClassifier
from roberta import RobertaAuthorClassifier
from my_cnn import CharCNNLSTMAuthorClassifier
from train import load_splits, train
from congress_tweets import load_congress_tweets
import os
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
nlp = spacy.load("en_core_web_sm")
TRANSCRIPT_TYPES= ['text','lemmatized']
CORPORA = ['trump', 'coca']
DB_NAME = "coca_corpus.db"

def make_master_wordbag(conn, transcript_type):
    wordbag = {}
    wordbag_table_name = f'master_{transcript_type}_wordbag'
    wordbag_table_params = {'word': 'TEXT', 'idx': 'INTEGER'}
    wordbag_table_pkey = 'word'
    create_table(conn, wordbag_table_name, wordbag_table_params, wordbag_table_pkey)
    coca_table_name = f'coca_{transcript_type}_wordbag'
    coca_words = get_table_contents(conn, coca_table_name, f'SELECT * FROM {coca_table_name}')
    trump_table_name = f'trump_{transcript_type}_wordbag'
    trump_words = get_table_contents(conn, trump_table_name, f'SELECT * FROM {trump_table_name}')
    ctr = 0
    for word, idx in coca_words:
        if word not in wordbag:
            wordbag[word] = ctr
            ctr+= 1
    for word, idx in trump_words:
        if word not in wordbag:
            wordbag[word] = ctr
            ctr+= 1
    for word, idx in wordbag.items():
        conn.execute(f"INSERT INTO {wordbag_table_name} VALUES (?, ?)", (word,idx))
    conn.commit()
    return wordbag

conn = sqlite3.connect(DB_NAME)
# print('Starting COCA')
# create_coca_corpus_wordbag(conn, DB_NAME, TRANSCRIPT_TYPES)
# print('Starting Trump')
# create_trump_corpus_wordbag(conn, nlp)
# print('Wordbags created')
# for transcript_type in TRANSCRIPT_TYPES:
#     master_wordbag = make_master_wordbag(conn, transcript_type)
#     print('Master Wordbag created')
#     create_coca_transcripts(conn, master_wordbag,transcript_type)
#     print('COCA Transcribed')
#     create_trump_transcripts(conn, master_wordbag, transcript_type)
#     print('Done. Starting grams')
#     for corpus in CORPORA:
#         build_all_grams(transcript_type, corpus, conn)
#     results = find_significant_grams_all(conn, transcript_type, 'trump', 'coca')
#     file_out = f'significant_gram_{transcript_type}.csv'
#     results.to_csv(file_out, index=False)
#

# load_congress_tweets(DB_NAME, "/home/jj/PyProjects/congresstweets/data")
# db_summary(DB_NAME)
#
results = pd.read_csv('significant_gram_text.csv')
signature_ngrams = results['gram_decoded'].tolist()
train_dataset, val_dataset, test_dataset = load_splits(conn, 'text')
conn.close()
models_to_train = {
   # "distilbert": DistilBertAuthorClassifier(signature_ngrams),
   # "roberta":    RobertaAuthorClassifier(signature_ngrams),
    "charcnn":    CharCNNLSTMAuthorClassifier(signature_ngrams),
}

for name, model in models_to_train.items():
    model, threshold = train(
        model,
        train_dataset.texts, train_dataset.labels.tolist(),
        val_dataset.texts, val_dataset.labels.tolist(),
    )
    save_model(model, threshold, path=f"{name}.pt")
    model.cpu()
    del model
    torch.cuda.empty_cache()
    gc.collect()

# models = {
#     "distilbert": load_model(DistilBertAuthorClassifier, "distilbert.pt"),
#     "roberta": load_model(RobertaAuthorClassifier, "roberta.pt"),
#     "charcnn": load_model(CharCNNLSTMAuthorClassifier, "charcnn.pt"),
# }
predict()
# for name, (model, threshold) in models.items():
#     print(f"\n--- {name} (threshold: {threshold:.2f}) ---")
#     texts= test_dataset.texts
#     for result in predict(model, threshold, texts):
#         print(f"  [{result['prediction']:>5}]  p={result['prob']:.4f}  \"{result['text']}\"")
