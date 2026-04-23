import sqlite3

import torch

# def predict(model, threshold: float, texts: list[str]) -> list[dict]:
#     """
#     Returns a list of dicts with the raw probability and binary prediction
#     for each input text.
#     """
#     device = next(model.parameters()).device
#     model.eval()
#
#     with torch.no_grad():
#         logits = model(texts)
#         probs  = torch.sigmoid(logits).cpu().tolist()
#
#     if isinstance(probs, float):  # single input edge case
#         probs = [probs]
#
#     return [
#         {
#             "text":       t[:80] + "..." if len(t) > 80 else t,
#             "prob":       round(p, 4),
#             "prediction": "trump" if p >= threshold else "other"
#         }
#         for t, p in zip(texts, probs)
#     ]

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#
import gc
# import torch
# import pandas as pd
from confusion import get_predictions, plot_confusion_matrix
from db_functions_and_helpers import load_model
from distilbert import DistilBertAuthorClassifier
from roberta import RobertaAuthorClassifier
from my_cnn import CharCNNLSTMAuthorClassifier
from train import load_splits
from congress_tweets import clean_tweets
import re

no_urls   = re.compile(r'https?://\S+')
no_hashes = re.compile(r'#\w+')
no_ats    = re.compile(r'@\w+')

def clean_tweet(text):
    clean = no_urls.sub('', text)
    clean = no_hashes.sub('', clean)
    clean = no_ats.sub('', clean)
    return clean.strip()

def predict():
    DB_NAME = "coca_corpus.db"
    conn = sqlite3.connect(DB_NAME)
    _, _, test_dataset = load_splits(conn, "text", min_tokens=5)
    # test_dataset = [('FOX HATES THAT I AM AMERICA’S FAVORITE GOVERNOR (‘RATINGS KING’) SAVING AMERICA – WHILE TRUMP CAN’T EVEN CONQUER THE ‘BIG’ STAIRS ON AIR FORCE ONE ANY MORE!!! … FOX IS LOSING IT BECAUSE WHEN I TYPE, AMERICA NOW WINS!!! THANK YOU FOR YOUR ATTENTION TO THIS MATTER.', 1)]
    # test_dataset = [('AMERICA’S FAVORITE GOVERNOR  GREAT RT (‘RATINGS KING’) SAVING AMERICA  @CRN_Supplements Thank you @RepErikPaulsen for speaking to the dietary supplement industry today! And thank you for your support of HR 1175! ',0)]
    # test_dataset = [('This is a test',0)]
    configs = [
        ("DistilBERT", load_model, DistilBertAuthorClassifier, "distilbert.pt"),
        # ("RoBERTa",    load_model,    RobertaAuthorClassifier,     "roberta.pt"),
        ("CharCNN",    load_model,    CharCNNLSTMAuthorClassifier,  "charcnn.pt"),
    ]
    all_results = {}
    for name, loader, cls, path in configs:
        print(f"\nEvaluating {name}...")
        model, threshold = loader(cls, path)
        cleaned_text = [ (clean_tweet(tweet), label) for tweet, label in test_dataset]
        preds, labels, probs = get_predictions(model, test_dataset, threshold, batch_size=4)
        all_results[name] = plot_confusion_matrix(
            preds, labels,
            model_name=name,
            save_path=f"confusion_{name.lower()}.png"
        )
        all_results[name]['mean_prob'] = sum(probs) / len(probs)
        model.cpu()
        del model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'Model':<14} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Mean P(+)':>10}")
    print("-" * 63)
    for name, r in all_results.items():
        print(f"{name:<14} {r['accuracy']:>9.3f} {r['precision']:>10.3f} {r['recall']:>8.3f} {r['f1']:>8.3f} {r['mean_prob']:>10.3f}")