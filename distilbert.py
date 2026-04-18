import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
from ngram_for_model import NgramFeatureExtractor
import numpy as np



class DistilBertAuthorClassifier(nn.Module):
    """
    DistilBERT CLS embedding  (768-d)
    + n-gram TF-IDF branch    (ngram_dim → 64-d via MLP)
    → concat → classifier head → P(is target author)
    """

    def __init__(self, signature_ngrams: list[str], dropout: float = 0.3):
        super().__init__()
        ngram_dim = len(signature_ngrams)

        # --- Transformer backbone ---
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )

        # --- N-gram branch ---
        self.ngram_extractor = NgramFeatureExtractor(signature_ngrams)
        self.ngram_proj = nn.Sequential(
            nn.Linear(ngram_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # --- Classifier head ---
        # 768 (CLS) + 64 (n-gram) = 832
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768 + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),  # raw logit; use BCEWithLogitsLoss
        )

    def _encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        out = self.encoder(**tokens)
        return out.last_hidden_state[:, 0, :]  # CLS token → (B, 768)

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device

        cls_emb = self._encode_text(texts, device)                         # (B, 768)
        ngram_vec = self.ngram_extractor.transform(texts).to(device)       # (B, ngram_dim)
        ngram_emb = self.ngram_proj(ngram_vec)                             # (B, 64)

        combined = torch.cat([cls_emb, ngram_emb], dim=-1)                # (B, 832)
        return self.head(combined).squeeze(-1)                             # (B,)

    def freeze_backbone(self):
        """Freeze transformer weights; useful for early training epochs."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = True