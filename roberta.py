import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
from ngram_for_model import NgramFeatureExtractor


class RobertaAuthorClassifier(nn.Module):
    """
    RoBERTa CLS embedding  (768-d)
    + n-gram TF-IDF branch (ngram_dim → 64-d)
    → concat → classifier head

    RoBERTa is pre-trained without NSP and with BPE, making it
    stronger for stylometric tasks than BERT/DistilBERT.
    """

    def __init__(self, signature_ngrams: list[str], dropout: float = 0.3):
        super().__init__()
        ngram_dim = len(signature_ngrams)

        self.encoder = RobertaModel.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        self.ngram_extractor = NgramFeatureExtractor(signature_ngrams)
        self.ngram_proj = nn.Sequential(
            nn.Linear(ngram_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # RoBERTa also outputs 768-d CLS
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768 + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
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
        return out.last_hidden_state[:, 0, :]  # CLS → (B, 768)

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device

        cls_emb = self._encode_text(texts, device)
        ngram_vec = self.ngram_extractor.transform(texts).to(device)
        ngram_emb = self.ngram_proj(ngram_vec)

        combined = torch.cat([cls_emb, ngram_emb], dim=-1)
        return self.head(combined).squeeze(-1)

    def freeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = True