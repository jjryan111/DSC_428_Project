import torch
import torch.nn as nn
from ngram_for_model import NgramFeatureExtractor


class CharTokenizer:
    """
    Lightweight character-level tokenizer.
    Converts text → integer tensor of char IDs (padded/truncated to max_len).
    """

    VOCAB_SIZE = 128  # printable ASCII

    def __init__(self, max_len: int = 1024):
        self.max_len = max_len

    def encode(self, texts: list[str]) -> torch.Tensor:
        out = torch.zeros(len(texts), self.max_len, dtype=torch.long)
        for i, text in enumerate(texts):
            ids = [min(ord(c), self.VOCAB_SIZE - 1) for c in text[: self.max_len]]
            out[i, : len(ids)] = torch.tensor(ids)
        return out


class CharCNNBlock(nn.Module):
    """Parallel conv filters at multiple kernel sizes, then max-over-time pooling."""

    def __init__(self, embed_dim: int, num_filters: int, kernel_sizes: list[int]):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(embed_dim, num_filters, k, padding=k // 2),
                    nn.ReLU(),
                )
                for k in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, embed_dim, L)
        pooled = [conv(x).max(dim=-1).values for conv in self.convs]
        return torch.cat(pooled, dim=-1)  # (B, num_filters * len(kernel_sizes))


class CharCNNLSTMAuthorClassifier(nn.Module):
    """
    Character embedding → CNN (multi-scale kernels) → LSTM → pooled 256-d
    + n-gram branch (ngram_dim → 64-d)
    → concat → classifier head

    This architecture captures subword patterns (spelling quirks, punctuation
    habits) that transformers sometimes miss when tokenising aggressively.
    """

    KERNEL_SIZES = [2, 3, 4, 5]
    NUM_FILTERS = 64       # per kernel size
    LSTM_HIDDEN = 128
    EMBED_DIM = 32

    def __init__(self, signature_ngrams: list[str], dropout: float = 0.3):
        super().__init__()
        ngram_dim = len(signature_ngrams)

        self.tokenizer = CharTokenizer()

        # --- Character branch ---
        self.char_embed = nn.Embedding(
            CharTokenizer.VOCAB_SIZE, self.EMBED_DIM, padding_idx=0
        )
        self.cnn = CharCNNBlock(self.EMBED_DIM, self.NUM_FILTERS, self.KERNEL_SIZES)
        cnn_out_dim = self.NUM_FILTERS * len(self.KERNEL_SIZES)  # 256

        self.lstm = nn.LSTM(
            input_size=self.EMBED_DIM,
            hidden_size=self.LSTM_HIDDEN,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        lstm_out_dim = self.LSTM_HIDDEN * 2  # bidirectional → 256

        # Fuse CNN and LSTM features
        char_fused_dim = cnn_out_dim + lstm_out_dim  # 512
        self.char_proj = nn.Sequential(
            nn.Linear(char_fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
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

        # --- Classifier head (256 + 64 = 320) ---
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device

        # Character encoding
        char_ids = self.tokenizer.encode(texts).to(device)        # (B, L)
        emb = self.char_embed(char_ids)                           # (B, L, E)

        # CNN path: needs (B, E, L)
        cnn_out = self.cnn(emb.permute(0, 2, 1))                 # (B, 256)

        # LSTM path: mean-pool over time
        lstm_out, _ = self.lstm(emb)                             # (B, L, 256)
        lstm_pooled = lstm_out.mean(dim=1)                       # (B, 256)

        char_fused = self.char_proj(
            torch.cat([cnn_out, lstm_pooled], dim=-1)
        )                                                         # (B, 256)

        # N-gram branch
        ngram_vec = self.ngram_extractor.transform(texts).to(device)
        ngram_emb = self.ngram_proj(ngram_vec)                   # (B, 64)

        combined = torch.cat([char_fused, ngram_emb], dim=-1)   # (B, 320)
        return self.head(combined).squeeze(-1)                   # (B,)