import re
from collections import Counter
import torch

class NgramFeatureExtractor:
    def __init__(self, signature_ngrams: list[str]):
        self.signature_ngrams = signature_ngrams
        # Index for O(1) lookup
        self.gram_to_idx = {g: i for i, g in enumerate(signature_ngrams)}
        self.vocab_size = len(signature_ngrams)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _extract(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.vocab_size

        total = len(tokens)
        counts = Counter()

        # Unigrams
        for tok in tokens:
            if tok in self.gram_to_idx:
                counts[tok] += 1

        # Bigrams
        for a, b in zip(tokens, tokens[1:]):
            gram = f"{a} {b}"
            if gram in self.gram_to_idx:
                counts[gram] += 1

        # Trigrams
        for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
            gram = f"{a} {b} {c}"
            if gram in self.gram_to_idx:
                counts[gram] += 1

        # Skipgrams (1-skip bigrams)
        for a, b in zip(tokens, tokens[2:]):
            gram = f"{a}_{b}"
            if gram in self.gram_to_idx:
                counts[gram] += 1

        vec = [counts.get(g, 0) / total for g in self.signature_ngrams]
        return vec

    def transform(self, texts: list[str]) -> torch.Tensor:
        return torch.tensor(
            [self._extract(t) for t in texts],
            dtype=torch.float32
        )

