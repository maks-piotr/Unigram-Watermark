import re
import hashlib
from typing import List
import numpy as np
from scipy.stats import norm
import torch
from transformers import LogitsWarper, GPT2Tokenizer



class GPTWatermarkBase:
    def __init__(self, fraction: float = 0.5, strength: float = 2.0, vocab_size: int = 50257, watermark_key: int = 0, excluded_tokens: List[str] = None):
        rng = np.random.default_rng(self._hash_fn(watermark_key))

        all_tokens = [str(i) for i in range(vocab_size)]
        print(f"Before filtering: {len(all_tokens)}")

        if excluded_tokens:
            tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2-xl')
            
            # Przechodzimy przez tokeny i wykluczamy te, które odpowiadają wzorcom
            all_tokens = [
                token for token in all_tokens 
                if not self.is_excluded(token, excluded_tokens, tokenizer)
            ]

        print(f"After filtering: {len(all_tokens)}")
        
        mask = np.array([True] * int(fraction * len(all_tokens)) + [False] * (len(all_tokens) - int(fraction * len(all_tokens)) ))
        rng.shuffle(mask)

        mask_indices = [int(token) for token in all_tokens if token.isdigit()]
        self.green_list_mask = torch.zeros(vocab_size, dtype=torch.float32)
        self.green_list_mask[mask_indices] = torch.tensor(mask, dtype=torch.float32)

        self.strength = strength
        self.fraction = fraction

    def is_excluded(self, token: str, excluded_tokens: List[str], tokenizer: GPT2Tokenizer) -> bool:
        """Check if a token should be excluded based on given patterns (e.g., prefix, suffix)."""
        # Przekształcamy token na słowo
        word = tokenizer.decode([int(token)])

        # Przekształcenie listy tokenów w całe słowo
        word = word.replace(' ', '')  # Usuwamy ewentualne spacje (tokenizacja może dzielić słowa na więcej niż jeden token)

        # Sprawdzamy, czy którykolwiek fragment słowa pasuje do wzorców
        for pattern in excluded_tokens:
            if re.search(pattern, word):  # Zmieniamy na 'search' zamiast 'match', by sprawdzić w dowolnym miejscu w słowie
                return True  # Token pasuje do wzorca (w dowolnym miejscu słowa)
        return False

    @staticmethod
    def _hash_fn(x: int) -> int:
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')

class GPTWatermarkLogitsWarper(GPTWatermarkBase, LogitsWarper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        """Add the watermark to the logits and return new logits."""
        watermark = self.strength * self.green_list_mask
        new_logits = scores + watermark.to(scores.device)
        return new_logits


class GPTWatermarkDetector(GPTWatermarkBase):
    """
    Class for detecting watermarks in a sequence of tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _z_score(num_green: int, total: int, fraction: float) -> float:
        """Calculate and return the z-score of the number of green tokens in a sequence."""
        return (num_green - fraction * total) / np.sqrt(fraction * (1 - fraction) * total)
    
    @staticmethod
    def _compute_tau(m: int, N: int, alpha: float) -> float:
        """
        Compute the threshold tau for the dynamic thresholding.

        Args:
            m: The number of unique tokens in the sequence.
            N: Vocabulary size.
            alpha: The false positive rate to control.
        Returns:
            The threshold tau.
        """
        factor = np.sqrt(1 - (m - 1) / (N - 1))
        tau = factor * norm.ppf(1 - alpha)
        return tau

    def detect(self, sequence: List[int]) -> float:
        """Detect the watermark in a sequence of tokens and return the z value."""
        green_tokens = int(sum(self.green_list_mask[i] for i in sequence))

        return self._z_score(green_tokens, len(sequence), self.fraction)

    def unidetect(self, sequence: List[int]) -> float:
        """Detect the watermark in a sequence of tokens and return the z value. Just for unique tokens."""
        sequence = list(set(sequence))
        green_tokens = int(sum(self.green_list_mask[i] for i in sequence))
        return self._z_score(green_tokens, len(sequence), self.fraction)
    
    def dynamic_threshold(self, sequence: List[int], alpha: float, vocab_size: int) -> (bool, float):
        """Dynamic thresholding for watermark detection. True if the sequence is watermarked, False otherwise."""
        z_score = self.unidetect(sequence)
        tau = self._compute_tau(len(list(set(sequence))), vocab_size, alpha)
        return z_score > tau, z_score
