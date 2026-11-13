import hashlib
import logging
import numpy as np
from typing import Optional

from .ollama_base import OllamaBase
from src.config import EMBEDDING_MODEL_NAME, OLLAMA_MAX_RETRIES, OLLAMA_BASE_DELAY

logger = logging.getLogger(__name__)

class EmbeddingManager(OllamaBase):
    """
    Embedding manager for the fraud detection system.
    Generates dense vector representations using Ollama (nomic-embed-text or others),
    with deterministic fallback when the local daemon is unavailable.
    """
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        max_retries: int = OLLAMA_MAX_RETRIES,
        base_delay: float = OLLAMA_BASE_DELAY,
        dimension: int = 768
    ):
        super().__init__(model_name=model_name, max_retries=max_retries, base_delay=base_delay)
        self.dimension = dimension

    def _fallback_deterministic_embedding(self, text: str) -> np.ndarray:
        """
        Generates a dense normalized (L2) embedding based on cryptographic hashing
        to guarantee continuity in tests and environments without an active Ollama daemon.
        """
        words = text.lower().split()
        vec = np.zeros(self.dimension, dtype=np.float32)

        for i, word in enumerate(words):
            # Reproducible per-word seed
            h = int(hashlib.sha256(word.encode('utf-8')).hexdigest()[:8], 16)
            idx = h % self.dimension
            sign = 1.0 if (h % 2 == 0) else -1.0
            vec[idx] += sign * (1.0 / (1.0 + i * 0.1))

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Retrieves the embedding vector for a given text.
        Prioritizes Ollama and seamlessly falls back to deterministic vectorization if offline.
        """
        if not text:
            return np.zeros(self.dimension, dtype=np.float32)

        if self.is_service_available():
            try:
                response = self.embeddings_request(prompt=text)
                if 'embedding' in response and response['embedding']:
                    emb = np.array(response['embedding'], dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    return emb
            except Exception as e:
                logger.warning(f"Ollama embedding request failed ({e}). Employing deterministic fallback.")

        return self._fallback_deterministic_embedding(text)