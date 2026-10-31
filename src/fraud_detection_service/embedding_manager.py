import hashlib
import logging
import numpy as np
from typing import Optional

from .ollama_base import OllamaBase
from src.config import EMBEDDING_MODEL_NAME, OLLAMA_MAX_RETRIES, OLLAMA_BASE_DELAY

logger = logging.getLogger(__name__)

class EmbeddingManager(OllamaBase):
    """
    Gestor de embeddings para el sistema de detección de fraude.
    Genera representaciones vectoriales densas utilizando Ollama (nomic-embed-text u otro),
    con fallback determinista en caso de que el daemon local no esté activo.
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
        Genera un embedding denso normalizado (L2) basado en hashing criptográfico
        para garantizar continuidad en tests y entornos sin daemon de Ollama activo.
        """
        words = text.lower().split()
        vec = np.zeros(self.dimension, dtype=np.float32)

        for i, word in enumerate(words):
            # Seed reproducible por palabra
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
        Obtiene el vector de embedding de un texto.
        Prioriza Ollama y conmuta de forma transparente al fallback determinista si no está accesible.
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
                logger.warning(f"Ollama embedding falló ({e}). Usando fallback semántico determinista.")

        return self._fallback_deterministic_embedding(text)