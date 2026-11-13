import logging
import random
import time
from typing import Dict, Any, Callable

try:
    import ollama
    OLLAMA_INSTALLED = True
except ImportError:
    OLLAMA_INSTALLED = False

logger = logging.getLogger(__name__)

class OllamaBase:
    """
    Base abstraction client for local Ollama interactions with exponential backoff and jitter.
    """
    def __init__(self, model_name: str, max_retries: int = 2, base_delay: float = 0.5):
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._is_available: bool = False
        self._checked_availability: bool = False

    def is_service_available(self) -> bool:
        """
        Non-blocking health probe checking if the local Ollama daemon is responsive.
        """
        if not OLLAMA_INSTALLED:
            return False
        if self._checked_availability:
            return self._is_available
        try:
            ollama.list()
            self._is_available = True
        except Exception:
            self._is_available = False
        self._checked_availability = True
        return self._is_available

    def _resilience_wrapper(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Wraps an Ollama invocation with exponential backoff and random jitter retry policy.
        """
        if not OLLAMA_INSTALLED:
            raise RuntimeError("The 'ollama' Python package is not installed in the environment.")

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries}: Error in '{func.__name__}': {e}"
                )
                if attempt == self.max_retries - 1:
                    logger.error(f"Retries exhausted for '{func.__name__}'.")
                    raise e
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.3)
                time.sleep(delay)

        if last_exception:
            raise last_exception

    def generate_request(self, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a text generation request with resilient retry handling."""
        def _call_generate():
            return ollama.generate(model=self.model_name, prompt=prompt, options=options)
        return self._resilience_wrapper(_call_generate)

    def embeddings_request(self, prompt: str) -> Dict[str, Any]:
        """Executes an embedding vector request with resilient retry handling."""
        def _call_embeddings():
            if hasattr(ollama, 'embed'):
                try:
                    res = ollama.embed(model=self.model_name, input=prompt)
                    embeddings = res.get('embeddings', [])
                    if embeddings:
                        return {'embedding': embeddings[0]}
                except Exception:
                    pass
            return ollama.embeddings(model=self.model_name, prompt=prompt)
        return self._resilience_wrapper(_call_embeddings)

    def chat_request(self, messages: list, options: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a chat completion request with resilient retry handling."""
        def _call_chat():
            return ollama.chat(model=self.model_name, messages=messages, options=options)
        return self._resilience_wrapper(_call_chat)