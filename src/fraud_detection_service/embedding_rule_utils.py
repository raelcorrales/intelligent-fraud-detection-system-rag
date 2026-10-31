import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import FRAUD_RULES_PATH, EMBEDDING_MODEL_NAME
from src.fraud_detection_service.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

_cached_rules: Optional[List[Dict[str, Any]]] = None
_embedding_manager: Optional[EmbeddingManager] = None

def get_embedding_manager_instance(model_name: str = EMBEDDING_MODEL_NAME) -> EmbeddingManager:
    """Retorna una instancia singleton de EmbeddingManager."""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(model_name=model_name)
    return _embedding_manager

def load_and_embed_rules(
    model_name: str = EMBEDDING_MODEL_NAME,
    rules_file_path: Optional[str] = None,
    force_reload: bool = False
) -> List[Dict[str, Any]]:
    """
    Carga las reglas de fraude desde el archivo JSON y calcula/adjunta
    un vector de embedding normalizado para cada regla para habilitar RAG semántico real.
    """
    global _cached_rules
    if _cached_rules is not None and not force_reload:
        return _cached_rules

    path = Path(rules_file_path) if rules_file_path else FRAUD_RULES_PATH

    if not path.exists():
        logger.error(f"El archivo de reglas no existe en: {path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_rules = json.load(f)
    except Exception as e:
        logger.error(f"Error al leer reglas de {path}: {e}")
        return []

    manager = get_embedding_manager_instance(model_name=model_name)
    embedded_rules = []

    logger.info(f"Generando embeddings vectoriales para {len(raw_rules)} reglas de fraude...")
    for rule in raw_rules:
        rule_copy = rule.copy()
        keywords_str = ", ".join(rule.get("keywords", []))
        tags_str = ", ".join(rule.get("tags", []))
        
        # Representación textual enriquecida para similitud semántica
        rule_text = (
            f"Regla: {rule.get('id', '')}. Descripción: {rule.get('description', '')}. "
            f"Palabras clave: {keywords_str}. Tags: {tags_str}."
        )
        embedding = manager.get_embedding(rule_text)
        rule_copy["embedding"] = embedding
        rule_copy["semantic_text"] = rule_text
        embedded_rules.append(rule_copy)

    _cached_rules = embedded_rules
    logger.info(f"Embeddings de reglas listos y cacheados exitosamente.")
    return embedded_rules