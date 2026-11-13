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
    """Returns a singleton instance of EmbeddingManager."""
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
    Loads fraud rules from the JSON catalog and attaches normalized embedding
    vectors for each rule to power real vector-based semantic RAG retrieval.
    """
    global _cached_rules
    if _cached_rules is not None and not force_reload:
        return _cached_rules

    path = Path(rules_file_path) if rules_file_path else FRAUD_RULES_PATH

    if not path.exists():
        logger.error(f"Fraud rules file not found at: {path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_rules = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read rules from {path}: {e}")
        return []

    manager = get_embedding_manager_instance(model_name=model_name)
    embedded_rules = []

    logger.info(f"Generating dense embedding vectors for {len(raw_rules)} fraud rules...")
    for rule in raw_rules:
        rule_copy = rule.copy()
        keywords_str = ", ".join(rule.get("keywords", []))
        tags_str = ", ".join(rule.get("tags", []))
        
        # Enriched semantic representation for cosine similarity
        rule_text = (
            f"Rule: {rule.get('id', '')}. Description: {rule.get('description', '')}. "
            f"Keywords: {keywords_str}. Tags: {tags_str}."
        )
        embedding = manager.get_embedding(rule_text)
        rule_copy["embedding"] = embedding
        rule_copy["semantic_text"] = rule_text
        embedded_rules.append(rule_copy)

    _cached_rules = embedded_rules
    logger.info("Rule embeddings successfully computed and cached in memory.")
    return embedded_rules