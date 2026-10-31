import logging
import numpy as np
from typing import List, Dict, Any, Optional

from .ollama_base import OllamaBase
from src.config import (
    LLM_MODEL_NAME,
    RAG_SIMILARITY_THRESHOLD,
    RAG_TOP_K_RULES,
    OLLAMA_MAX_RETRIES,
    OLLAMA_BASE_DELAY
)

logger = logging.getLogger(__name__)

RISK_LEVEL_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}

class RAGRetriever(OllamaBase):
    """
    Recuperador RAG semántico para reglas de fraude financiero.
    Combina:
    1. Búsqueda vectorial por similitud coseno sobre embeddings densos.
    2. Matching por heurísticas de dominio financiero estructuradas (grounding).
    3. Clasificación de severidad y consolidación de mitigaciones operativas.
    """
    def __init__(
        self,
        fraud_rules: List[Dict[str, Any]],
        model_name: str = LLM_MODEL_NAME,
        similarity_threshold: float = RAG_SIMILARITY_THRESHOLD,
        top_k: int = RAG_TOP_K_RULES,
        max_retries: int = OLLAMA_MAX_RETRIES,
        base_delay: float = OLLAMA_BASE_DELAY
    ):
        super().__init__(model_name=model_name, max_retries=max_retries, base_delay=base_delay)
        self.fraud_rules = fraud_rules
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calcula la similitud coseno entre dos vectores numéricos."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def retrieve_context(
        self,
        transaction,
        transaction_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Recupera el contexto semántico de reglas de fraude más afines a la transacción.
        """
        matched_rules = []

        # 1. Búsqueda Vectorial por Similitud Coseno
        if transaction_embedding is not None:
            scored_rules = []
            for rule in self.fraud_rules:
                rule_emb = rule.get("embedding")
                if rule_emb is not None and isinstance(rule_emb, np.ndarray):
                    score = self.cosine_similarity(transaction_embedding, rule_emb)
                    scored_rules.append((score, rule))

            scored_rules.sort(key=lambda x: x[0], reverse=True)

            for score, rule in scored_rules[:self.top_k]:
                if score >= self.similarity_threshold or (scored_rules and score > 0.25):
                    rule_copy = rule.copy()
                    rule_copy["similarity_score"] = round(score, 4)
                    matched_rules.append(rule_copy)

        # 2. Reglas Financieras Estructuradas de Dominio (Grounding de Auditoría)
        tx_str = transaction.to_string_for_embedding().lower()
        amt = getattr(transaction, "TransactionAmount", 0.0)
        bal = getattr(transaction, "AccountBalance", 0.0)
        logins = getattr(transaction, "LoginAttempts", 0)
        device = getattr(transaction, "DeviceID", "").lower()

        for rule in self.fraud_rules:
            rid = rule.get("id", "")
            triggered = False

            # Vaciado de cuenta
            if rid == "RULE_006_LARGE_PERCENTAGE_OF_ACCOUNT_BALANCE" and bal > 0 and (amt / bal) > 0.70:
                triggered = True
            # Intentos fallidos de autenticación
            elif rid == "RULE_003_MULTIPLE_FAILED_LOGIN_ATTEMPTS" and logins >= 3:
                triggered = True
            # Alto valor o categoría inusual
            elif rid == "RULE_001_HIGH_VALUE_UNUSUAL_CATEGORY" and amt >= 2000:
                triggered = True
            # Dispositivo desconocido
            elif rid == "RULE_005_UNUSUAL_DEVICE_OR_CHANNEL" and "unknown" in device:
                triggered = True
            # Coincidencia por keywords
            elif any(kw.lower() in tx_str for kw in rule.get("keywords", [])):
                triggered = True

            if triggered and rid not in [r.get("id") for r in matched_rules]:
                rule_copy = rule.copy()
                rule_copy["similarity_score"] = 0.85  # Score alto para reglas de negocio directas
                matched_rules.append(rule_copy)

        # Desduplicación preservando mayor score
        deduped = {}
        for r in matched_rules:
            rid = r.get("id")
            if rid not in deduped or r.get("similarity_score", 0) > deduped[rid].get("similarity_score", 0):
                deduped[rid] = r
        matched_rules = list(deduped.values())

        if matched_rules:
            max_rule = max(
                matched_rules,
                key=lambda r: RISK_LEVEL_ORDER.get(r.get("risk_level", "LOW"), 1)
            )
            risk_level = max_rule.get("risk_level", "HIGH")
            tags = list({tag for rule in matched_rules for tag in rule.get("tags", [])})
            rule_descriptions = [rule["description"] for rule in matched_rules]
            rule_ids = [rule["id"] for rule in matched_rules]
            mitigation_steps = list({step for rule in matched_rules for step in rule.get("mitigation_steps", [])})

            return {
                "matched_rule_ids": rule_ids,
                "risk_level": risk_level,
                "tags": tags,
                "rule_descriptions": rule_descriptions,
                "mitigation_steps": mitigation_steps,
                "matched_rules_details": [
                    {
                        "id": r["id"],
                        "risk_level": r.get("risk_level"),
                        "similarity_score": r.get("similarity_score", 0.0),
                        "description": r.get("description")
                    }
                    for r in matched_rules
                ],
                "explanation_base": f"Activación semántica y de negocio de {len(matched_rules)} regla(s) de riesgo."
            }

        # 3. Fallback general para alertas sin regla directa
        return {
            "matched_rule_ids": [],
            "risk_level": "HIGH",
            "tags": ["anomalous-behavior", "risk-mitigation"],
            "rule_descriptions": ["Patrón transaccional atípico respecto al perfil demográfico o de cuenta."],
            "mitigation_steps": ["Poner la transacción en revisión manual y solicitar MFA."],
            "matched_rules_details": [],
            "explanation_base": "Anomalía detectada sin regla específica directa."
        }
