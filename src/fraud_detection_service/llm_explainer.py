import logging
from typing import Dict, Any

from .ollama_base import OllamaBase
from src.config import LLM_MODEL_NAME, LLM_TEMPERATURE, OLLAMA_MAX_RETRIES, OLLAMA_BASE_DELAY

logger = logging.getLogger(__name__)

class LLMExplainer(OllamaBase):
    """
    Natural language explanation generator for fraud analysts.
    Combines contextual generation with Ollama LLM and a structured synthetic
    generator fallback for high availability in production environments.
    """
    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        temperature: float = LLM_TEMPERATURE,
        max_retries: int = OLLAMA_MAX_RETRIES,
        base_delay: float = OLLAMA_BASE_DELAY
    ):
        super().__init__(model_name=model_name, max_retries=max_retries, base_delay=base_delay)
        self.temperature = temperature

    def _generate_fallback_explanation(self, transaction_details: str, fraud_context: Dict[str, Any]) -> str:
        """
        Generates a rigorous technical rationale grounded in activated RAG rules
        when the local LLM service is offline or unreachable.
        """
        risk_level = fraud_context.get("risk_level", "HIGH")
        tags = ", ".join(fraud_context.get("tags", ["Anomalous Behavior"]))
        descriptions = fraud_context.get("rule_descriptions", [])
        rules_str = " ".join(descriptions) if descriptions else "Deviation detected from baseline behavior."
        mitigations = fraud_context.get("mitigation_steps", ["Manual review by fraud risk operations team."])
        mitigations_str = "; ".join(mitigations)

        return (
            f"Fraud Alert ({risk_level}): Transaction exhibits anomalous patterns tagged as [{tags}]. "
            f"Audit rationale: {rules_str} "
            f"Recommended mitigation actions: {mitigations_str}"
        )

    def generate_explanation(self, transaction_details: str, fraud_context: Dict[str, Any]) -> str:
        """
        Generates a natural language technical audit report justifying the fraud alert.
        """
        if self.is_service_available():
            rules_context_str = "\n".join(fraud_context.get("rule_descriptions", []))
            prompt = (
                "You are an expert risk and financial cybersecurity analyst.\n"
                "Provide a concise technical rationale (maximum 3 paragraphs) explaining why this transaction "
                "was flagged as suspicious, citing the activated rules context and suggesting mitigation actions.\n\n"
                f"🧾 TRANSACTION DETAILS:\n{transaction_details}\n\n"
                f"📚 ACTIVATED RAG RULES CONTEXT:\n{rules_context_str}\n"
                f"Risk Level: {fraud_context.get('risk_level', 'HIGH')}\n\n"
                "🔍 EVALUATION REPORT:\n"
            )

            try:
                response = self.generate_request(
                    prompt=prompt,
                    options={'temperature': self.temperature, 'num_predict': 250}
                )
                generated_text = response.get('response', '').strip()
                if generated_text:
                    return generated_text
            except Exception as e:
                logger.warning(f"LLM generation failed ({e}). Employing structured fallback generator.")

        return self._generate_fallback_explanation(transaction_details, fraud_context)