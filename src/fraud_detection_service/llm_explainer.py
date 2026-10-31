import logging
from typing import Dict, Any

from .ollama_base import OllamaBase
from src.config import LLM_MODEL_NAME, LLM_TEMPERATURE, OLLAMA_MAX_RETRIES, OLLAMA_BASE_DELAY

logger = logging.getLogger(__name__)

class LLMExplainer(OllamaBase):
    """
    Generador de explicaciones en lenguaje natural para analistas de fraude.
    Combina generación contextual con Ollama LLM y un generador sintético estructurado
    de respaldo para máxima disponibilidad en entornos de producción.
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
        Genera una explicación técnica rigurosa fundamentada en las reglas RAG activadas
        cuando el servicio local de LLM no está accesible.
        """
        risk_level = fraud_context.get("risk_level", "HIGH")
        tags = ", ".join(fraud_context.get("tags", ["Comportamiento Anómalo"]))
        descriptions = fraud_context.get("rule_descriptions", [])
        rules_str = " ".join(descriptions) if descriptions else "Se detectó desviación respecto al patrón habitual."
        mitigations = fraud_context.get("mitigation_steps", ["Revisión manual por equipo de riesgo."])
        mitigations_str = "; ".join(mitigations)

        return (
            f"Alerta de Fraude ({risk_level}): La transacción presenta patrones sospechosos etiquetados como [{tags}]. "
            f"Fundamento de auditoría: {rules_str} "
            f"Acciones de mitigación sugeridas: {mitigations_str}"
        )

    def generate_explanation(self, transaction_details: str, fraud_context: Dict[str, Any]) -> str:
        """
        Genera la justificación técnica en lenguaje natural sobre la alerta de fraude.
        """
        if self.is_service_available():
            rules_context_str = "\n".join(fraud_context.get("rule_descriptions", []))
            prompt = (
                "Eres un perito analista de riesgos en ciberseguridad financiera.\n"
                "Genera una justificación técnica concisa (máximo 3 párrafos) explicando por qué esta transacción "
                "fue clasificada como sospechosa, citando el contexto de reglas activadas y sugiriendo mitigaciones.\n\n"
                f"🧾 DETALLES DE LA TRANSACCIÓN:\n{transaction_details}\n\n"
                f"📚 REGLAS Y CONTEXTO RAG ACTIVADO:\n{rules_context_str}\n"
                f"Nivel de Riesgo: {fraud_context.get('risk_level', 'HIGH')}\n\n"
                "🔍 INFORME DE EVALUACIÓN:\n"
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
                logger.warning(f"Fallo en generación con LLM ({e}). Empleando generador estructurado de respaldo.")

        return self._generate_fallback_explanation(transaction_details, fraud_context)