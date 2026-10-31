import logging
import numpy as np
from typing import Optional

from .ollama_base import OllamaBase
from src.config import LLM_MODEL_NAME, OLLAMA_MAX_RETRIES, OLLAMA_BASE_DELAY
from src.transaction import Transaction

logger = logging.getLogger(__name__)

class AnomalyDetector(OllamaBase):
    """
    Detector de anomalías transaccionales.
    Implementa una arquitectura de decisión combinada:
    1. Si Ollama está activo: Razonamiento semántico mediante LLM guiado por prompt de riesgo.
    2. Fallback resiliente: Evaluación estadística sobre ratios de balance, intentos de login y duración.
    """
    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        max_retries: int = OLLAMA_MAX_RETRIES,
        base_delay: float = OLLAMA_BASE_DELAY
    ):
        super().__init__(model_name=model_name, max_retries=max_retries, base_delay=base_delay)
        self.logger = logging.getLogger(__name__)

    def _predict_with_llm(self, transaction: Transaction) -> Optional[bool]:
        """
        Ejecuta la inferencia de clasificación a través del LLM local de Ollama.
        """
        transaction_str = transaction.to_string_for_embedding()

        system_prompt = (
            "Eres un analista senior de ciberseguridad y fraude bancario. "
            "Tu tarea es analizar la transacción y clasificarla determinísticamente como FRAUDULENTO o NORMAL. "
            "Responde estrictamente con una sola palabra: 'FRAUDULENTO' o 'NORMAL'. "
            "Señales de fraude: transacciones que vacían más del 70% de la cuenta, múltiples intentos fallidos de login "
            "seguidos de transacciones rápidas (<40s), montos desproporcionados a la edad o saldo del cliente."
        )

        user_prompt = f"Evalúa la siguiente transacción:\n{transaction_str}\n\n¿Es FRAUDULENTO o NORMAL?"

        try:
            response = self.chat_request(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                options={'temperature': 0.1, 'num_predict': 10}
            )
            content = response.get('message', {}).get('content', '').strip().upper()

            if "FRAUDULENTO" in content:
                self.logger.info(f"Ollama clasifica como: FRAUDULENTO")
                return True
            elif "NORMAL" in content:
                self.logger.info(f"Ollama clasifica como: NORMAL")
                return False
            return None
        except Exception as e:
            self.logger.warning(f"Error al evaluar con Ollama LLM ({e}). Activando fallback heurístico.")
            return None

    def _rule_based_detection(self, transaction: Transaction) -> bool:
        """
        Lógica estadística y de reglas para el dataset de transacciones bancarias.
        Cubre los patrones clave: vaciado de cuenta, ataque de credenciales y desbalances de gasto.
        """
        amt = transaction.TransactionAmount
        bal = transaction.AccountBalance
        logins = transaction.LoginAttempts
        duration = transaction.TransactionDuration
        device = transaction.DeviceID.lower()

        # Regla 1: Vaciado o compromiso severo del balance (monto representa > 75% del saldo y monto relevante)
        if bal > 0 and (amt / bal) > 0.75 and amt > 1000:
            return True

        # Regla 2: Posible ataque de fuerza bruta/bot (>= 3 intentos de login y duración < 40s)
        if logins >= 3 and duration < 40:
            return True

        # Regla 3: Monto extremo con dispositivo o canal anómalo
        if amt > 3000 and ("unknown" in device or logins >= 2):
            return True

        # Regla 4: Transacción muy rápida para montos altos
        if amt > 2500 and duration < 15:
            return True

        # Regla 5: Estudiante o perfil joven (<25) con monto muy elevado (>1500) y saldo bajo
        if transaction.CustomerAge < 25 and amt > 1500 and (bal < 500 or (bal > 0 and amt / bal > 0.6)):
            return True

        return False

    def predict_fraud(self, embedding: np.ndarray, transaction: Transaction) -> bool:
        """
        Punto de entrada de detección de anomalías.
        Combina razonamiento LLM si Ollama está disponible con heurística financiera calibrada.
        """
        if self.is_service_available():
            decision = self._predict_with_llm(transaction)
            if decision is not None:
                return decision

        return self._rule_based_detection(transaction)