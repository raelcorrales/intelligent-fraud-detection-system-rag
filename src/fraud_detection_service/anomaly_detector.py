import logging
import numpy as np
from typing import Optional

from .ollama_base import OllamaBase
from src.config import LLM_MODEL_NAME, OLLAMA_MAX_RETRIES, OLLAMA_BASE_DELAY
from src.transaction import Transaction

logger = logging.getLogger(__name__)

class AnomalyDetector(OllamaBase):
    """
    Transaction anomaly detector.
    Implements a hybrid decision architecture:
    1. If Ollama is active: Semantic reasoning via local LLM with structured fraud prompt.
    2. Resilient fallback: Domain statistical heuristics based on balance ratios, login attempts, and durations.
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
        Executes classification inference through the local Ollama LLM.
        """
        transaction_str = transaction.to_string_for_embedding()

        system_prompt = (
            "You are a senior banking cybersecurity and fraud detection analyst. "
            "Your task is to analyze the transaction details and classify it deterministically as FRAUDULENT or NORMAL. "
            "Respond strictly with a single word: 'FRAUDULENT' or 'NORMAL'. "
            "Fraud indicators: transactions draining over 70% of account balance, multiple failed login attempts "
            "followed by rapid transaction execution (<40s), amounts disproportionate to customer age or balance."
        )

        user_prompt = f"Evaluate the following transaction:\n{transaction_str}\n\nIs it FRAUDULENT or NORMAL?"

        try:
            response = self.chat_request(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                options={'temperature': 0.1, 'num_predict': 10}
            )
            content = response.get('message', {}).get('content', '').strip().upper()

            if "FRAUDULENT" in content:
                self.logger.info("Ollama classified as: FRAUDULENT")
                return True
            elif "NORMAL" in content:
                self.logger.info("Ollama classified as: NORMAL")
                return False
            return None
        except Exception as e:
            self.logger.warning(f"Error during Ollama LLM evaluation ({e}). Falling back to heuristic rules.")
            return None

    def _rule_based_detection(self, transaction: Transaction) -> bool:
        """
        Calibrated domain heuristics for bank transaction datasets.
        Evaluates account draining, credential attacks, and behavioral spending imbalances.
        """
        amt = transaction.TransactionAmount
        bal = transaction.AccountBalance
        logins = transaction.LoginAttempts
        duration = transaction.TransactionDuration
        device = transaction.DeviceID.lower()

        # Rule 1: Severe account balance depletion (> 75% of balance with substantial amount)
        if bal > 0 and (amt / bal) > 0.75 and amt > 1000:
            return True

        # Rule 2: Brute force / automated bot signature (>= 3 login attempts and duration < 40s)
        if logins >= 3 and duration < 40:
            return True

        # Rule 3: Extreme transaction amount with unknown device or repeated login attempts
        if amt > 3000 and ("unknown" in device or logins >= 2):
            return True

        # Rule 4: Abnormally fast execution for high-value transaction
        if amt > 2500 and duration < 15:
            return True

        # Rule 5: Student or young demographic (<25) with high amount (>1500) and low/drained balance
        if transaction.CustomerAge < 25 and amt > 1500 and (bal < 500 or (bal > 0 and amt / bal > 0.6)):
            return True

        return False

    def predict_fraud(self, embedding: np.ndarray, transaction: Transaction) -> bool:
        """
        Entry point for transaction anomaly detection.
        Combines LLM reasoning when Ollama is available with calibrated financial heuristics.
        """
        if self.is_service_available():
            decision = self._predict_with_llm(transaction)
            if decision is not None:
                return decision

        return self._rule_based_detection(transaction)