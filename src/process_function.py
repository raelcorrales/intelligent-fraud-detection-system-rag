import logging
from typing import List, Dict, Any, Optional

from src.transaction import Transaction
from src.fraud_detection_service.embedding_manager import EmbeddingManager
from src.fraud_detection_service.anomaly_detector import AnomalyDetector
from src.fraud_detection_service.rag_retriever import RAGRetriever
from src.fraud_detection_service.llm_explainer import LLMExplainer
from src.fraud_detection_service.embedding_rule_utils import load_and_embed_rules

logger = logging.getLogger(__name__)

# Singletons en memoria para optimizar rendimiento en inferencias recurrentes
_default_embedding_manager: Optional[EmbeddingManager] = None
_default_anomaly_detector: Optional[AnomalyDetector] = None
_default_llm_explainer: Optional[LLMExplainer] = None

def get_or_create_components():
    """Garantiza la reutilización de componentes sin reinicializaciones costosas."""
    global _default_embedding_manager, _default_anomaly_detector, _default_llm_explainer
    if _default_embedding_manager is None:
        _default_embedding_manager = EmbeddingManager()
    if _default_anomaly_detector is None:
        _default_anomaly_detector = AnomalyDetector()
    if _default_llm_explainer is None:
        _default_llm_explainer = LLMExplainer()
    return _default_embedding_manager, _default_anomaly_detector, _default_llm_explainer

def process_transaction(
    transaction_data: Dict[str, Any],
    fraud_rules: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Orquestador principal del pipeline de detección y explicabilidad de fraude:
    1. Parseo y tipado de la transacción bancaria.
    2. Vectorización densa semántica (Embedding).
    3. Clasificación de anomalías (LLM + Heurística financiera calibrada).
    4. Recuperación RAG por similitud coseno vectorial de reglas de riesgo.
    5. Síntesis de justificación en lenguaje natural para analistas.
    """
    try:
        transaction = Transaction(transaction_data)
    except Exception as e:
        logger.error(f"Error al parsear los datos de la transacción: {e}")
        return {'error': f"Datos de transacción inválidos: {e}"}

    # Cargar reglas con embeddings si no se proporcionan
    if fraud_rules is None:
        fraud_rules = load_and_embed_rules()

    embedding_manager, anomaly_detector, llm_explainer = get_or_create_components()
    rag_retriever = RAGRetriever(fraud_rules=fraud_rules)

    transaction_details_str = transaction.to_string_for_embedding()

    # PASO 1: Generación de Vector de Embedding
    transaction_embedding = embedding_manager.get_embedding(transaction_details_str)

    # PASO 2: Detección de Anomalías
    is_fraudulent = anomaly_detector.predict_fraud(transaction_embedding, transaction)

    return_data = transaction_data.copy()
    return_data.update({
        'is_fraudulent': is_fraudulent,
        'transaction_details_str_for_llm': transaction_details_str,
        'transaction_embedding_shape': list(transaction_embedding.shape) if hasattr(transaction_embedding, 'shape') else [],
    })

    if is_fraudulent:
        logger.warning(
            f"🚨 Transacción Sospechosa Detectada: ID {transaction.TransactionID} | "
            f"Monto: ${transaction.TransactionAmount:.2f} | Cuenta: {transaction.AccountID}"
        )

        # PASO 3: Recuperación Semántica de Reglas (RAG con Similitud Coseno)
        fraud_context = rag_retriever.retrieve_context(
            transaction=transaction,
            transaction_embedding=transaction_embedding
        )
        detected_risk_level = fraud_context.get('risk_level', 'HIGH')
        detected_rule_tags = fraud_context.get('tags', [])
        rule_descriptions = fraud_context.get('rule_descriptions', [])
        mitigation_steps = fraud_context.get('mitigation_steps', [])
        matched_rules_details = fraud_context.get('matched_rules_details', [])

        # PASO 4: Generación de Explicación Técnica para Auditoría
        explanation = llm_explainer.generate_explanation(transaction_details_str, fraud_context)

        return_data.update({
            'fraud_context': rule_descriptions,
            'detected_risk_level': detected_risk_level,
            'detected_rule_tags': detected_rule_tags,
            'mitigation_steps': mitigation_steps,
            'matched_rules_details': matched_rules_details,
            'explanation': explanation,
        })
    else:
        logger.info(f"✅ Transacción Normal: ID {transaction.TransactionID}")
        return_data.update({
            'fraud_context': None,
            'detected_risk_level': "NONE",
            'detected_rule_tags': [],
            'mitigation_steps': [],
            'matched_rules_details': [],
            'explanation': "Transacción dentro de los parámetros habituales de comportamiento.",
        })

    return return_data

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    print("\n--- Ejecutando prueba de inferencia de transacción sospechosa ---")
    sample_transaction = {
        "TransactionID": "DEMO_TXN_001",
        "AccountID": "ACC_DEMO_99",
        "TransactionAmount": 3800.0,
        "TransactionDate": "2024-06-15 02:30:00",
        "TransactionType": "Débito",
        "Location": "Unknown",
        "DeviceID": "DEV_UNKNOWN",
        "IP Address": "10.0.0.12",
        "MerchantID": "MERCH_ELECTRONICS",
        "Channel": "Online",
        "CustomerAge": 21,
        "CustomerOccupation": "Student",
        "TransactionDuration": 15,
        "LoginAttempts": 4,
        "AccountBalance": 4000.0,
        "PreviousTransactionDate": "2024-06-14 10:00:00"
    }
    result = process_transaction(sample_transaction)
    print("\nResultado del análisis:")
    print(f"- Es Fraudulenta: {result['is_fraudulent']}")
    print(f"- Nivel de Riesgo: {result['detected_risk_level']}")
    print(f"- Tags: {result['detected_rule_tags']}")
    print(f"- Mitigación: {result['mitigation_steps']}")
    print(f"- Explicación: {result['explanation']}")