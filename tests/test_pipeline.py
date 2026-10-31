import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.transaction import Transaction
from src.fraud_detection_service.embedding_manager import EmbeddingManager
from src.fraud_detection_service.embedding_rule_utils import load_and_embed_rules
from src.fraud_detection_service.rag_retriever import RAGRetriever
from src.fraud_detection_service.anomaly_detector import AnomalyDetector
from src.process_function import process_transaction
from main_api import app

@pytest.fixture
def sample_normal_transaction_data():
    return {
        "TransactionID": "TXN_NORM_001",
        "AccountID": "ACC_100",
        "TransactionAmount": 45.0,
        "TransactionDate": "2024-05-10 12:00:00",
        "TransactionType": "Débito",
        "Location": "New York",
        "DeviceID": "DEV_US_1",
        "IP Address": "192.168.1.10",
        "MerchantID": "MERCH_GROCERY",
        "Channel": "Online",
        "CustomerAge": 38,
        "CustomerOccupation": "Engineer",
        "TransactionDuration": 50,
        "LoginAttempts": 1,
        "AccountBalance": 5000.0,
        "PreviousTransactionDate": "2024-05-09 11:00:00"
    }

@pytest.fixture
def sample_fraud_transaction_data():
    return {
        "TransactionID": "TXN_FRAUD_001",
        "AccountID": "ACC_200",
        "TransactionAmount": 4200.0,
        "TransactionDate": "2024-05-10 03:15:00",
        "TransactionType": "Débito",
        "Location": "Unknown",
        "DeviceID": "DEV_UNKNOWN",
        "IP Address": "10.0.0.99",
        "MerchantID": "MERCH_JEWELRY",
        "Channel": "Online",
        "CustomerAge": 22,
        "CustomerOccupation": "Student",
        "TransactionDuration": 10,
        "LoginAttempts": 4,
        "AccountBalance": 4500.0,  # Vaciado del 93% de la cuenta
        "PreviousTransactionDate": "2024-05-01 10:00:00"
    }

def test_transaction_model(sample_normal_transaction_data):
    tx = Transaction(sample_normal_transaction_data)
    assert tx.TransactionID == "TXN_NORM_001"
    assert tx.TransactionAmount == 45.0
    text_repr = tx.to_string_for_embedding()
    assert "TXN_NORM_001" in text_repr
    assert "ACC_100" in text_repr

def test_embedding_manager():
    manager = EmbeddingManager(dimension=768)
    emb = manager.get_embedding("Prueba de vectorización de transacción bancaria")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (768,)
    # Verificar normalización unitaria (L2 norm ~ 1.0)
    norm = np.linalg.norm(emb)
    assert pytest.approx(norm, 0.01) == 1.0

def test_load_and_embed_rules():
    rules = load_and_embed_rules()
    assert len(rules) > 0
    first_rule = rules[0]
    assert "id" in first_rule
    assert "embedding" in first_rule
    assert isinstance(first_rule["embedding"], np.ndarray)
    assert first_rule["embedding"].shape == (768,)

def test_rag_retriever_cosine_similarity(sample_fraud_transaction_data):
    rules = load_and_embed_rules()
    retriever = RAGRetriever(fraud_rules=rules, similarity_threshold=0.3)
    tx = Transaction(sample_fraud_transaction_data)
    manager = EmbeddingManager()
    tx_emb = manager.get_embedding(tx.to_string_for_embedding())

    context = retriever.retrieve_context(tx, transaction_embedding=tx_emb)
    assert "risk_level" in context
    assert "tags" in context
    assert "rule_descriptions" in context
    assert context["risk_level"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

def test_anomaly_detector_classification(sample_normal_transaction_data, sample_fraud_transaction_data):
    detector = AnomalyDetector()
    manager = EmbeddingManager()

    tx_norm = Transaction(sample_normal_transaction_data)
    emb_norm = manager.get_embedding(tx_norm.to_string_for_embedding())
    is_fraud_normal = detector.predict_fraud(emb_norm, tx_norm)
    assert is_fraud_normal is False

    tx_fraud = Transaction(sample_fraud_transaction_data)
    emb_fraud = manager.get_embedding(tx_fraud.to_string_for_embedding())
    is_fraud_suspicious = detector.predict_fraud(emb_fraud, tx_fraud)
    assert is_fraud_suspicious is True

def test_process_transaction_end_to_end(sample_fraud_transaction_data, sample_normal_transaction_data):
    # Transacción sospechosa
    result_fraud = process_transaction(sample_fraud_transaction_data)
    assert result_fraud["is_fraudulent"] is True
    assert result_fraud["detected_risk_level"] in ["CRITICAL", "HIGH", "MEDIUM"]
    assert len(result_fraud["explanation"]) > 0

    # Transacción normal
    result_normal = process_transaction(sample_normal_transaction_data)
    assert result_normal["is_fraudulent"] is False
    assert result_normal["detected_risk_level"] == "NONE"

def test_fastapi_endpoints(sample_normal_transaction_data, sample_fraud_transaction_data):
    client = TestClient(app)
    
    # 1. Root & Health
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert resp_root.json()["service"] == "FraudAI Detection API"

    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "healthy"

    # 2. Inferencia POST normal
    resp_norm = client.post("/process-transaction/", json=sample_normal_transaction_data)
    assert resp_norm.status_code == 200
    json_norm = resp_norm.json()
    assert json_norm["is_fraudulent"] is False
    assert json_norm["detected_risk_level"] == "NONE"

    # 3. Inferencia POST fraude
    resp_fraud = client.post("/process-transaction/", json=sample_fraud_transaction_data)
    assert resp_fraud.status_code == 200
    json_fraud = resp_fraud.json()
    assert json_fraud["is_fraudulent"] is True
    assert len(json_fraud["explanation"]) > 0
