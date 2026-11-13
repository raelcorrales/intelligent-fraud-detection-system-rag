import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict

from src.config import FRAUD_RULES_PATH, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME
from src.fraud_detection_service.embedding_rule_utils import load_and_embed_rules
from src.fraud_detection_service.ollama_base import OllamaBase
from src.process_function import process_transaction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FraudAI Detection & Explainability API",
    description="Intelligent financial fraud detection system powered by dense vector RAG and local LLM explainability.",
    version="0.1.0"
)

# Preload fraud rules catalog on application startup
logger.info("Preloading fraud rules catalog with dense embeddings...")
embedded_rules = load_and_embed_rules(rules_file_path=str(FRAUD_RULES_PATH))
logger.info(f"Catalog loaded: {len(embedded_rules)} vector rules cached in memory.")

class TransactionSchema(BaseModel):
    """Input validation schema for incoming banking transactions."""
    model_config = ConfigDict(populate_by_name=True)

    TransactionID: str = Field(..., examples=["TXN10293"])
    AccountID: str = Field(..., examples=["ACC9821"])
    TransactionAmount: float = Field(..., ge=0.0, examples=[2450.50])
    TransactionDate: str = Field(..., examples=["2024-05-12 14:35:00"])
    TransactionType: str = Field(..., examples=["Debit"])
    Location: str = Field(default="N/A", examples=["New York"])
    DeviceID: str = Field(default="N/A", examples=["DEV_048"])
    IPAddress: str = Field(default="N/A", alias="IP Address", examples=["192.168.1.45"])
    MerchantID: str = Field(default="N/A", examples=["MERCH_003"])
    Channel: str = Field(default="Online", examples=["Online"])
    CustomerAge: int = Field(default=35, ge=0, examples=[32])
    CustomerOccupation: str = Field(default="Engineer", examples=["Engineer"])
    TransactionDuration: int = Field(default=30, ge=0, examples=[12])
    LoginAttempts: int = Field(default=1, ge=0, examples=[3])
    AccountBalance: float = Field(default=0.0, ge=0.0, examples=[1200.00])
    PreviousTransactionDate: Optional[str] = Field(default="N/A", examples=["2024-05-11 10:00:00"])

class RuleDetail(BaseModel):
    id: str
    risk_level: Optional[str] = None
    similarity_score: Optional[float] = None
    description: Optional[str] = None

class FraudAnalysisResponse(BaseModel):
    """Structured, auditable fraud assessment response."""
    TransactionID: str
    TransactionAmount: float
    is_fraudulent: bool
    detected_risk_level: str
    detected_rule_tags: List[str]
    mitigation_steps: List[str]
    explanation: str
    matched_rules_details: Optional[List[RuleDetail]] = None
    transaction_embedding_shape: Optional[List[int]] = None

@app.get("/", tags=["General"])
def root():
    return {
        "service": "FraudAI Detection API",
        "version": "0.1.0",
        "status": "ready",
        "models": {
            "embedding": EMBEDDING_MODEL_NAME,
            "llm": LLM_MODEL_NAME
        },
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Health"])
def health_check():
    ollama_client = OllamaBase(model_name=LLM_MODEL_NAME)
    ollama_active = ollama_client.is_service_available()
    return {
        "status": "healthy",
        "ollama_connected": ollama_active,
        "rules_cached": len(embedded_rules)
    }

@app.post(
    "/process-transaction/",
    response_model=FraudAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Fraud Inference"]
)
async def process_single_transaction(transaction_in: TransactionSchema):
    """
    Processes an individual banking transaction through the hybrid RAG + LLM pipeline.
    Returns risk classification, triggered semantic rules, and audit explanation.
    """
    try:
        data_dict = transaction_in.model_dump(by_alias=True)
        result = process_transaction(transaction_data=data_dict, fraud_rules=embedded_rules)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result["error"]
            )

        return FraudAnalysisResponse(
            TransactionID=result.get("TransactionID", transaction_in.TransactionID),
            TransactionAmount=float(result.get("TransactionAmount", transaction_in.TransactionAmount)),
            is_fraudulent=bool(result.get("is_fraudulent", False)),
            detected_risk_level=result.get("detected_risk_level", "NONE"),
            detected_rule_tags=result.get("detected_rule_tags", []),
            mitigation_steps=result.get("mitigation_steps", []),
            explanation=result.get("explanation", "No explanation available."),
            matched_rules_details=[
                RuleDetail(**r) for r in result.get("matched_rules_details", [])
            ],
            transaction_embedding_shape=result.get("transaction_embedding_shape", [])
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled inference failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {str(e)}"
        )