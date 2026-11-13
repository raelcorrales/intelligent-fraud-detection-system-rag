# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-10-31

### Version Freeze — First Usable Release
First official consolidated and frozen release ready for end-to-end basic usage across local and demonstration environments.

### Added
- **Vector Store & Cosine Similarity Search**: Real dense vector cosine similarity search ($\text{cosine similarity}$) in `RAGRetriever` over 768-dimensional normalized ($L_2$) embeddings.
- **Strict API Data Validation**: Strongly-typed validation schemas with Pydantic V2 (`TransactionSchema` and `FraudAnalysisResponse`) with field aliasing for spaced names (`IP Address`).
- **Monitoring Endpoint `/health`**: FastAPI health probe checking local Ollama daemon connectivity and the count of cached rules in memory.
- **Automated Test Suite**: 7 unit and integration tests in `tests/test_pipeline.py` covering model parsing, vectorization, cosine similarity, anomaly detection, and REST API invocation.
- **CLI Demo Execution**: Direct runnable demo block in `src/process_function.py` invocable via `python -m src.process_function`.
- **Centralized Version Module**: Created `src/__version__.py`.

### Changed
- `AnomalyDetector`: Replaced hardcoded bypass with a hybrid decision architecture (Ollama LLM inference + calibrated financial heuristics on Kaggle dataset).
- `load_and_embed_rules`: Real vectorization and in-memory caching for the `fraud_rules.json` catalog.
- `LLMExplainer`: Structured technical audit reporting with anti-hallucination grounded prompts.
- `main_api.py`: Replaced fragile relative file paths with resilient resolution via `pathlib.Path` and `src.config`.
- `README.md`: Architecture Mermaid diagrams, quick-start guide, and endpoint specifications in English.

---

## [0.1.0-beta] - 2026-10-31

### Added
- Unified configuration module `src/config.py`.
- Deterministic fallback logic for embeddings and explanation generation in offline environments.
- Singleton caching in `process_function.py` to optimize latency during recurring invocations.

### Fixed
- Logging call syntax error in `AnomalyDetector` (`self.logger.errot`).
- Pydantic V2 deprecation warnings (`example` replaced by `examples`).
- Environment variable naming discrepancies for Ollama models.

---

## [0.1.0-alpha] - 2026-10-31

### Added
- Base project scaffold and transactional data modeling via `Transaction` class.
- Initial catalog of 9 banking fraud rules in `data/fraud_rules.json`.
- `OllamaBase` abstraction layer with exponential backoff and random jitter.
- Interactive cybersecurity dashboard in Streamlit (`src/app_fraude.py`) with USA choropleth map and t-SNE.
- Exploratory data analysis script and notebook (`app_main.py` / `app_main.ipynb`).
- Initial REST service with FastAPI in `main_api.py`.
