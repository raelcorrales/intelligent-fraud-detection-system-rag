# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/)
y este proyecto se adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-10-31

### Version Freeze — First Usable Release
Primera versión oficial consolidada, congelada y lista para su uso básico end-to-end en entornos locales o de demostración técnica.

### Añadido
- **Vector Store & Búsqueda Coseno**: Búsqueda vectorial real ($\text{cosine similarity}$) en `RAGRetriever` sobre embeddings normalizados ($L_2$) de 768 dimensiones.
- **Validación Estricta de API**: Contratos de datos tipados con Pydantic V2 (`TransactionSchema` y `FraudAnalysisResponse`) con alias para campos con espacios (`IP Address`).
- **Endpoint de Monitoreo `/health`**: Health check en FastAPI para verificar el estado de conexión con Ollama y el número de reglas cacheadas en memoria.
- **Suite de Pruebas Automatizadas**: 7 pruebas unitarias y de integración en `tests/test_pipeline.py` con cobertura de tipado, vectorización, cosine similarity, detección y API REST.
- **Modo Demostración por CLI**: Bloque de ejecución directa en `src/process_function.py` invocable con `python -m src.process_function`.
- **Módulo de Versión Centralizado**: Creación de `src/__version__.py`.

### Modificado
- `AnomalyDetector`: Reemplazado el bypass fijo por una arquitectura de decisión combinada (Inferencia con Ollama LLM + Heurísticas calibradas con datos de Kaggle).
- `load_and_embed_rules`: Generación y caching real de representaciones vectoriales para el catálogo `fraud_rules.json`.
- `LLMExplainer`: Explicaciones técnicas estructuradas y prompt optimizado para evitar alucinaciones.
- `main_api.py`: Eliminadas rutas relativas frágiles en favor de resolución robusta con `pathlib.Path` y `src.config`.
- `README.md`: Diagramas de flujo en Mermaid, guías de instalación rápida y especificaciones de endpoints.

---

## [0.1.0-beta] - 2026-10-31

### Añadido
- Módulo de configuración unificada `src/config.py`.
- Lógica de fallback determinista para vectorización y síntesis de explicaciones en entornos offline.
- Caching de singletons en `process_function.py` para optimizar latencia en llamadas recurrentes.

### Corregido
- Typo en llamada de logging en `AnomalyDetector` (`self.logger.errot`).
- Warnings de deprecación de Pydantic V2 (`example` reemplazado por `examples`).
- Discordancia en nombres de variables de entorno para modelos Ollama.

---

## [0.1.0-alpha] - 2026-10-31

### Añadido
- Estructura base del proyecto y modelado de datos con la clase `Transaction`.
- Catálogo inicial de 9 reglas de fraude bancario en `data/fraud_rules.json`.
- Capa de abstracción `OllamaBase` con reintentos exponenciales y jitter.
- Dashboard analítico en Streamlit (`src/app_fraude.py`) con mapas coropléticos de EE.UU. y t-SNE.
- Notebook y script de análisis exploratorio (`app_main.py` / `app_main.ipynb`).
- API básica inicial con FastAPI en `main_api.py`.
