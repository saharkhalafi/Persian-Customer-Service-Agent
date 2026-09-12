from app.services.product_llm_config import load_product_llm_config
from app.services.product_name_matching import load_name_matching_enabled


def test_production_product_flags_keep_safe_defaults(monkeypatch):
    monkeypatch.delenv("LLM_METADATA_FALLBACK_ENABLED", raising=False)
    monkeypatch.delenv("LLM_CONFIDENCE_THRESHOLD", raising=False)
    monkeypatch.delenv("LLM_RERANK_ENABLED", raising=False)
    monkeypatch.delenv("LLM_QUERY_UNDERSTANDING_ENABLED", raising=False)
    monkeypatch.delenv("PRODUCT_NAME_MATCHING_ENABLED", raising=False)

    config = load_product_llm_config()
    assert config.metadata_fallback_enabled is True
    assert config.confidence_threshold == 0.7
    assert config.rerank_enabled is False
    assert config.query_understanding_enabled is False
    assert load_name_matching_enabled() is False
