from __future__ import annotations

import pytest

from anonlm.config import AnonLMConfig, ProviderPreset


def test_from_env_uses_provider_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANONLM_PROVIDER", "openrouter")
    monkeypatch.delenv("ANONLM_MODEL_NAME", raising=False)
    monkeypatch.delenv("ANONLM_BASE_URL", raising=False)

    config = AnonLMConfig.from_env()
    assert config.provider == ProviderPreset.OPENROUTER
    assert config.base_url == "https://openrouter.ai/api/v1"


def test_explicit_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANONLM_PROVIDER", "openai")
    config = AnonLMConfig.from_env().with_overrides(model_name="openai/gpt-oss-20b")
    assert config.model_name == "openai/gpt-oss-20b"


def test_resolve_api_key_from_custom_env_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_SECRET_KEY", "abc123")
    config = AnonLMConfig(api_key_env="MY_SECRET_KEY")
    assert config.resolved_api_key() == "abc123"


def test_custom_provider_requires_base_url() -> None:
    with pytest.raises(ValueError):
        AnonLMConfig(provider=ProviderPreset.CUSTOM, base_url="").validate()


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        AnonLMConfig(max_chunk_chars=100, chunk_overlap_chars=100).validate()
