"""Runtime configuration for AnonLM."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class ProviderPreset(str, Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    CUSTOM = "custom"


_PROVIDER_DEFAULTS: dict[ProviderPreset, dict[str, str]] = {
    ProviderPreset.OPENAI: {
        "model_name": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
    },
    ProviderPreset.OPENROUTER: {
        "model_name": "openai/gpt-4o-mini",
        "base_url": "https://openrouter.ai/api/v1",
    },
    ProviderPreset.GROQ: {
        "model_name": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
    },
    ProviderPreset.CUSTOM: {
        "model_name": "gpt-4o-mini",
        "base_url": "",
    },
}


@dataclass(frozen=True)
class AnonLMConfig:
    provider: ProviderPreset = ProviderPreset.OPENAI
    model_name: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "ANONLM_API_KEY"
    api_key: str | None = None
    temperature: float = 0.0
    max_chunk_chars: int = 1000
    chunk_overlap_chars: int = 100
    retry_initial_interval: float = 0.5
    retry_backoff_factor: float = 2.0
    retry_max_interval: float = 8.0
    retry_max_attempts: int = 4

    @classmethod
    def from_env(cls) -> AnonLMConfig:
        provider_raw = os.getenv("ANONLM_PROVIDER", ProviderPreset.OPENAI.value).lower().strip()
        provider = ProviderPreset(provider_raw)
        provider_defaults = _PROVIDER_DEFAULTS[provider]

        model_name = os.getenv("ANONLM_MODEL_NAME", provider_defaults["model_name"])
        base_url = os.getenv("ANONLM_BASE_URL", provider_defaults["base_url"])
        api_key_env = os.getenv("ANONLM_API_KEY_ENV", "ANONLM_API_KEY")
        api_key = _empty_as_none(os.getenv("ANONLM_API_KEY"))

        temperature = _parse_float(os.getenv("ANONLM_TEMPERATURE"), 0.0)
        max_chunk_chars = _parse_int(os.getenv("ANONLM_MAX_CHUNK_CHARS"), 1000)
        chunk_overlap_chars = _parse_int(os.getenv("ANONLM_CHUNK_OVERLAP_CHARS"), 100)

        return cls(
            provider=provider,
            model_name=model_name,
            base_url=base_url,
            api_key_env=api_key_env,
            api_key=api_key,
            temperature=temperature,
            max_chunk_chars=max_chunk_chars,
            chunk_overlap_chars=chunk_overlap_chars,
        ).validate()

    def with_overrides(self, **kwargs: object) -> AnonLMConfig:
        updated = self

        provider_value = kwargs.pop("provider", None)
        if provider_value is not None:
            provider = ProviderPreset(str(provider_value))
            provider_defaults = _PROVIDER_DEFAULTS[provider]
            updated = replace(
                updated,
                provider=provider,
                model_name=provider_defaults["model_name"],
                base_url=provider_defaults["base_url"],
            )

        if "model_name" in kwargs and kwargs["model_name"] is not None:
            updated = replace(updated, model_name=str(kwargs["model_name"]))
        if "base_url" in kwargs and kwargs["base_url"] is not None:
            updated = replace(updated, base_url=str(kwargs["base_url"]))
        if "api_key_env" in kwargs and kwargs["api_key_env"] is not None:
            updated = replace(updated, api_key_env=str(kwargs["api_key_env"]))
        if "api_key" in kwargs and kwargs["api_key"] is not None:
            updated = replace(updated, api_key=str(kwargs["api_key"]))
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            updated = replace(updated, temperature=float(kwargs["temperature"]))
        if "max_chunk_chars" in kwargs and kwargs["max_chunk_chars"] is not None:
            updated = replace(updated, max_chunk_chars=int(kwargs["max_chunk_chars"]))
        if "chunk_overlap_chars" in kwargs and kwargs["chunk_overlap_chars"] is not None:
            updated = replace(updated, chunk_overlap_chars=int(kwargs["chunk_overlap_chars"]))

        return updated.validate()

    def validate(self) -> AnonLMConfig:
        if self.provider == ProviderPreset.CUSTOM and not self.base_url.strip():
            raise ValueError("ANONLM_BASE_URL is required when provider is 'custom'.")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty.")
        if self.max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be greater than 0.")
        if self.chunk_overlap_chars < 0:
            raise ValueError("chunk_overlap_chars must be 0 or greater.")
        if self.chunk_overlap_chars >= self.max_chunk_chars:
            raise ValueError("chunk_overlap_chars must be smaller than max_chunk_chars.")
        if self.retry_max_attempts < 1:
            raise ValueError("retry_max_attempts must be at least 1.")
        return self

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key

        env_name = self.api_key_env.strip() or "ANONLM_API_KEY"
        value = _empty_as_none(os.getenv(env_name))
        if value:
            return value

        fallback = _empty_as_none(os.getenv("ANONLM_API_KEY"))
        if fallback:
            return fallback

        # Backward compatibility fallback for users migrating from other setups.
        openai_fallback = _empty_as_none(os.getenv("OPENAI_API_KEY"))
        if openai_fallback:
            return openai_fallback

        raise ValueError(
            "Missing API key. Set ANONLM_API_KEY or set ANONLM_API_KEY_ENV to an existing env var."
        )

    def to_public_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "temperature": self.temperature,
            "max_chunk_chars": self.max_chunk_chars,
            "chunk_overlap_chars": self.chunk_overlap_chars,
            "retry_initial_interval": self.retry_initial_interval,
            "retry_backoff_factor": self.retry_backoff_factor,
            "retry_max_interval": self.retry_max_interval,
            "retry_max_attempts": self.retry_max_attempts,
        }


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    return float(value)


def _parse_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    return int(value)


def _empty_as_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None
