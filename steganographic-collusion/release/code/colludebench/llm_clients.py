"""LLM client abstractions for ColludeBench agents.

Hosted models are reached through OpenRouter only (`backend: openrouter`); the direct
OpenAI, Gemini and Groq adapters were removed. `transformers` (local weights) and
`template` (no model) remain. The key is read from this project's own `.env`.
"""

from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import re
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
import time

# This project's own .env, by explicit path so a parent directory's .env never leaks in.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # pragma: no cover - optional convenience
    pass


class LLMClient(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Return a completion for the given prompt."""


class OpenAIChatClient(LLMClient):
    """OpenAI-compatible Chat Completions client, pointed at OpenRouter by default."""

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
    ) -> None:
        # Route hosted traffic through the repo-root shared LLM config (OpenRouter/Nebius).
        from poseidon_compute import make_client, resolve_model

        # make_client() already picks base_url+api_key from the shared config; honor any
        # explicit overrides passed by the caller (e.g. per-config `api_key`/`base_url`).
        overrides = {}
        if api_key:
            overrides["api_key"] = api_key
        if base_url:
            overrides["base_url"] = base_url
        self._client = make_client(**overrides)
        self.model = resolve_model(model)
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> str:
        messages = []
        if extra:
            system_prompt = extra.get("system_prompt")
            if system_prompt:
                messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            stop=stop,
        )
        return response.choices[0].message.content.strip()


class TransformersClient(LLMClient):
    """Wrapper around Hugging Face transformers pipelines."""

    def __init__(
        self,
        model: str,
        *,
        device: Optional[str | int] = None,
        max_new_tokens: int = 128,
        default_temperature: Optional[float] = None,
        pipeline_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - missing optional dependency
            raise RuntimeError(
                "TransformersClient requires the `transformers` package. "
                "Install it with `pip install transformers accelerate`."
            ) from exc

        kwargs = dict(pipeline_kwargs or {})
        task = kwargs.pop("task", "text-generation")
        self._generator = pipeline(task, model=model, device=device, **kwargs)
        tokenizer = getattr(self._generator, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token  # type: ignore[attr-defined]

        self.max_new_tokens = max_new_tokens
        self.default_temperature = default_temperature

    def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> str:
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens if max_tokens is not None else self.max_new_tokens,
        }
        if temperature is not None or self.default_temperature is not None:
            generation_kwargs["temperature"] = temperature if temperature is not None else self.default_temperature
        outputs = self._generator(prompt, **generation_kwargs)
        text = outputs[0]["generated_text"]
        if stop:
            for token in stop:
                if token in text:
                    text = text.split(token)[0]
                    break
        return text.strip()


class TemplateClient(LLMClient):
    """Deterministic template-based client for offline tests."""

    def __init__(self, template: str) -> None:
        self.template = template

    def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> str:
        variables = extra.get("variables", {}) if extra else {}
        return self.template.format_map(_SafeDefaultDict(variables)).strip()


class RateLimitedClient(LLMClient):
    """Wrapper that enforces a minimum interval between API calls."""

    def __init__(self, base: LLMClient, min_interval: float) -> None:
        self._base = base
        self._min_interval = max(min_interval, 0.0)
        self._lock = Lock()
        self._last_request = 0.0

    def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> str:
        with self._lock:
            if self._min_interval > 0:
                now = time.time()
                wait = self._min_interval - (now - self._last_request)
                if wait > 0:
                    time.sleep(wait)
            result = self._base.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                extra=extra,
            )
            self._last_request = time.time()
            return result


def create_llm_client(config: Mapping[str, Any]) -> LLMClient:
    """Instantiate an LLM client from configuration."""

    backend = config.get("backend")
    if not backend:
        raise ValueError("LLM configuration must include a 'backend' field.")

    backend = backend.lower()
    # Only the `openrouter` backend is supported for hosted models. A closed/hosted backend label
    # (openai/gemini/groq) must fail loudly rather than be silently routed to a different endpoint
    # (COMPUTE.md §7 "No silent substitution"); tell the caller the sanctioned way to run it.
    if backend in {"openai", "openai_chat", "gemini", "google", "google_ai", "groq"}:
        raise ValueError(
            f"LLM backend '{backend}' is not supported: the direct OpenAI / Gemini / Groq "
            "adapters are not available here. Hosted models (including closed ones like GPT/Gemini) run via "
            "OpenRouter through the shared poseidon_compute client. Set `backend: openrouter` and "
            "give `model` in Nebius / Hugging Face casing; the entry point routes it to the right "
            "provider (and fails loudly if none serves it). See COMPUTE.md §§2-3,7."
        )
    if backend == "openrouter":
        # Hosted traffic is routed centrally via the repo-root llm_config (make_client picks
        # base_url + key from compute.env / NEBIUS_API_KEY). Per-config api_key/base_url
        # overrides are still honored when present.
        client = OpenAIChatClient(
            model=config["model"],
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            default_temperature=config.get("temperature"),
            default_max_tokens=config.get("max_tokens"),
        )
        min_interval = float(config.get("min_request_interval", 0) or 0)
        if min_interval > 0:
            return RateLimitedClient(client, min_interval)
        return client
    if backend in {"hf", "transformers", "huggingface"}:
        return TransformersClient(
            model=config["model"],
            device=config.get("device"),
            max_new_tokens=config.get("max_tokens", 128),
            default_temperature=config.get("temperature"),
            pipeline_kwargs=config.get("pipeline_kwargs"),
        )
    if backend in {"template", "static"}:
        template = config.get("template")
        if not template:
            raise ValueError("Template backend requires a 'template' field.")
        return TemplateClient(template)

    raise ValueError(f"Unsupported LLM backend '{backend}'.")


class _SafeDefaultDict(dict):
    """dict subclass that returns empty string for missing keys during formatting."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - simple default
        return ""
