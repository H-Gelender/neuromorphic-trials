"""Service LLM — génération de code via un modèle de langage local.

Branche pydantic-ai sur un endpoint OpenAI-compatible (llama-server / llama.cpp).
"""
from __future__ import annotations

import asyncio

import requests
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

__all__ = ["GeneratedCode", "LLMService", "server_healthy"]


def server_healthy(base_url: str = "http://localhost:8080/v1",
                   timeout: float = 3.0) -> bool:
    """Vérifie que le serveur llama répond sur son endpoint /health."""
    health = base_url.rsplit("/v1", 1)[0] + "/health"
    try:
        return requests.get(health, timeout=timeout).status_code == 200
    except Exception:
        return False


class GeneratedCode(BaseModel):
    """Sortie structurée du LLM local."""
    language: str = Field(description="Langage du code généré")
    description: str = Field(description="Brève explication de l'approche")
    code: str = Field(description="Le code Python à exécuter")


class LLMService:
    """Service de génération de code via un LLM local (endpoint OpenAI-compatible)."""

    def __init__(self, base_url: str = "http://localhost:8080/v1",
                 model: str = "qwen2.5-7b", temperature: float = 0.1,
                 system_prompt: str | None = None) -> None:
        self.base_url = base_url
        provider = OpenAIProvider(base_url=base_url, api_key="local")
        llm = OpenAIChatModel(model_name=model, provider=provider)
        self.agent = Agent(
            llm, output_type=GeneratedCode,
            system_prompt=system_prompt
            or "You are a concise Python coding assistant. Generate correct, minimal, type-hinted code.",
            model_settings={"temperature": temperature},
        )

    def healthy(self) -> bool:
        return server_healthy(self.base_url)

    async def generate_code(self, task: str, max_retries: int = 3) -> GeneratedCode:
        """Génère du code structuré. Async + retry.

        - On utilise `await agent.run()` et non `run_sync()` : ce dernier appelle
          `asyncio.run()` en interne et échoue avec `RuntimeError: This event loop
          is already running` dès qu'un event loop tourne (cas d'un notebook).
        - Retry : un modèle 7B échoue parfois à remplir tous les champs de la
          sortie structurée (le champ `code` peut revenir vide).
        """
        if not self.healthy():
            raise RuntimeError(
                "Serveur local injoignable. Lance llama-server "
                "(voir README / notebook).")
        for attempt in range(max_retries):
            result = await self.agent.run(task)
            out = result.output
            if out.code.strip():
                return out
            print(f"  ⚠ code vide, retry {attempt + 1}/{max_retries}")
        raise RuntimeError(
            f"Le LLM n'a pas produit de code valide après {max_retries} essais.")
