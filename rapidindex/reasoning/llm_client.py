# rapidindex/reasoning/llm_client.py
"""Multi-provider LLM client supporting Anthropic and Ollama."""

import asyncio
import os
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from loguru import logger


try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider (paid)."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed")
        if not api_key:
            raise ValueError("Anthropic API key required")

        self.api_key = api_key
        self.model = model
        # FIX: use AsyncAnthropic so generate() is a true coroutine
        # and doesn't block the event loop.
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        logger.info(f"Anthropic provider initialized: {model}")

    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate using Claude (non-blocking)."""
        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as exc:
            logger.error(f"Anthropic API error: {exc}")
            raise

    def is_available(self) -> bool:
        return ANTHROPIC_AVAILABLE and bool(self.api_key)


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider (free)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
    ):
        if not OLLAMA_AVAILABLE:
            raise ImportError("requests package not installed")

        self.base_url = base_url
        self.model = model

        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Ollama provider initialized: {model} at {base_url}")
        except Exception as exc:
            logger.warning(f"Ollama connection failed: {exc}")
            logger.warning("Make sure Ollama is running: ollama serve")

    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate using Ollama (non-blocking via thread pool).

        FIX: previously called requests.post() directly inside an async
        method, which blocked the entire event loop for the full LLM
        inference time (potentially minutes).  We now offload the
        synchronous HTTP call to a thread so the loop stays free.
        """
        return await asyncio.to_thread(self._generate_sync, prompt, max_tokens)

    # ------------------------------------------------------------------
    # Sync helper (runs in thread pool — must not touch async primitives)
    # ------------------------------------------------------------------

    def _generate_sync(self, prompt: str, max_tokens: int = 1000) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.Timeout:
            logger.error("Ollama request timeout")
            raise
        except Exception as exc:
            logger.error(f"Ollama API error: {exc}")
            raise

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False


class LLMClient:
    """Multi-provider LLM client."""

    def __init__(
        self,
        provider: str = "anthropic",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
    ):
        self.provider_name = provider.lower()

        if self.provider_name == "anthropic":
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            model = model or "claude-sonnet-4-20250514"
            if not api_key:
                logger.warning("No Anthropic API key — will use fallback")
                self.provider = None
            else:
                self.provider = AnthropicProvider(api_key=api_key, model=model)

        elif self.provider_name == "ollama":
            model = model or "llama3.2"
            self.provider = OllamaProvider(base_url=ollama_url, model=model)

        else:
            raise ValueError(f"Unknown provider: {provider}")

        if self.provider and self.provider.is_available():
            logger.info(f"LLMClient initialized with {self.provider_name}")
        else:
            logger.warning(f"Provider {self.provider_name} not available — using fallback")
            self.provider = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def reason_over_sections(
        self,
        query: str,
        sections: List[Dict],
        max_tokens: int = 1000,
    ) -> Dict:
        """Use LLM to select the most relevant sections."""
        if not self.provider:
            return {
                "selected_sections": [s["id"] for s in sections[:2]],
                "reasoning": "LLM not available, using fallback",
                "confidence": 0.5,
            }

        prompt = self._build_reasoning_prompt(query, sections)

        try:
            response_text = await self.provider.generate(prompt, max_tokens)
            return self._parse_reasoning_response(response_text, sections)
        except Exception as exc:
            logger.error(f"LLM reasoning error: {exc}")
            return {
                "selected_sections": [s["id"] for s in sections[:2]],
                "reasoning": f"LLM call failed: {exc}",
                "confidence": 0.5,
            }

    async def generate_answer(
        self,
        query: str,
        relevant_sections: List[Dict],
        max_tokens: int = 2000,
    ) -> str:
        """Generate final answer from selected sections."""
        if not self.provider:
            return "\n\n".join(
                f"{s['title']}: {s['content'][:200]}..." for s in relevant_sections
            )

        context = "\n\n".join(
            f"From {s['title']} (Pages {s.get('pages', [])}):\n{s['content']}"
            for s in relevant_sections
        )

        prompt = (
            f"Answer the following question using only the provided context.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            f"Provide a clear, concise answer. "
            f"If the context doesn't contain enough information, say so."
        )

        try:
            return await self.provider.generate(prompt, max_tokens)
        except Exception as exc:
            logger.error(f"Answer generation error: {exc}")
            return f"Error generating answer: {exc}"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_reasoning_prompt(self, query: str, sections: List[Dict]) -> str:
        sections_text = "\n\n".join(
            f"Section {i + 1} (ID: {s['id']}):\n"
            f"Title: {s['title']}\n"
            f"Pages: {s.get('pages', [])}\n"
            f"Preview: {s.get('preview', '')[:200]}..."
            for i, s in enumerate(sections)
        )

        return (
            f"You are a document analysis assistant. Given a user query and "
            f"candidate document sections, determine which sections are most "
            f"likely to contain the answer.\n\n"
            f"Query: {query}\n\n"
            f"Candidate Sections:\n{sections_text}\n\n"
            f"Analyze each section and select the 1-3 most relevant ones. "
            f"Respond in this format:\n\n"
            f"SELECTED SECTIONS: [list section IDs, e.g., sec_0001, sec_0005]\n"
            f"REASONING: [brief explanation]\n"
            f"CONFIDENCE: [high/medium/low]\n\n"
            f"Think step by step about which sections best answer the query."
        )

    def _parse_reasoning_response(self, response: str, sections: List[Dict]) -> Dict:
        import re

        selected_match = re.search(r"SELECTED SECTIONS:\s*\[(.*?)\]", response)
        selected_ids = []
        if selected_match:
            selected_ids = [s.strip() for s in selected_match.group(1).split(",")]

        if not selected_ids:
            found = re.findall(r"sec_\d{4}", response)
            selected_ids = list(dict.fromkeys(found))[:3]  # dedup, preserve order

        if not selected_ids:
            selected_ids = [s["id"] for s in sections[:2]]

        reasoning_match = re.search(
            r"REASONING:\s*(.*?)(?:CONFIDENCE:|$)", response, re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response[:200]

        confidence_match = re.search(
            r"CONFIDENCE:\s*(high|medium|low)", response, re.IGNORECASE
        )
        confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
        confidence = confidence_map.get(
            confidence_match.group(1).lower() if confidence_match else "medium", 0.7
        )

        return {
            "selected_sections": selected_ids,
            "reasoning": reasoning,
            "confidence": confidence,
            "full_response": response,
        }


__all__ = ["LLMClient", "AnthropicProvider", "OllamaProvider"]
