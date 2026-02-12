# rapidindex/reasoning/llm_client.py
"""Multi-provider LLM client supporting Anthropic and Ollama."""

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
        """Initialize Anthropic provider."""
        self.api_key = api_key
        self.model = model
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed")
        
        if not api_key:
            raise ValueError("Anthropic API key required")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Anthropic provider initialized: {model}")
    
    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate using Claude."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        return ANTHROPIC_AVAILABLE and bool(self.api_key)


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider (free)."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2"
    ):
        """
        Initialize Ollama provider.
        
        Args:
            base_url: Ollama API URL
            model: Model name (llama3.2, mistral, etc.)
        """
        self.base_url = base_url
        self.model = model
        
        if not OLLAMA_AVAILABLE:
            raise ImportError("requests package not installed")
        
        # Test connection
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Ollama provider initialized: {model} at {base_url}")
        except Exception as e:
            logger.warning(f"Ollama connection failed: {e}")
            logger.warning("Make sure Ollama is running: ollama serve")
    
    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate using Ollama."""
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
                    }
                },
                timeout=120  # 2 minutes timeout for long responses
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '')
            
        except requests.exceptions.Timeout:
            logger.error("Ollama request timeout")
            raise
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False


class LLMClient:
    """Multi-provider LLM client."""
    
    def __init__(
        self,
        provider: str = "anthropic",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        ollama_url: str = "http://localhost:11434"
    ):
        """
        Initialize LLM client with specified provider.
        
        Args:
            provider: 'anthropic' or 'ollama'
            api_key: API key for paid providers
            model: Model name
            ollama_url: Ollama server URL
        """
        self.provider_name = provider.lower()
        
        # Initialize appropriate provider
        if self.provider_name == "anthropic":
            api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            model = model or "claude-sonnet-4-20250514"
            
            if not api_key:
                logger.warning("No Anthropic API key - will use fallback")
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
            logger.warning(f"Provider {self.provider_name} not available - using fallback")
            self.provider = None
    
    async def reason_over_sections(
        self, 
        query: str, 
        sections: List[Dict],
        max_tokens: int = 1000
    ) -> Dict:
        """Use LLM to reason about which sections are most relevant."""
        
        if not self.provider:
            # Fallback: return first 2 sections
            return {
                'selected_sections': [s['id'] for s in sections[:2]],
                'reasoning': 'LLM not available, using fallback',
                'confidence': 0.5
            }
        
        # Build prompt
        prompt = self._build_reasoning_prompt(query, sections)
        
        # Generate response
        try:
            response_text = await self.provider.generate(prompt, max_tokens)
            
            # Parse response
            return self._parse_reasoning_response(response_text, sections)
            
        except Exception as e:
            logger.error(f"LLM reasoning error: {e}")
            return {
                'selected_sections': [s['id'] for s in sections[:2]],
                'reasoning': f'LLM call failed: {str(e)}',
                'confidence': 0.5
            }
    
    async def generate_answer(
        self,
        query: str,
        relevant_sections: List[Dict],
        max_tokens: int = 2000
    ) -> str:
        """Generate final answer from selected sections."""
        
        if not self.provider:
            # Fallback: concatenate sections
            return "\n\n".join([
                f"{s['title']}: {s['content'][:200]}..."
                for s in relevant_sections
            ])
        
        context = "\n\n".join([
            f"From {s['title']} (Pages {s.get('pages', [])}):\n{s['content']}"
            for s in relevant_sections
        ])
        
        prompt = f"""Answer the following question using only the provided context.

Question: {query}

Context:
{context}

Provide a clear, concise answer. If the context doesn't contain enough information, say so."""
        
        try:
            return await self.provider.generate(prompt, max_tokens)
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return f"Error generating answer: {str(e)}"
    
    def _build_reasoning_prompt(self, query: str, sections: List[Dict]) -> str:
        """Build prompt for section selection."""
        sections_text = "\n\n".join([
            f"Section {i+1} (ID: {s['id']}):\n"
            f"Title: {s['title']}\n"
            f"Pages: {s.get('pages', [])}\n"
            f"Preview: {s.get('preview', '')[:200]}..."
            for i, s in enumerate(sections)
        ])
        
        return f"""You are a document analysis assistant. Given a user query and candidate document sections, determine which sections are most likely to contain the answer.

Query: {query}

Candidate Sections:
{sections_text}

Analyze each section and select the 1-3 most relevant ones. Provide your response in this format:

SELECTED SECTIONS: [list section IDs, e.g., sec_0001, sec_0005]
REASONING: [brief explanation of why these sections are relevant]
CONFIDENCE: [high/medium/low]

Think step by step about which sections would best answer this query."""
    
    def _parse_reasoning_response(self, response: str, sections: List[Dict]) -> Dict:
        """Parse LLM's structured response."""
        import re
        
        # Extract selected sections
        selected_match = re.search(r'SELECTED SECTIONS:\s*\[(.*?)\]', response)
        selected_ids = []
        if selected_match:
            ids_text = selected_match.group(1)
            selected_ids = [s.strip() for s in ids_text.split(',')]
        
        # If parsing failed, try to find section IDs in response
        if not selected_ids:
            # Look for sec_XXXX patterns
            section_pattern = r'sec_\d{4}'
            found_ids = re.findall(section_pattern, response)
            if found_ids:
                selected_ids = list(set(found_ids))[:3]  # Max 3
        
        # Still no IDs? Use first 2 sections
        if not selected_ids:
            selected_ids = [s['id'] for s in sections[:2]]
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.*?)(?:CONFIDENCE:|$)', response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response[:200]
        
        # Extract confidence
        confidence_match = re.search(r'CONFIDENCE:\s*(high|medium|low)', response, re.IGNORECASE)
        confidence_map = {'high': 0.9, 'medium': 0.7, 'low': 0.5}
        confidence = confidence_map.get(
            confidence_match.group(1).lower() if confidence_match else 'medium',
            0.7
        )
        
        return {
            'selected_sections': selected_ids,
            'reasoning': reasoning,
            'confidence': confidence,
            'full_response': response
        }


__all__ = ['LLMClient', 'AnthropicProvider', 'OllamaProvider']