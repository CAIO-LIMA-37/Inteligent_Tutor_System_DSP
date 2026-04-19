"""Multi-provider LLM Service."""

from typing import List, Dict, Optional, Any
import logging
import os

from app.config import settings

logger = logging.getLogger(__name__)

def get_provider_for_model(model: str) -> str:
    """Determine the provider for a given model name."""
    if not model:
        return "openai"        
    model_lower = model.lower()
    if "gpt" in model_lower or "davinci" in model_lower or "o1" in model_lower:
        return "openai"
    elif "claude" in model_lower:
        return "anthropic"
    elif "deepseek" in model_lower:
        return "deepseek"
    elif "llama" in model_lower or "granite" in model_lower or "dolphin" in model_lower or "mistral" in model_lower or "phi" in model_lower or "gemma" in model_lower or "qwen" in model_lower: return "ollama" # Roteamento para modelos locais
    else:
        return "openai"  # Fallback

class LLMService:
    """Multi-provider LLM inference service."""

    def __init__(self):
        self._openai_client = None
        self._anthropic_client = None

    @property
    def openai_client(self):
        """Lazy load OpenAI client."""
        if self._openai_client is None and settings.openai_api_key:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    @property
    def anthropic_client(self):
        """Lazy load Anthropic client."""
        if self._anthropic_client is None and settings.anthropic_api_key:
            from anthropic import AsyncAnthropic
            self._anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        return self._anthropic_client

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Generate a response using the specified model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (determines provider)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Dict with 'text', 'total_tokens', 'prompt_tokens', 'completion_tokens'
        """
        model = model or settings.default_model
        provider = get_provider_for_model(model)

        logger.info(f"Generating with model={model}, provider={provider}")

        # [INSERÇÃO PONTUAL]: Capturar o resultado em vez de retornar diretamente
        result = None
        if provider == "openai":
            result = await self._generate_openai(messages, model, temperature, max_tokens)
        elif provider == "anthropic":
            result = await self._generate_anthropic(messages, model, temperature, max_tokens)
        elif provider == "deepseek":
            result = await self._generate_deepseek(messages, model, temperature, max_tokens)
        elif provider == "ollama":
            result = await self._generate_ollama(messages, model, temperature, max_tokens) # ADICIONADO PARA USO LOCAL 
        else:
            result = await self._generate_openai(messages, model, temperature, max_tokens)

                    
        return result


    async def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate using OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")

        try:
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
            usage = response.usage
            return {
                "text": response.choices[0].message.content,
                "total_tokens": usage.total_tokens if usage else None,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
            }

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    async def _generate_anthropic(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate using Anthropic API."""
        if not self.anthropic_client:
            raise ValueError("Anthropic API key not configured")

        try:
            # Extract system message
            system_content = ""
            user_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    user_messages.append({"role": msg["role"], "content": msg["content"]})

            response = await self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_content,
                messages=user_messages,
                temperature=temperature
            )
            usage = response.usage
            return {
                "text": response.content[0].text,
                "total_tokens": (usage.input_tokens + usage.output_tokens) if usage else None,
                "prompt_tokens": usage.input_tokens if usage else None,
                "completion_tokens": usage.output_tokens if usage else None,
            }

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise

    async def _generate_deepseek(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate using DeepSeek API (OpenAI-compatible)."""
        if not settings.deepseek_api_key:
            raise ValueError("DeepSeek API key not configured")

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url="https://api.deepseek.com/v1"
            )

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            usage = response.usage
            return {
                "text": response.choices[0].message.content,
                "total_tokens": usage.total_tokens if usage else None,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
            }

        except Exception as e:
            logger.error(f"DeepSeek generation failed: {e}")
            raise
    
    async def _generate_deepseek(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate using DeepSeek API (OpenAI-compatible)."""
        if not settings.deepseek_api_key:
            raise ValueError("DeepSeek API key not configured")

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url="https://api.deepseek.com/v1"
            )

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            usage = response.usage
            return {
                "text": response.choices[0].message.content,
                "total_tokens": usage.total_tokens if usage else None,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
            }

        except Exception as e:
            logger.error(f"DeepSeek generation failed: {e}")
            raise
    
    async def _generate_ollama(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate using local Ollama instance (OpenAI-compatible API)."""
        # Busca a URL do .env ou usa o gateway padrão do Docker como fallback
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://172.17.0.1:11434/v1")
        
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                api_key="ollama-local", 
                base_url=ollama_url,
                timeout=3600.0,  # Aumenta a paciência interna para 1 hora
                max_retries=0    # Proíbe o reenvio automático da requisição
            )

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            usage = response.usage
            return {
                "text": response.choices[0].message.content,
                "total_tokens": usage.total_tokens if usage else 0,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
            }

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

