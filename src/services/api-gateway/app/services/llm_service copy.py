"""Multi-provider LLM Service."""

from typing import List, Dict, Optional, Any, AsyncGenerator
import logging
import os

from app.config import settings

logger = logging.getLogger(__name__)

def get_provider_for_model(model: str) -> str:
    """Determina o provedor de inferência com base no nome do modelo."""
    if not model:
        return "openai"        
    
    model_lower = model.lower()
    if "gpt" in model_lower or "o1" in model_lower:
        return "openai"
    elif "claude" in model_lower:
        return "anthropic"
    elif "deepseek" in model_lower:
        return "deepseek"
    # O roteamento reconhecerá o Ministral e outros modelos locais direcionando ao vLLM
    elif "ministral" in model_lower or "llama" in model_lower or "qwen" in model_lower or "phi" in model_lower or "gemma" in model_lower: 
        return "vllm"
    else:
        return "openai"  # Fallback de segurança

class LLMService:
    """Serviço de inferência LLM multi-provedor com conexão persistente."""

    def __init__(self):
        self._openai_client = None
        self._anthropic_client = None
        self._deepseek_client = None
        self._vllm_client = None

    @property
    def openai_client(self):
        if self._openai_client is None and settings.openai_api_key:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    @property
    def anthropic_client(self):
        if self._anthropic_client is None and settings.anthropic_api_key:
            from anthropic import AsyncAnthropic
            self._anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        return self._anthropic_client
    
    @property
    def deepseek_client(self):
        if self._deepseek_client is None and getattr(settings, "deepseek_api_key", None):
            from openai import AsyncOpenAI
            self._deepseek_client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url="https://api.deepseek.com/v1"
            )
        return self._deepseek_client

    @property
    def vllm_client(self):
        """Inicializa o cliente compatível com OpenAI apontando para o servidor vLLM local."""
        vllm_url = getattr(settings, "vllm_base_url", os.getenv("VLLM_BASE_URL", "http://10.10.80.238:8006/v1"))
        
        if self._vllm_client is None:
            from openai import AsyncOpenAI
            self._vllm_client = AsyncOpenAI(
                api_key="vllm-local-dummy-key", # O vLLM exige uma string qualquer aqui
                base_url=vllm_url,
                timeout=3600.0,
                max_retries=0
            )
        return self._vllm_client

    # GERAÇÃO SÍNCRONA (MODO CLÁSSICO PARA O FRONTEND)
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """Orquestra a geração da resposta baseada no modelo solicitado."""
        model = model or getattr(settings, "default_model", "cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-8bit")
        provider = get_provider_for_model(model)

        logger.info(f"Gerando inferência | provider={provider} | model={model}")

        if provider == "openai":
            return await self._generate_openai(messages, model, temperature, max_tokens)
        elif provider == "anthropic":
            return await self._generate_anthropic(messages, model, temperature, max_tokens)
        elif provider == "deepseek":
            return await self._generate_deepseek(messages, model, temperature, max_tokens)
        elif provider == "vllm":
            return await self._generate_vllm(messages, model, temperature, max_tokens)
        else:
            logger.warning(f"Provedor '{provider}' não reconhecido. Usando fallback para OpenAI.")
            return await self._generate_openai(messages, model, temperature, max_tokens)

    async def _generate_openai(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        if not self.openai_client:
            raise ValueError("OpenAI API key não configurada.")
        
        kwargs = {"model": model, "messages": messages, "max_completion_tokens": max_tokens}
        model_lower = model.lower()
        if "o1" not in model_lower and "o3" not in model_lower and "gpt-5" not in model_lower:
            kwargs["temperature"] = temperature

        try:
            response = await self.openai_client.chat.completions.create(**kwargs)
            return self._format_openai_response(response)
        except Exception as e:
            logger.error(f"Falha na geração OpenAI: {e}")
            raise

    async def _generate_anthropic(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        if not self.anthropic_client:
            raise ValueError("Anthropic API key não configurada.")
        try:
            system_content = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
            user_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages if msg["role"] != "system"]

            response = await self.anthropic_client.messages.create(
                model=model, max_tokens=max_tokens, system=system_content, messages=user_messages, temperature=temperature
            )
            usage = response.usage
            return {
                "text": response.content[0].text,
                "total_tokens": (usage.input_tokens + usage.output_tokens) if usage else None,
                "prompt_tokens": usage.input_tokens if usage else None,
                "completion_tokens": usage.output_tokens if usage else None,
            }
        except Exception as e:
            logger.error(f"Falha na geração Anthropic: {e}")
            raise

    async def _generate_deepseek(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        if not self.deepseek_client:
            raise ValueError("DeepSeek API key não configurada.")
        try:
            response = await self.deepseek_client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            return self._format_openai_response(response)
        except Exception as e:
            logger.error(f"Falha na geração DeepSeek: {e}")
            raise

    async def _generate_vllm(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        if not self.vllm_client:
            raise ValueError("Cliente vLLM não pôde ser inicializado. Verifique a URL.")
        try:
            response = await self.vllm_client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            return self._format_openai_response(response)
        except Exception as e:
            logger.error(f"Falha na geração vLLM: {e}")
            raise

    def _format_openai_response(self, response: Any) -> Dict[str, Any]:
        usage = response.usage
        return {
            "text": response.choices[0].message.content,
            "total_tokens": usage.total_tokens if usage else 0,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
        }

     # GERAÇÃO EM STREAMING (MODO BENCHMARK PARA CAPTURA DE TTFT E ITL)
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> AsyncGenerator[str, None]:
        """
        Orquestra a geração da resposta em modo streaming (SSE) baseada no provedor.
        Retorna um iterador assíncrono que emite pedaços de texto.
        """
        model = model or getattr(settings, "default_model", "cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-8bit")
        provider = get_provider_for_model(model)

        logger.info(f"Gerando inferência (STREAM) | provider={provider} | model={model}")

        if provider == "openai":
            async for chunk in self._generate_openai_stream(messages, model, temperature, max_tokens):
                yield chunk
        elif provider == "deepseek":
            async for chunk in self._generate_deepseek_stream(messages, model, temperature, max_tokens):
                yield chunk
        elif provider == "vllm":
            async for chunk in self._generate_vllm_stream(messages, model, temperature, max_tokens):
                yield chunk
        elif provider == "anthropic":
            async for chunk in self._generate_anthropic_stream(messages, model, temperature, max_tokens):
                yield chunk
        else:
            logger.warning(f"Provedor '{provider}' não suporta stream ou não reconhecido. Fallback OpenAI.")
            async for chunk in self._generate_openai_stream(messages, model, temperature, max_tokens):
                yield chunk

    async def _generate_openai_stream(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> AsyncGenerator[str, None]:
        if not self.openai_client: raise ValueError("OpenAI API key não configurada.")
        kwargs = {"model": model, "messages": messages, "max_completion_tokens": max_tokens, "stream": True}
        if "o1" not in model.lower() and "o3" not in model.lower() and "gpt-5" not in model.lower():
            kwargs["temperature"] = temperature
        try:
            response_stream = await self.openai_client.chat.completions.create(**kwargs)
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Falha no streaming OpenAI: {e}")
            yield f" Erro: {str(e)}"

    async def _generate_deepseek_stream(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> AsyncGenerator[str, None]:
        if not self.deepseek_client: raise ValueError("DeepSeek API key não configurada.")
        try:
            response_stream = await self.deepseek_client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True
            )
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Falha no streaming DeepSeek: {e}")
            yield f" Erro: {str(e)}"

    async def _generate_vllm_stream(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> AsyncGenerator[str, None]:
        if not self.vllm_client: raise ValueError("Cliente vLLM não inicializado.")
        try:
            response_stream = await self.vllm_client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True
            )
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Falha no streaming vLLM: {e}")
            yield f" Erro: {str(e)}"

    async def _generate_anthropic_stream(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> AsyncGenerator[str, None]:
        if not self.anthropic_client: raise ValueError("Anthropic API key não configurada.")
        try:
            system_content = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
            user_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages if msg["role"] != "system"]
            async with self.anthropic_client.messages.stream(
                model=model, max_tokens=max_tokens, system=system_content, messages=user_messages, temperature=temperature
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Falha no streaming Anthropic: {e}")
            yield f" Erro: {str(e)}"