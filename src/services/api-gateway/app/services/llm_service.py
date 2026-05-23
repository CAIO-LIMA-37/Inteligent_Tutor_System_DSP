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
        temperature: Optional[float] = None,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """Orquestra a geração da resposta baseada no modelo solicitado."""
        
        # 1. Definição do Modelo
        model_name = model or getattr(settings, "default_model", "cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-8bit")
        provider = get_provider_for_model(model_name)

        # 2. Definição da Temperatura via .env
        if temperature is None:
            final_temperature = getattr(settings, "llm_temperature", 0.0)
        else:
            final_temperature = temperature
        
        # 3. Segurança para modelos de raciocínio (o1 e o3)
        model_lower = model_name.lower()
        if "o1" in model_lower or "o3" in model_lower or "gpt-5" in model_lower:
            final_temperature = None
            logger.debug(f"Modelo {model_name} detectado. Parâmetro 'temperature' suprimido")
        
        # 4. Captura do Seed Global
        llm_seed = getattr(settings, "llm_seed", 42)
        
        logger.info(f"Gerando inferência | provider={provider} | model={model}")

        if provider == "openai":
            return await self._generate_openai(
                messages=messages,
                model=model_name,
                temperature=final_temperature,
                seed=llm_seed,
                max_tokens=max_tokens
            )
        elif provider == "anthropic":
            return await self._generate_anthropic(
                messages=messages,
                model=model_name,
                temperature=final_temperature,
                seed=llm_seed,
                max_tokens=max_tokens
            )
        elif provider == "deepseek":
            return await self._generate_deepseek(
                messages=messages,
                model=model_name,
                temperature=final_temperature,
                seed=llm_seed,
                max_tokens=max_tokens
            )
        elif provider == "vllm":
            return await self._generate_vllm(
                messages=messages,
                model=model_name,
                temperature=final_temperature,
                seed=llm_seed,
                max_tokens=max_tokens
            )
        else:
            logger.warning(f"Provedor '{provider}' não reconhecido. Usando fallback para OpenAI.")
            return await self._generate_openai(
                messages=messages,
                model=model_name,
                temperature=final_temperature,
                seed=llm_seed,
                max_tokens=max_tokens
            )

    async def _generate_openai(self, messages: List[Dict[str, str]], model: str, temperature: Optional[float], seed: int, max_tokens: int) -> Dict[str, Any]:
        if not self.openai_client:
            raise ValueError("OpenAI API key não configurada.")
        
        kwargs = {"model": model, "messages": messages, "seed": seed}

        # Correção de tokens
        model_lower = model.lower()
        if any(m in model_lower for m in ["o1", "o3", "gpt-5"]):
            kwargs["max_completion_tokens"] = 30000
        else:
            kwargs["max_completion_tokens"] = max_tokens
        
        # Como a supressão para o1/o3 já foi feita no 'generate' (passando None), basta checar:
        if temperature is not None:
            kwargs["temperature"] = temperature
            if temperature == 0.0:
                kwargs["top_p"] = 0.0001 # Trava matemática de determinismo

        try:
            response = await self.openai_client.chat.completions.create(**kwargs)
            return self._format_openai_response(response)
        except Exception as e:
            logger.error(f"Falha na geração OpenAI: {e}")
            raise

    async def _generate_anthropic(self, messages: List[Dict[str, str]], model: str, temperature: Optional[float], seed: int, max_tokens: int) -> Dict[str, Any]:
        if not self.anthropic_client:
            raise ValueError("Anthropic API key não configurada.")
        try:
            system_content = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
            user_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages if msg["role"] != "system"]

            kwargs = {"model": model, "max_tokens": max_tokens, "system": system_content, "messages": user_messages}
            if temperature is not None:
                kwargs["temperature"] = temperature

            response = await self.anthropic_client.messages.create(**kwargs)
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

    async def _generate_deepseek(self, messages: List[Dict[str, str]], model: str, temperature: Optional[float], seed: int, max_tokens: int) -> Dict[str, Any]:
        if not self.deepseek_client:
            raise ValueError("DeepSeek API key não configurada.")
        
        kwargs = {"model": model, "messages": messages, "seed": seed}

        # Correção de tokens específica para o DeepSeek
        model_lower = model.lower()
        if "reasoner" in model_lower:
            kwargs["max_tokens"] = 8192
        else:
            kwargs["max_tokens"] = max_tokens

        if temperature is not None:
            kwargs["temperature"] = temperature
            if temperature == 0.0: kwargs["top_p"] = 0.0001

        try:
            response = await self.deepseek_client.chat.completions.create(**kwargs)
            return self._format_openai_response(response)
        except Exception as e:
            logger.error(f"Falha na geração DeepSeek: {e}")
            raise

    async def _generate_vllm(self, messages: List[Dict[str, str]], model: str, temperature: Optional[float], seed: int, max_tokens: int) -> Dict[str, Any]:
        if not self.vllm_client:
            raise ValueError("Cliente vLLM não pôde ser inicializado. Verifique a URL.")
        
        kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens, "seed": seed}
        if temperature is not None:
            kwargs["temperature"] = temperature
            if temperature == 0.0: kwargs["top_p"] = 0.0001

        try:
            response = await self.vllm_client.chat.completions.create(**kwargs)
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
        temperature: Optional[float] = None,
        max_tokens: int = 4096
    ) -> AsyncGenerator[str, None]:
        
        model_name = model or getattr(settings, "default_model", "cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-8bit")
        provider = get_provider_for_model(model_name)

        if temperature is None:
            final_temp = getattr(settings, "llm_temperature", 0.0)
        else:
            final_temp = temperature
            
        model_lower = model_name.lower()
        if "o1" in model_lower or "o3" in model_lower or "gpt-5" in model_lower:
            final_temp = None
            
        llm_seed = getattr(settings, "llm_seed", 42)

        logger.info(f"Gerando inferência (STREAM) | provider={provider} | model={model_name} | temp={final_temp}")

        if provider == "openai":
            async for chunk in self._generate_openai_stream(messages, model_name, final_temp, llm_seed, max_tokens): yield chunk
        elif provider == "deepseek":
            async for chunk in self._generate_deepseek_stream(messages, model_name, final_temp, llm_seed, max_tokens): yield chunk
        elif provider == "vllm":
            async for chunk in self._generate_vllm_stream(messages, model_name, final_temp, llm_seed, max_tokens): yield chunk
        elif provider == "anthropic":
            async for chunk in self._generate_anthropic_stream(messages, model_name, final_temp, llm_seed, max_tokens): yield chunk
        else:
            logger.warning(f"Provedor '{provider}' não suporta stream ou não reconhecido. Fallback OpenAI.")
            async for chunk in self._generate_openai_stream(messages, model_name, final_temp, llm_seed, max_tokens): yield chunk

    async def _generate_openai_stream(self, messages: List[Dict[str, str]], model: str, temperature: Optional[float], seed: int, max_tokens: int) -> AsyncGenerator[str, None]:
        if not self.openai_client: raise ValueError("OpenAI API key não configurada.")
        
        kwargs = {"model": model, "messages": messages, "seed": seed, "stream": True}

        # Correção de tokens
        model_lower = model.lower()
        if any(m in model_lower for m in ["o1", "o3", "gpt-5"]):
            kwargs["max_completion_tokens"] = 30000
        else:
            kwargs["max_completion_tokens"] = max_tokens

        if temperature is not None:
            kwargs["temperature"] = temperature
            if temperature == 0.0: kwargs["top_p"] = 0.0001
            
        try:
            response_stream = await self.openai_client.chat.completions.create(**kwargs)
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Falha no streaming OpenAI: {e}")
            yield f" Erro: {str(e)}"

    async def _generate_deepseek_stream(self, messages: List[Dict[str, str]], model: str, temperature: Optional[float], seed: int, max_tokens: int) -> AsyncGenerator[str, None]:
        if not self.deepseek_client: raise ValueError("DeepSeek API key não configurada.")
        
        kwargs = {"model": model, "messages": messages, "seed": seed, "stream": True}

        # Correção de tokens específica para o DeepSeek
        model_lower = model.lower()
        if "reasoner" in model_lower:
            kwargs["max_tokens"] = 8192
        else:
            kwargs["max_tokens"] = max_tokens
        
        if temperature is not None:
            kwargs["temperature"] = temperature
            if temperature == 0.0: kwargs["top_p"] = 0.0001
            
        try:
            response_stream = await self.deepseek_client.chat.completions.create(**kwargs)
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Falha no streaming DeepSeek: {e}")
            yield f" Erro: {str(e)}"

    async def _generate_vllm_stream(self, messages: List[Dict[str, str]], model: str, temperature: Optional[float], seed: int, max_tokens: int) -> AsyncGenerator[str, None]:
        if not self.vllm_client: raise ValueError("Cliente vLLM não inicializado.")
        kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens, "seed": seed, "stream": True}
        if temperature is not None:
            kwargs["temperature"] = temperature
            if temperature == 0.0: kwargs["top_p"] = 0.0001
            
        try:
            response_stream = await self.vllm_client.chat.completions.create(**kwargs)
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Falha no streaming vLLM: {e}")
            yield f" Erro: {str(e)}"

    async def _generate_anthropic_stream(self, messages: List[Dict[str, str]], model: str, temperature: Optional[float], seed: int, max_tokens: int) -> AsyncGenerator[str, None]:
        if not self.anthropic_client: raise ValueError("Anthropic API key não configurada.")
        try:
            system_content = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
            user_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages if msg["role"] != "system"]
            kwargs = {"model": model, "max_tokens": max_tokens, "system": system_content, "messages": user_messages}
            if temperature is not None:
                kwargs["temperature"] = temperature
                
            async with self.anthropic_client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Falha no streaming Anthropic: {e}")
            yield f" Erro: {str(e)}"