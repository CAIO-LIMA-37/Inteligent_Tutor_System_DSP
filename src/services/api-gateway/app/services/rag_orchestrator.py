"""RAG Orchestrator"""
import asyncio
import time
import re
from typing import Dict, List, Optional, Any, AsyncGenerator
import logging

from app.clients.intent_client import IntentClient
from app.clients.embedding_client import EmbeddingClient
from app.clients.qdrant_client import QdrantManager
from app.services.search_service import SearchService
from app.services.llm_service import LLMService
from app.services.prompt_engineering import get_rag_system_prompt, get_enhanced_query_prompt
from redis import asyncio as aioredis
from app.config import settings

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """Orchestrates the RAG pipeline for answering queries."""

    def __init__(
        self,
        intent_client: IntentClient,
        embedding_client: EmbeddingClient,
        qdrant: QdrantManager,
        redis: aioredis.Redis
    ):
        self.intent_client = intent_client
        self.qdrant = qdrant
        self.search_service = SearchService(
            qdrant=qdrant,
            embedding_client=embedding_client,
            redis=redis
        )
        self.llm_service = LLMService()

    async def _generate_enhanced_queries(
        self,
        query: str,
        subject: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> List[Dict[str, Optional[str]]]:
        """
        Generate multiple focused search queries from the user query.
        Uses a fast model (GPT-5 Nano) for query enhancement.

        Returns:
            List of {"query": str, "book": Optional[str]}
        """
        try:
            # Get available books
            available_books = await self.qdrant.get_books()

            # Generate enhancement prompt
            prompt = get_enhanced_query_prompt(
                query=query,
                subject=subject,
                available_books=available_books,
                conversation_history=conversation_history
            )

            # Call LLM for query enhancement
            messages = [{"role": "user", "content": prompt}]
            llm_result = await self.llm_service.generate(
                messages=messages,
                model=settings.query_enhancement_model
            )
            response = llm_result["text"]

            logger.debug(f"Query enhancement response: {response}")

            # Parse the XML-like response
            retrievals = []
            for i in range(1, 4):
                pattern = f'<retrieval{i} book="([^"]+)">(.*?)</retrieval{i}>'
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    book = match.group(1).strip()
                    retrieval_query = match.group(2).strip()

                    # Convert "all" to None for no book filter
                    if book.lower() == "all":
                        book = None

                    retrievals.append({
                        "query": retrieval_query,
                        "book": book
                    })

            logger.info(f"Generated {len(retrievals)} enhanced queries: {retrievals}")
            return retrievals if retrievals else [{"query": query, "book": None}]

        except Exception as e:
            logger.warning(f"Query enhancement failed, using original query: {e}")
            return [{"query": query, "book": None}]

    async def _build_rag_messages(
        self,
        query: str,
        subject: str,
        conversation_history: Optional[List[Dict]] = None,
        book_filter: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Função auxiliar para construir as mensagens do RAG.
        Evita duplicação de código entre os métodos síncronos e de streaming.
        """
        intent_task = asyncio.create_task(self.intent_client.classify(query))
        enhanced_queries_task = asyncio.create_task(
            self._generate_enhanced_queries(query, subject, conversation_history)
        )

        intent_result = await intent_task
        intent = intent_result.get("intent", "question_answering")
        top_k = 4 if intent == "searching_for_information" else 3

        enhanced_queries = await enhanced_queries_task

        search_results = await self.search_service.search_with_enhanced_queries(
            queries=enhanced_queries,
            intent=intent,
            top_k=top_k
        )

        if search_results:
            logger.info("\n" + "="*60)
            logger.info("INSPECIONANDO O PRIMEIRO CHUNK RECUPERADO DO QDRANT:")
            logger.info(f"Chaves disponíveis: {list(search_results[0].keys())}")
            logger.info(f"Objeto bruto: {search_results[0]}")
            logger.info("="*60 + "\n")

        system_prompt = get_rag_system_prompt(
            intent=intent,
            subject=subject,
            context_chunks=search_results
        )

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            for msg in conversation_history[-6:]:
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

        fontes_disponiveis = ""
        if search_results:
            seen_sources = set()
            for chunk in search_results:
                book = chunk.get("book_name", "Desconhecido")
                chapter = chunk.get("chapter_title", "Desconhecido")
                topic = chunk.get("topic", "Desconhecido")
                page = chunk.get("page", "Não informada")
                
                source_sig = f"{book}-{chapter}-{topic}-{page}"
                if source_sig not in seen_sources:
                    seen_sources.add(source_sig)
                    fontes_disponiveis += f"- Book: {book} | Chapter: {chapter} | Section: {topic} | Page: {page}\n"

        enhanced_query = f"""{query}

<instrucoes_criticas>
[LANGUAGE LOCK]
Detect the language of the question above.
IF ENGLISH: You MUST start your response with "The answer to your question is:" and proceed in English.
IF PORTUGUESE: Você DEVE iniciar sua resposta com "A resposta para sua pergunta é:" e prosseguir em português.

[SYSTEM RULES]
Answer using EXCLUSIVELY the provided academic context.
Follow all formatting rules (LaTeX for math, no brackets for equations).

[REFERENCES TEMPLATE]
(EN) For more details, consult the following reference:
Book: [Name] | Chapter: [Name] | Section: [Name] | Page: [Page]

(PT) Para aprofundar seus conhecimentos consulte a seguinte referência:
Livro: [Nome] | Capítulo: [Nome] | Seção: [Seção] | Página: [Página]
</instrucoes_criticas>"""

        messages.append({"role": "user", "content": enhanced_query})
        
        # Retornamos as mensagens e os search_results para compor a resposta do benchmark síncrono
        return messages, search_results, intent

    async def process_query(
        self,
        query: str,
        subject: str = settings.default_subject,
        conversation_history: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        book_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.
        (Versão Síncrona Clássica para o Frontend)
        """
        start_time = time.time()

        messages, search_results, intent = await self._build_rag_messages(
            query, subject, conversation_history, book_filter
        )

        tokens_used = None
        try:
            llm_result = await self.llm_service.generate(
                messages=messages,
                model=model
            )
            response = llm_result["text"]
            tokens_used = llm_result.get("total_tokens")

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            response = "I apologize, but I encountered an error generating a response. Please try again."

        processing_time = (time.time() - start_time) * 1000

        return {
            "response": response,
            "tokens_used": tokens_used,
            "intent": intent,
            "sources": [
                {
                    "text": chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"],
                    "book": chunk["book_name"],
                    "chapter": chunk["chapter_title"],
                    "topic": chunk.get("topic"),
                    "score": chunk["score"]
                }
                for chunk in search_results
            ],
            "search_results": search_results,
            "model_used": model or "gpt-5-nano",
            "processing_time_ms": processing_time
        }

    async def process_single_query(
        self,
        query: str,
        subject: str = settings.default_subject,
        model: Optional[str] = None,
        book_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single query without conversation history.
        Simplified version for one-shot queries.
        """
        return await self.process_query(
            query=query,
            subject=subject,
            conversation_history=None,
            model=model,
            book_filter=book_filter
        )

    # MÉTODOS DE STREAMING ADICIONADOS PARA O BENCHMARK
    async def process_query_stream(
        self,
        query: str,
        subject: str = settings.default_subject,
        conversation_history: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        book_filter: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Executa a pipeline do RAG, mas retorna a resposta como um fluxo de dados.
        (Versão de Streaming Exclusiva para Testes)
        """
        # Montamos a pergunta e pegamos os vetores normalmente (processo síncrono rápido)
        messages, _, _ = await self._build_rag_messages(
            query, subject, conversation_history, book_filter
        )

        try:
            # Em vez de chamar generate(), chamamos generate_stream() e iteramos os blocos
            async for chunk in self.llm_service.generate_stream(
                messages=messages,
                model=model
            ):
                yield chunk
        except Exception as e:
            logger.error(f"LLM streaming generation failed: {e}")
            yield f"Erro durante o streaming do LLM: {str(e)}"

    async def process_single_query_stream(
        self,
        query: str,
        subject: str = settings.default_subject,
        model: Optional[str] = None,
        book_filter: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Executa uma única query em formato de streaming.
        """
        # Reutiliza o método principal sem o histórico, aproveitando o yield assíncrono
        async for chunk in self.process_query_stream(
            query=query,
            subject=subject,
            conversation_history=None,
            model=model,
            book_filter=book_filter
        ):
            yield chunk