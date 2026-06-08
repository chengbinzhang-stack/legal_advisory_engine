"""Generates responses for the legal advisory chatbot."""
import os
import json
from typing import Dict, Any, Optional, List
from src.chatbot.prompt_builder import PromptBuilder
from src.rag.query_engine import QueryEngine

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


class MiniMaxClient:
    """MiniMax API client."""

    def __init__(self, api_key: str, base_url: str = "https://api.minimax.chat/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.http_client = httpx.Client(timeout=60.0)

    def chat(self, messages: List[Dict[str, str]], model: str = "MiniMax-Text-01", max_tokens: int = 2048) -> str:
        """Send chat request to MiniMax API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        }
        response = self.http_client.post(
            f"{self.base_url}/text/chatcompletion_v2",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]


class ResponseGenerator:
    """Generates responses for the legal advisory chatbot."""

    def __init__(
        self,
        query_engine: QueryEngine,
        prompt_builder: PromptBuilder,
        llm_client: Any = None
    ):
        self.query_engine = query_engine
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client

    def generate_response(
        self,
        query: str,
        website_domain: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        rag_results = self.query_engine.query(
            query_text=query,
            website_domain=website_domain,
            n_results=5
        )
        messages = self.prompt_builder.build_query_prompt(
            query=query,
            context=rag_results["context"],
            website_domain=website_domain
        )
        if conversation_history:
            messages = self._add_history(messages, conversation_history)
        response_text = self._call_llm(messages)
        return {
            "response": response_text,
            "sources": rag_results["results"],
            "query": query,
            "website_domain": website_domain,
            "context_used": rag_results["context"][:500] + "..." if len(rag_results["context"]) > 500 else rag_results["context"]
        }

    def _add_history(
        self,
        messages: List[Dict[str, str]],
        history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        system_msg = messages[0]
        rest = messages[1:]
        return [system_msg] + history + rest

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        if self.llm_client is None:
            return "LLM client not configured. Please set MINIMAX_API_KEY environment variable."
        try:
            return self.llm_client.chat(messages)
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
