"""Generates responses for the legal advisory chatbot."""
import os
import json
from typing import Dict, Any, Optional, List
from src.chatbot.prompt_builder import PromptBuilder
from src.rag.query_engine import QueryEngine
from config import EngineConfig

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


class GeminiClient:
    """Google Gemini API client (supports free-tier Flash models)."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.http_client = httpx.Client(timeout=120.0)

    def chat(self, messages: List[Dict[str, str]], model: str = None, max_tokens: int = 2048) -> str:
        """Send chat request to Gemini API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                      Gemini uses 'user' and 'model' roles only (no 'system' at top level).
            model: Override model name (uses self.model if None).
            max_tokens: Maximum output tokens.
        """
        model = model or self.model

        # Gemini API format: separate systemInstruction from contents
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "system":
                system_instruction = text
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})

        url = (
            f"{self.base_url}/models/{model}:generateContent"
            f"?key={self.api_key}"
        )
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.0,
            }
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        print(f"[DEBUG] Gemini request URL: {url}")
        print(f"[DEBUG] Gemini request payload: {json.dumps(payload, indent=2)}")
        response = self.http_client.post(url, json=payload)
        response.raise_for_status()
        result = response.json()

        # Extract text from response
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise ValueError(f"Gemini API returned unexpected structure: {result}")


class ResponseGenerator:
    """
    Generates responses for the legal advisory chatbot.
    Uses stored LegalAnalysis (from summary JSON) for permission/bucket questions,
    so results are consistent with the website analysis page.
    Falls back to RAG for general questions about legal concepts.
    """

    PERMISSION_PARAMS = [
        "scraping", "manual_collection", "storing",
        "free_display", "subscription_display",
        "free_redistribute", "subscription_redistribute"
    ]

    def __init__(
        self,
        query_engine: QueryEngine,
        prompt_builder: PromptBuilder,
        llm_client: Any = None,
        summaries_directory: str = "./data/summaries"
    ):
        self.query_engine = query_engine
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client
        self.summaries_directory = summaries_directory

    def generate_response(
        self,
        query: str,
        website_domain: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        # Load stored analysis for this website
        analysis = self._load_stored_analysis(website_domain)

        # Check if query is about a specific permission or bucket category
        query_lower = query.lower()
        param_key = self._match_permission_param(query_lower)
        is_bucket_question = "bucket" in query_lower or "category" in query_lower

        if analysis and (param_key or is_bucket_question):
            # Use stored analysis — consistent with website analysis page
            return self._build_stored_response(query, website_domain, analysis, param_key, is_bucket_question)
        else:
            # General question — use RAG as before
            return self._generate_rag_response(query, website_domain, conversation_history)

    def _load_stored_analysis(self, website_domain: str) -> Optional[Dict]:
        """Load stored LegalAnalysis from summary JSON file."""
        summary_path = os.path.join(
            self.summaries_directory,
            f"summary_{website_domain.replace('.', '_')}.json"
        )
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                return json.load(f)
        return None

    def _match_permission_param(self, query_lower: str) -> Optional[str]:
        """Check if query mentions a specific permission parameter."""
        for param in self.PERMISSION_PARAMS:
            if param.replace("_", " ") in query_lower or param.replace("_", "") in query_lower:
                return param
        return None

    def _source_label(self, source: str) -> str:
        """Map source key to display label."""
        labels = {
            "terms_of_service": "Terms of Service",
            "privacy_policy": "Privacy Policy",
            "robots_txt": "robots.txt"
        }
        return labels.get(source, source)

    def _build_stored_response(
        self,
        query: str,
        website_domain: str,
        analysis: Dict,
        param_key: Optional[str],
        is_bucket_question: bool
    ) -> Dict[str, Any]:
        """Build response from stored analysis with quoted source excerpts."""
        parts = []

        if is_bucket_question:
            bucket_num = analysis.get("category", 4)
            bucket_names = {
                1: "Full Access (scrape, store, display, redistribute allowed)",
                2: "Display Only (scrape, store, display allowed; no redistribute)",
                3: "Storage Only (scrape, store allowed; no display, no redistribute)",
                4: "Manual Collection Only (no scrape, no store, no display, no redistribute)"
            }
            parts.append(f"**Bucket {bucket_num}: {bucket_names.get(bucket_num, 'Unknown')}**")
            parts.append("")
            parts.append(f"Based on my analysis of the legal documents for {website_domain}.")
            perms = analysis.get("permissions", {})
            allowed = [k for k, v in perms.items() if v.get("level") == "allowed"]
            not_allowed = [k for k, v in perms.items() if v.get("level") == "not_allowed"]
            if allowed:
                parts.append(f"**Allowed:** {', '.join(allowed)}")
            if not_allowed:
                parts.append(f"**Not Allowed:** {', '.join(not_allowed)}")

        if param_key:
            perms = analysis.get("permissions", {})
            if param_key in perms:
                p = perms[param_key]
                level = p.get("level", "uncertain")
                excerpts = p.get("relevant_excerpts", [])
                source_docs = p.get("source_documents", [])

                # Build answer in requested format: "Yes/No, because [reason] as quoted in '...' in [Document]"
                verb_map = {"allowed": "Yes", "not_allowed": "No", "uncertain": "Uncertain"}
                verb = verb_map.get(level, "Uncertain")
                param_display = param_key.replace("_", " ")

                parts.append(f"**{param_display.title()}:** {verb.upper()}")

                if excerpts and source_docs:
                    parts.append("")
                    parts.append(f"**Answer:** {verb}, you **{'can' if level == 'allowed' else 'cannot' if level == 'not_allowed' else 'may or may not be able to'}** {param_display}.")
                    parts.append("")
                    for i, (excerpt, src) in enumerate(zip(excerpts, source_docs), 1):
                        doc_label = self._source_label(src)
                        parts.append(f'  Reason {i}: "{excerpt}" — quoted in [{doc_label}]')
                else:
                    reasoning = p.get("reasoning", "No specific reasoning available.")
                    parts.append(f"\n**Answer:** {verb}. {reasoning}")

        response_text = "\n".join(parts)
        return {
            "response": response_text,
            "sources": [],
            "query": query,
            "website_domain": website_domain,
            "context_used": "Stored LegalAnalysis (consistent with website analysis page)"
        }

    def _generate_rag_response(
        self,
        query: str,
        website_domain: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Fall back to RAG for general legal concept questions."""
        rag_results = self.query_engine.query(
            query_text=query,
            website_domain=website_domain,
            n_results=3
        )

        # Hallucination guard: refuse if no relevant results or context too short
        MIN_CONTEXT_LENGTH = 100
        if rag_results["total_results"] == 0 or len(rag_results["context"]) < MIN_CONTEXT_LENGTH:
            return {
                "response": (
                    f"I don't have sufficient relevant information about '{website_domain}' in my knowledge base "
                    f"to answer that question. Please try asking about specific permissions "
                    f"(e.g., 'Can I scrape this website?') or 'What is the bucket category?' — "
                    f"or re-analyze the website to populate the knowledge base first."
                ),
                "sources": [],
                "query": query,
                "website_domain": website_domain,
                "context_used": ""
            }

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
            return "LLM client not configured. Please set MINIMAX_API_KEY or GEMINI_API_KEY environment variable."
        try:
            # MiniMaxClient uses model kwarg, GeminiClient uses model kwarg too
            return self.llm_client.chat(messages)
        except Exception as e:
            return f"Error calling LLM: {str(e)}"