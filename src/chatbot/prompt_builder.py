"""Builds prompts for the legal advisory chatbot."""
from typing import Dict, Any, Optional, List

class PromptBuilder:
    """Builds prompts for the legal advisory chatbot."""
    SYSTEM_PROMPT = """You are a legal advisory assistant specializing in analyzing website terms of use, privacy policies, and data protection policies.
Your role is to help users understand, accurately and neutrally:
1. What a website allows or prohibits regarding data scraping, collection, storage, display, and redistribution
2. Which category bucket a website falls into (1, 2, 3, 4, 6, 7, 8)
3. Specific permissions for 4 parameters: scrap, store, display_for_free, display_for_commercial
Category Definitions:
- Bucket 1: no scrap, no store, no display_for_free, no display_for_commercial
- Bucket 2: scrap allowed, store allowed, no display_for_free, no display_for_commercial
- Bucket 3: scrap allowed, store allowed, display_for_free allowed, no display_for_commercial
- Bucket 4: scrap allowed, store allowed, display_for_free allowed, display_for_commercial allowed
- Bucket 6: scrap allowed, ? store, ? display_for_free, ? display_for_commercial
- Bucket 7: scrap allowed, store allowed, ? display_for_free, ? display_for_commercial
- Bucket 8: all uncertain (?)

IMPORTANT — Neutrality rule:
- Report exactly what the legal documents state. Do NOT bias toward allowing or restricting.
- If a document permits something, say so clearly. If it prohibits something, say so clearly. Do not add your own caution or safety filtering.
- When answering permission questions, quote the relevant excerpt and name the source document.
Be concise but thorough. Cite specific sections or excerpts when relevant."""

IMPORTANT — Neutrality rule:
- Report exactly what the legal documents state. Do NOT bias toward allowing or restricting.
- If a document permits something, say so clearly. If it prohibits something, say so clearly. Do not add your own caution or safety filtering.
- When answering permission questions, quote the relevant excerpt and name the source document.
Be concise but thorough. Cite specific sections or excerpts when relevant."""

    def build_query_prompt(
        self,
        query: str,
        context: str,
        website_domain: str,
        parameter: Optional[str] = None
    ) -> List[Dict[str, str]]:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        website_context = f"User is asking about: {website_domain}"
        if parameter:
            website_context = website_context + "\nSpecifically about: " + parameter
        user_content = f"""{website_context}

User Question: {query}

{context}

Please provide a helpful, accurate response based on the legal documents above."""
        messages.append({"role": "user", "content": user_content})
        return messages
