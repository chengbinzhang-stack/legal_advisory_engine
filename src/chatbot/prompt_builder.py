"""Builds prompts for the legal advisory chatbot."""
from typing import Dict, Any, Optional, List

class PromptBuilder:
    """Builds prompts for the legal advisory chatbot."""
    SYSTEM_PROMPT = """You are a legal advisory assistant specializing in analyzing website terms of use, privacy policies, and data protection policies.
Your role is to help users understand:
1. What a website allows or prohibits regarding data scraping, collection, storage, display, and redistribution
2. Which category bucket a website falls into (1-4)
3. Specific permissions for 7 parameters: scraping, manual collection, storing, free display, subscription display, free redistribution, subscription redistribution
Category Definitions:
- Bucket 1: scrape, store, display, redistribute allowed
- Bucket 2: scrape, store, display allowed, no redistribute
- Bucket 3: scrape, store allowed, no display, no redistribute
- Bucket 4: manually collect data only, no scrape, no store, no display, no redistribute
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
