"""LLM-based legal document classifier for 7 parameters."""
import json
import re
from typing import Dict, List, Tuple, Optional
from src.models.legal_analysis import (
    LegalAnalysis, PermissionAnalysis, PermissionLevel
)
from src.classifier.category_buckets import CATEGORY_DEFINITIONS, WebsiteCategory, PARAM_TO_CATEGORY_FIELD


def _build_llm_client(api_key: str, base_url: str):
    """Create MiniMax LLM client."""
    import httpx
    class MiniMaxClient:
        def __init__(self, api_key, base_url):
            self.api_key = api_key
            self.base_url = base_url
            self.http_client = httpx.Client(timeout=120.0)

        def chat(self, messages, model="MiniMax-Text-01", max_tokens=4096):
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
            response = self.http_client.post(
                f"{self.base_url}/text/chatcompletion_v2",
                headers=headers, json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    return MiniMaxClient(api_key, base_url)


PERMISSION_PARAMS = [
    "scraping", "manual_collection", "storing",
    "free_display", "subscription_display",
    "free_redistribute", "subscription_redistribute"
]

SYSTEM_PROMPT = """You are a legal analyst specializing in Terms of Service / Privacy Policy analysis.

Given the full text of a website's legal document, analyze it and output a structured JSON object describing permissions for 7 parameters.

The 7 parameters to analyze:
1. scraping - whether automated crawling/scraping is allowed
2. manual_collection - whether manual copying/reading is allowed
3. storing - whether storing/caching data is allowed
4. free_display - whether displaying content publicly for free is allowed
5. subscription_display - whether displaying content behind a paywall is allowed
6. free_redistribute - whether redistributing/sharing/publishing content for free is allowed
7. subscription_redistribute - whether redistributing content for commercial/subscription purposes is allowed

For each parameter, determine: allowed, not_allowed, or uncertain

Also extract:
- reference_urls: List of URLs mentioned in the document (e.g., API docs, developer terms, data licensing pages)
- unique_findings: Special clauses like "API Access", "Non-Commercial Restriction", "Attribution Required", "DMCA Protected", "Open Access", "Creative Commons"

Output ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{
  "scraping": {"permission": "allowed|not_allowed|uncertain", "reasoning": "...", "reference_urls": [], "relevant_excerpts": [{"text": "exact quote from document", "source": "terms_of_service|privacy_policy|robots_txt"}]},
  "manual_collection": {"permission": "...", "reasoning": "...", "reference_urls": [], "relevant_excerpts": []},
  "storing": {"permission": "...", "reasoning": "...", "reference_urls": [], "relevant_excerpts": []},
  "free_display": {"permission": "...", "reasoning": "...", "reference_urls": [], "relevant_excerpts": []},
  "subscription_display": {"permission": "...", "reasoning": "...", "reference_urls": [], "relevant_excerpts": []},
  "free_redistribute": {"permission": "...", "reasoning": "...", "reference_urls": [], "relevant_excerpts": []},
  "subscription_redistribute": {"permission": "...", "reasoning": "...", "reference_urls": [], "relevant_excerpts": []},
  "reference_urls": ["https://..."],
  "unique_findings": ["Found indicators of ...", ...]
}

Key rules:
- For each relevant_excerpt, you MUST identify which of the 3 source documents it came from: "terms_of_service", "privacy_policy", or "robots_txt"
- Look for explicit permission or prohibition language: "you may", "you can", "you must not", "prohibited", "not allowed", "restricted"
- Check robots.txt references if present
- Check API/developer terms links mentioned in the document
- Check for "all rights reserved", "non-commercial", "attribution required", "DMCA" mentions
- If multiple contradictory statements exist, report "uncertain" and list both the allowing and restricting excerpts — do NOT automatically prefer the more restrictive interpretation. Be neutral: report what the document actually says.
- "reference_urls" should list all relevant URLs found in the legal text itself (terms pages, API docs, privacy policy links, etc.)
- Extract URLs from the text that are explicitly referenced (do not invent URLs)
- IMPORTANT: All reasoning and excerpts must be grounded ONLY in the provided legal documents. Do not make inferences beyond what is stated.
- IMPORTANT: Be strictly neutral and objective. Do NOT bias toward allowing or restricting. Report the document's actual position accurately, even if it is permissive. Do not add your own caution or safety interpretation.
"""


class LegalClassifier:
    """Classifies legal documents using LLM for semantic understanding."""

    def __init__(self, api_key: str = None, base_url: str = "https://api.minimax.chat/v1"):
        self.llm_client = None
        if api_key:
            self.llm_client = _build_llm_client(api_key, base_url)

    def classify_permissions(
        self,
        text: str,
        website_url: str,
        website_domain: str,
        robots_txt: str = ""
    ) -> LegalAnalysis:
        """
        Classify permissions using LLM if available, otherwise fall back to heuristic.
        """
        if self.llm_client:
            return self._classify_with_llm(text, website_url, website_domain, robots_txt)
        else:
            return self._classify_heuristic(text, website_url, website_domain, robots_txt)

    def _classify_with_llm(
        self,
        text: str,
        website_url: str,
        website_domain: str,
        robots_txt: str
    ) -> LegalAnalysis:
        """
        Use LLM to analyze the legal document with full semantic understanding.
        """
        # Truncate text to avoid token limits (first 8000 chars is usually enough for terms)
        text_to_analyze = text[:8000]

        user_prompt = f"Website URL: {website_url}\nWebsite domain: {website_domain}\n\nLegal document text:\n{text_to_analyze}"

        try:
            response = self.llm_client.chat([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ], max_tokens=4096)

            # Parse JSON response
            result = self._parse_llm_response(response)

            # Build permissions dict
            permissions = {}
            all_reference_urls = result.get("reference_urls", [])

            for param in PERMISSION_PARAMS:
                param_data = result.get(param, {})
                perm_level = param_data.get("permission", "uncertain")

                # Collect reference URLs specific to this parameter
                param_refs = param_data.get("reference_urls", [])
                refs_text = f" (References: {', '.join(param_refs)})" if param_refs else ""

                # Parse excerpts: new format is [{"text": "...", "source": "..."}]
                raw_excerpts = param_data.get("relevant_excerpts", [])
                excerpt_texts = []
                source_docs = []
                for ex in raw_excerpts:
                    if isinstance(ex, dict):
                        excerpt_texts.append(ex.get("text", ""))
                        src = ex.get("source", "")
                        if src:
                            source_docs.append(src)
                    elif isinstance(ex, str):
                        excerpt_texts.append(ex)

                permission = PermissionAnalysis(
                    parameter_name=param,
                    permission=PermissionLevel(perm_level),
                    reasoning=param_data.get("reasoning", "No reasoning provided") + refs_text,
                    relevant_excerpts=[e for e in excerpt_texts if e],
                    source_documents=source_docs,
                    confidence_score=0.95 if perm_level != "uncertain" else 0.5
                )
                permissions[param] = permission

            # Build unique findings
            unique_findings = result.get("unique_findings", [])

            # Determine category
            category, category_reasoning = self._determine_category(permissions)

            # Generate summary
            summary_text = self._generate_summary(website_domain, category, permissions, all_reference_urls)

            return LegalAnalysis(
                website_url=website_url,
                website_domain=website_domain,
                category=category,
                category_reasoning=category_reasoning,
                permissions=permissions,
                unique_findings=unique_findings,
                summary_text=summary_text,
                raw_analysis=text[:500]
            )

        except Exception as e:
            # Fall back to heuristic on error
            return self._classify_heuristic(text, website_url, website_domain, robots_txt, str(e))

    def _parse_llm_response(self, response: str) -> dict:
        """
        Parse the LLM JSON response. Tries to extract JSON from the response text.
        """
        # Try to find JSON in the response
        text = response.strip()

        # If it starts with markdown code block, strip it
        if text.startswith("```"):
            text = re.sub(r"^```json?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object within the text
        try:
            # Look for {...} pattern
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass

        # Return empty result if parsing fails
        return {}

    def _classify_heuristic(
        self,
        text: str,
        website_url: str,
        website_domain: str,
        robots_txt: str = "",
        error: str = ""
    ) -> LegalAnalysis:
        """
        Fallback heuristic classifier when LLM is unavailable.
        """
        text_lower = text.lower()

        # Heuristic rules (simplified)
        permissions = {}

        # Build a simple heuristic for each param
        param_rules = {
            "scraping": {
                "forbidden": ["prohibit scraping", "scraping prohibited", "no automated"],
                "allowed": ["scrap", "crawl", "bot"]
            },
            "manual_collection": {
                "forbidden": ["cannot copy", "copying prohibited"],
                "allowed": ["manual", "copy", "read", "browse"]
            },
            "storing": {
                "forbidden": ["cannot store", "no storage"],
                "allowed": ["store", "cache", "retain"]
            },
            "free_display": {
                "forbidden": ["cannot display", "display prohibited"],
                "allowed": ["display", "show", "view", "public"]
            },
            "subscription_display": {
                "forbidden": ["no paid access", "subscription not allowed"],
                "allowed": ["subscription", "paid", "premium"]
            },
            "free_redistribute": {
                "forbidden": ["cannot redistribute", "no redistribution", "must not distribute", "may not distribute"],
                "allowed": ["redistribute", "share", "distribute", "publish"]
            },
            "subscription_redistribute": {
                "forbidden": ["cannot redistribute", "no redistribution", "resale not allowed"],
                "allowed": ["redistribute", "share", "sell"]
            },
        }

        for param, rules in param_rules.items():
            forbid_count = sum(1 for kw in rules["forbidden"] if kw in text_lower)
            allow_count = sum(1 for kw in rules["allowed"] if kw in text_lower)

            # Negation context check for redistribute
            if param in ["free_redistribute", "subscription_redistribute"]:
                negate_count = self._count_negation_contexts(text_lower, rules["allowed"])
                allow_count = max(0, allow_count - negate_count)
                forbid_count += negate_count

            if forbid_count > allow_count:
                level = PermissionLevel.NOT_ALLOWED
                conf = 0.8
            elif allow_count > 0:
                level = PermissionLevel.ALLOWED
                conf = 0.7
            else:
                level = PermissionLevel.UNCERTAIN
                conf = 0.3

            reasoning = f"Found {forbid_count} prohibition and {allow_count} permission indicators"
            if error:
                reasoning += f" (LLM unavailable: {error})"

            permissions[param] = PermissionAnalysis(
                parameter_name=param,
                permission=level,
                reasoning=reasoning,
                confidence_score=conf
            )

        category, category_reasoning = self._determine_category(permissions)
        summary_text = self._generate_summary(website_domain, category, permissions, [])
        unique_findings = self._extract_unique_findings(text_lower, permissions)

        return LegalAnalysis(
            website_url=website_url,
            website_domain=website_domain,
            category=category,
            category_reasoning=category_reasoning,
            permissions=permissions,
            unique_findings=unique_findings,
            summary_text=summary_text,
            raw_analysis=text[:500]
        )

    def _count_negation_contexts(self, text_lower: str, allowed_keywords: List[str]) -> int:
        """Detect negation contexts (must not, cannot, etc.) around keywords."""
        negation_patterns = [
            r"must\s+not\s+\w+", r"cannot\s+\w+", r"may\s+not\s+\w+",
            r"shall\s+not\s+\w+", r"prohibit\S*\s+\w+",
            r"not\s+authorized\s+to\s+\w+", r"not\s+permitted\s+to\s+\w+",
            r"no\s+right\s+to\s+\w+",
        ]
        count = 0
        for kw in allowed_keywords:
            for match in re.finditer(re.escape(kw), text_lower):
                pos = match.start()
                context = text_lower[max(0, pos-80):pos]
                for pat in negation_patterns:
                    if re.search(pat, context):
                        count += 1
                        break
        return count

    def _determine_category(
        self,
        permissions: Dict[str, PermissionAnalysis]
    ) -> Tuple[WebsiteCategory, str]:
        """
        Determine bucket by profile matching: for each bucket, count how many
        of its expected-permitted params are ALLOWED and how many of its
        expected-prohibited params are NOT_ALLOWED. Score = matches - mismatches.
        The bucket with the highest score is the best fit.
        """
        best_bucket = WebsiteCategory.BUCKET_4
        best_score = -999

        for bucket, definition in CATEGORY_DEFINITIONS.items():
            score = 0
            for param_name, category_field in PARAM_TO_CATEGORY_FIELD.items():
                expected_allowed = getattr(definition, category_field)
                actual = permissions.get(param_name)
                if actual is None:
                    continue
                actual_is_allowed = actual.permission == PermissionLevel.ALLOWED

                if expected_allowed:
                    # Bucket expects this param to be allowed
                    if actual_is_allowed:
                        score += 1  # matches expectation
                    elif actual.permission == PermissionLevel.NOT_ALLOWED:
                        score -= 1  # violates expectation
                    # uncertain: no score change
                else:
                    # Bucket expects this param to be NOT allowed
                    if actual.permission == PermissionLevel.NOT_ALLOWED:
                        score += 1  # matches expectation
                    elif actual_is_allowed:
                        score -= 1  # violates expectation (allowed when bucket expects not)
                    # uncertain: no score change

            if score > best_score:
                best_score = score
                best_bucket = bucket

        definitions = CATEGORY_DEFINITIONS[best_bucket]
        reasoning = f"Website matches {definitions.name} profile with score {best_score}"
        return best_bucket, reasoning

    def _extract_unique_findings(
        self,
        text_lower: str,
        permissions: Dict[str, PermissionAnalysis]
    ) -> List[str]:
        findings = []
        special_patterns = [
            ("API Access", ["api", "developer", "programmatic", "developer access"]),
            ("Creative Commons", ["creative commons", "cc-by", "cc0", "open license"]),
            ("Non-Commercial", ["non-commercial", "noncommercial", "personal use only"]),
            ("Attribution Required", ["attribution", "credit", "must attribute", "credit required"]),
            ("DMCA Protected", ["dmca", "copyright", "all rights reserved", "protected by copyright"]),
            ("Open Access", ["open access", "public domain", "free to use", "no restrictions"]),
        ]
        for finding_name, keywords in special_patterns:
            if any(kw in text_lower for kw in keywords):
                findings.append(f"Found indicators of {finding_name}")
        return findings

    def _generate_summary(
        self,
        website_domain: str,
        category: WebsiteCategory,
        permissions: Dict[str, PermissionAnalysis],
        reference_urls: List[str]
    ) -> str:
        definition = CATEGORY_DEFINITIONS[category]
        lines = [
            f"Website: {website_domain}",
            f"Category: Bucket {category.value} - {definition.name}",
            "",
            "Permission Summary:",
        ]
        for param_name, perm in permissions.items():
            emoji = "YES" if perm.permission == PermissionLevel.ALLOWED else "NO" if perm.permission == PermissionLevel.NOT_ALLOWED else "?"
            lines.append(f"  [{emoji}] {param_name}: {perm.permission.value}")

        if reference_urls:
            lines.append("")
            lines.append("Reference URLs from legal document:")
            for url in reference_urls[:10]:
                lines.append(f"  - {url}")

        return "\n".join(lines)