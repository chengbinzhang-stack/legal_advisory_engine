"""LLM-based legal document classifier for 7 parameters."""
import json
import re
from typing import Dict, List, Tuple, Optional
from src.models.legal_analysis import (
    LegalAnalysis, PermissionAnalysis, PermissionLevel
)
from src.classifier.category_buckets import CATEGORY_DEFINITIONS, WebsiteCategory, PARAM_TO_CATEGORY_FIELD


# Secondary-pass prompt used to semantically resolve UNCERTAIN permission calls.
# Forces the LLM to reason about commercial/revenue-generating intent, then
# emit a strict Y/N answer grounded in the supplied reasoning + excerpts.
UNCERTAIN_RESOLUTION_PROMPT = """You are resolving an UNCERTAIN permission classification.

The 4 permission parameters and their definitions:
1. scrap - whether automated crawling/scraping is allowed
2. store - whether storing/caching data is allowed
3. display_for_free - whether displaying content publicly for free is allowed
4. display_for_commercial - whether displaying content behind a paywall / for commercial purposes is allowed

For the parameter below, the first-pass LLM could not decide between allowed and not_allowed.
Your job: read the reasoning and relevant excerpts, then determine whether the underlying
content appears commercial or revenue-generating (e.g., subscription, paywall, ads,
"Subscription", "paid access", "resale", "API fees", "premium tier"). Output ONLY
one of:
  - Y  (treat as ALLOWED, the document language supports it or is permissive)
  - N  (treat as NOT_ALLOWED, the document language restricts it)

You MUST also explain WHY in a single sentence starting with "Because".

Output format (strict JSON, no markdown, no extra prose):
{{"decision": "Y"|"N", "reasoning": "Because ..."}}
"""


def _build_llm_client(api_key: str, base_url: str):
    """Create MiniMax LLM client."""
    import httpx
    class MiniMaxClient:
        def __init__(self, api_key, base_url):
            self.api_key = api_key
            self.base_url = base_url
            self.http_client = httpx.Client(timeout=120.0)

        def chat(self, messages, model="MiniMax-Text-01", max_tokens=4096, temperature=0):
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
            response = self.http_client.post(
                f"{self.base_url}/text/chatcompletion_v2",
                headers=headers, json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    return MiniMaxClient(api_key, base_url)


def _build_gemini_client(api_key: str, model: str):
    """Create Gemini LLM client."""
    import httpx
    class GeminiClient:
        def __init__(self, api_key, model):
            self.api_key = api_key
            self.model = model
            self.base_url = "https://generativelanguage.googleapis.com/v1beta"
            self.http_client = httpx.Client(timeout=120.0)

        def chat(self, messages, model=None, max_tokens=4096):
            model = model or self.model
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
            url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
            payload = {
                "contents": contents,
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.0}
            }
            if system_instruction:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
            response = self.http_client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
    return GeminiClient(api_key, model)


PERMISSION_PARAMS = [
    "scrap",
    "store",
    "display_for_free",
    "display_for_commercial",
]

SYSTEM_PROMPT = """
Given the full text of a website's legal document, analyze it and output a structured JSON object describing permissions for 4 parameters.

The 4 parameters to analyze:
1. scrap - whether automated crawling/scraping is allowed
2. store - whether storing/caching data is allowed
3. display_for_free - whether displaying content publicly for free is allowed
4. display_for_commercial - whether displaying content behind a paywall or for commercial / revenue-generating purposes is allowed

For each parameter, determine: allowed, not_allowed, or uncertain

Also extract:
- reference_urls: List of URLs mentioned in the document (e.g., API docs, developer terms, data licensing pages)
- unique_findings: Special clauses like "API Access", "Non-Commercial Restriction", "Attribution Required", "DMCA Protected", "Open Access", "Creative Commons"

Output ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{
  "scrap": {"permission": "allowed|not_allowed|uncertain", "reasoning": "...", "reference_urls": [], "relevant_excerpts": [{"text": "exact quote from document", "source": "https://actual/url/path"}]},
  "store": {"permission": "...", "reasoning": "...", "reference_urls": [], "relevant_excerpts": []},
  "display_for_free": {"permission": "...", "reasoning": "...", "reference_urls": [], "relevant_excerpts": []},
  "display_for_commercial": {"permission": "...", "reasoning": "...", "reference_urls": [], "relevant_excerpts": []},
  "reference_urls": ["https://..."],
  "unique_findings": ["Found indicators of ...", ...]
}

Key rules:
- For each relevant_excerpt, you MUST use the ACTUAL URL of the source document as the "source" field (e.g., "https://example.com/terms", "https://example.com/privacy", "https://example.com/robots.txt")
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

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://api.minimax.chat/v1",
        provider: str = "minimax",
        gemini_api_key: str = None,
        gemini_model: str = "gemini-2.5-flash"
    ):
        self.llm_client = None
        if provider == "gemini" and gemini_api_key:
            self.llm_client = _build_gemini_client(gemini_api_key, gemini_model)
        elif api_key:
            self.llm_client = _build_llm_client(api_key, base_url)

    def classify_permissions(
        self,
        text: str,
        website_url: str,
        website_domain: str,
        robots_txt: str = "",
        document_urls: Dict[str, str] = None
    ) -> LegalAnalysis:
        """
        Classify permissions using LLM if available, otherwise fall back to heuristic.
        """
        if document_urls is None:
            document_urls = {}
        if self.llm_client:
            return self._classify_with_llm(text, website_url, website_domain, robots_txt, document_urls)
        else:
            return self._classify_heuristic(text, website_url, website_domain, robots_txt)

    def _classify_with_llm(
        self,
        text: str,
        website_url: str,
        website_domain: str,
        robots_txt: str,
        document_urls: Dict[str, str]
    ) -> LegalAnalysis:
        """
        Use LLM to analyze the legal document with full semantic understanding.
        """
        # Truncate text to avoid token limits (first 8000 chars is usually enough for terms)
        text_to_analyze = text[:8000]

        # Format document URLs for the prompt
        doc_urls_str = "\n".join([f"- {doc_type}: {url}" for doc_type, url in document_urls.items()])

        user_prompt = f"""Website URL: {website_url}
Website domain: {website_domain}

Source document URLs:
{doc_urls_str}

Legal document text:
{text_to_analyze}"""

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
                raw_perm = param_data.get("permission", "uncertain")

                # Map LLM's varied permission strings to enum values
                perm_map = {
                    "allowed": "allowed",
                    "not_allowed": "not_allowed",
                    "not permitted": "not_allowed",
                    "prohibited": "not_allowed",
                    "restricted": "not_allowed",
                    "uncertain": "uncertain",
                    "unclear": "uncertain",
                    "not applicable": "not_applicable",
                    "n/a": "not_applicable",
                    "allowed_with_attribution": "allowed",
                    "allowed_without_attribution": "allowed",
                    "not_allowed_with_attribution": "not_allowed",
                }
                perm_level_str = perm_map.get(raw_perm.lower() if isinstance(raw_perm, str) else raw_perm, "uncertain")

                # Override perm_level based on reasoning keywords to fix LLM misclassifications
                # The LLM may return "uncertain" or an incorrect label, but the reasoning text
                # often contains explicit keywords that reveal the true permission status
                reasoning_lower = param_data.get("reasoning", "").lower()

                # Keywords that indicate prohibition/denial
                forbid_kw = [
                    "explicitly prohibit", "expressly prohibit", "prohibited", "not allowed",
                    "strictly forbidden", "is not permitted", "are not permitted", "is prohibited",
                    "are prohibited", "prohibits", "forbidden", "disallowed",
                    "without our express prior permission", "without prior permission",
                    "expressly reserved", "all rights reserved", "no part of the service"
                ]
                # Keywords that indicate explicit permission/allowance
                allow_kw = [
                    "explicitly permit", "expressly permit", "expressly allowed", "explicitly allowed",
                    "you may", "you can", "is permitted", "are permitted", "is allowed", "are allowed",
                    "grants the right", "has the right to", "hereby grants", "freely use",
                    "no restriction", "without restriction", "open license"
                ]

                # Always check reasoning keywords — they override whatever the LLM returned
                # because the reasoning is the LLM's own explanation, which is more reliable
                has_forbid = any(kw in reasoning_lower for kw in forbid_kw)
                has_allow = any(kw in reasoning_lower for kw in allow_kw)

                if has_forbid and not has_allow:
                    perm_level_str = "not_allowed"
                elif has_allow and not has_forbid:
                    perm_level_str = "allowed"
                elif has_forbid and has_allow:
                    # Both present — stay with LLM's original classification
                    pass
                # If neither keyword present and LLM already gave a non-uncertain answer, keep it

                # Parse excerpts: [{"text": "...", "source": "actual_url"}]
                raw_excerpts = param_data.get("relevant_excerpts", [])
                excerpt_list = []
                for ex in raw_excerpts:
                    if isinstance(ex, dict):
                        text = ex.get("text", "")
                        src = ex.get("source", "")
                        if text:
                            excerpt_list.append({"text": text, "source": src})
                    elif isinstance(ex, str) and ex:
                        excerpt_list.append({"text": ex, "source": ""})

                # Resolve uncertain buckets (6/7/8 in the PM spec — i.e. the params
                # that frequently come back as ?) via a second LLM call that asks
                # the model to reason about commercial / revenue-generating intent.
                # Run AFTER the keyword-override pass so keyword-driven definite
                # answers (Y/N) are not re-litigated.
                #
                # In the 4-perm schema (scrap / store / display_for_free /
                # display_for_commercial), the commercial-intent semantic judgment
                # is most decisive for `display_for_commercial` — but the same
                # pass runs uniformly across all 4 axes so UNCERTAIN on any axis
                # is given one chance to flip to a definite Y/N.
                final_perm = PermissionLevel(perm_level_str)
                final_reasoning = param_data.get("reasoning", "No reasoning provided")

                if final_perm == PermissionLevel.UNCERTAIN:
                    # Direct rule match for definite buckets (1-4): if a non-uncertain
                    # value already came out of the first pass, the keyword override
                    # pass would have caught it. Reaching here means we need LLM
                    # semantic judgment to resolve the ? -> Y/N question.
                    resolved_perm, resolved_reasoning = self._resolve_uncertain_param(
                        param=param,
                        reasoning=final_reasoning,
                        excerpts=excerpt_list,
                    )

                    # Re-apply reasoning-keyword override on the resolved reasoning
                    # so a "prohibited" / "you may" appearing in the new "Because ..."
                    # justification still wins over the LLM's Y/N vote.
                    resolved_lower = resolved_reasoning.lower()
                    resolved_has_forbid = any(kw in resolved_lower for kw in forbid_kw)
                    resolved_has_allow = any(kw in resolved_lower for kw in allow_kw)

                    if resolved_has_forbid and not resolved_has_allow:
                        final_perm = PermissionLevel.NOT_ALLOWED
                    elif resolved_has_allow and not resolved_has_forbid:
                        final_perm = PermissionLevel.ALLOWED
                    elif resolved_perm == PermissionLevel.UNCERTAIN:
                        # Resolution pass could not decide; keep uncertain.
                        final_perm = PermissionLevel.UNCERTAIN
                    else:
                        final_perm = resolved_perm

                    final_reasoning = resolved_reasoning

                permission = PermissionAnalysis(
                    parameter_name=param,
                    permission=final_perm,
                    reasoning=final_reasoning,
                    relevant_excerpts=excerpt_list,
                    source_documents=[],  # No longer used, kept for backward compat
                    confidence_score=0.95 if final_perm not in (PermissionLevel.UNCERTAIN, PermissionLevel.NOT_APPLICABLE) else 0.5
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

    def _resolve_uncertain_param(
        self,
        param: str,
        reasoning: str,
        excerpts: List[Dict[str, str]]
    ) -> Tuple[PermissionLevel, str]:
        """
        Secondary LLM pass used when a parameter is still marked UNCERTAIN after
        the first classification pass. The LLM is asked: does the content seem
        commercial / revenue-generating? It must return Y or N with a one-sentence
        justification, which we map to ALLOWED / NOT_ALLOWED.

        Uses temperature=0 for deterministic output. If the LLM is unavailable,
        or the response is unparseable, the param stays UNCERTAIN.
        """
        if not self.llm_client:
            return PermissionLevel.UNCERTAIN, reasoning

        # Build a compact evidence packet from the first-pass reasoning + excerpts
        excerpt_text = ""
        for ex in (excerpts or [])[:5]:
            if isinstance(ex, dict):
                excerpt_text += f"\n- {ex.get('text', '')}"
            elif isinstance(ex, str):
                excerpt_text += f"\n- {ex}"

        user_msg = (
            f"Parameter: {param}\n\n"
            f"First-pass reasoning:\n{reasoning}\n\n"
            f"Relevant excerpts:{excerpt_text}"
        )

        try:
            response = self.llm_client.chat(
                [
                    {"role": "system", "content": UNCERTAIN_RESOLUTION_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=256,
                temperature=0,
            )

            data = self._parse_llm_response(response)
            decision = (data.get("decision") or "").strip().upper()
            why = (data.get("reasoning") or "").strip()

            if decision == "Y":
                resolved = PermissionLevel.ALLOWED
            elif decision == "N":
                resolved = PermissionLevel.NOT_ALLOWED
            else:
                # Unparseable — leave as UNCERTAIN, keep original reasoning
                return PermissionLevel.UNCERTAIN, reasoning

            # Compose a richer reasoning that preserves the original LLM text
            # plus the commercial-intent signal that flipped the call.
            combined = f"{reasoning} [Uncertain-resolution: '{why}' -> {decision}]"
            return resolved, combined

        except Exception:
            return PermissionLevel.UNCERTAIN, reasoning

    def _apply_direct_rule_match(
        self,
        param: str,
        perm_level: PermissionLevel,
        reasoning: str
    ) -> Tuple[PermissionLevel, str]:
        """
        Apply direct rule matching for definite Y/N cases (buckets 1-4).
        Buckets 1-4 correspond to permission params that have a clear allowed /
        not_allowed determination in the legal text. The existing keyword
        override already handles these in `_classify_with_llm`. This helper
        exists as a single, auditable chokepoint so future rule additions
        (e.g. param-specific allow/forbid lists) can be added in one place.

        The current implementation is a thin pass-through: it trusts the
        keyword-override pass that already runs in `_classify_with_llm` and
        only re-classifies UNCERTAIN results when no override fired. The
        actual uncertain-resolution (which may flip ? -> Y/N) lives in
        `_resolve_uncertain_param`.
        """
        # Definite buckets (1-4): the first-pass result is either Y or N.
        # If the first pass already produced a definite answer, keep it.
        if perm_level != PermissionLevel.UNCERTAIN:
            return perm_level, reasoning

        # For uncertain cases the caller will route through
        # `_resolve_uncertain_param`. We don't double-resolve here.
        return perm_level, reasoning

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
            "scrap": {
                "forbidden": ["prohibit scraping", "scraping prohibited", "no automated", "no scraping"],
                "allowed": ["scrap", "crawl", "bot", "automated access"]
            },
            "store": {
                "forbidden": ["cannot store", "no storage", "no caching", "may not cache"],
                "allowed": ["store", "cache", "retain", "archive"]
            },
            "display_for_free": {
                "forbidden": ["cannot display", "display prohibited", "no public display"],
                "allowed": ["display", "show", "view", "public", "free use"]
            },
            "display_for_commercial": {
                "forbidden": ["no paid access", "no commercial use", "non-commercial only", "commercial use prohibited"],
                "allowed": ["subscription", "paid", "premium", "commercial", "paywall", "monetize"]
            },
        }

        for param, rules in param_rules.items():
            forbid_count = sum(1 for kw in rules["forbidden"] if kw in text_lower)
            allow_count = sum(1 for kw in rules["allowed"] if kw in text_lower)

            # Negation context check for display params (where "no X" is a common prohibition)
            if param in ["display_for_free", "display_for_commercial", "store"]:
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
        Determine bucket by profile matching against the 4 permission axes
        (scrap, store, display_for_free, display_for_commercial).

        For each bucket, for each axis where the bucket profile specifies
        a definite value (True/False):
          +1 if the actual permission matches the expectation
          -1 if it contradicts the expectation
          0  if the actual permission is UNCERTAIN

        Axes where the bucket profile is None (wildcard, i.e. buckets 6/7/8
        trailing axes) contribute 0 — the LLM semantic-judgment pass that
        resolves ? -> Y/N is what eventually determines whether those wildcards
        match buckets 1-4 (which have definite profiles) or fall through to
        6/7/8 (which leave those axes open).

        The bucket with the highest score wins. Tie-break rule: when two or
        more buckets tie on score, prefer the bucket with MORE wildcard axes
        (i.e. the more "uncertain" / less committed bucket). This means
        BUCKET_6/7/8 naturally surface when the LLM could not produce a
        definite Y/N on the trailing axes, while still letting a fully
        definite bucket (BUCKET_1-4) win when it scores strictly higher.
        """
        best_bucket: Optional[WebsiteCategory] = None
        best_score = -999
        best_wildcards = -1  # higher = more uncertain bucket wins on tie

        for bucket, definition in CATEGORY_DEFINITIONS.items():
            score = 0
            wildcards = 0
            for param_name, category_field in PARAM_TO_CATEGORY_FIELD.items():
                expected_allowed = getattr(definition, category_field)
                # Wildcard axis: bucket is permissive about this dimension,
                # so neither match nor mismatch is recorded. The LLM
                # semantic-judgment pass (which turns ? into Y/N) is what
                # narrows the bucket fit.
                if expected_allowed is None:
                    wildcards += 1
                    continue
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

            # Strictly higher score wins. On score tie, more wildcards wins
            # (so uncertain buckets surface when the LLM left axes open).
            if (score > best_score) or (score == best_score and wildcards > best_wildcards):
                best_score = score
                best_bucket = bucket
                best_wildcards = wildcards

        # Default safety net: if nothing matched (shouldn't normally happen),
        # fall back to BUCKET_8 (Fully Uncertain).
        if best_bucket is None:
            best_bucket = WebsiteCategory.BUCKET_8
            best_score = 0

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