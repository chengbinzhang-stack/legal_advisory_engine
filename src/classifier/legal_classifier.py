"""Classifies legal documents for the 7 parameters."""
from typing import Dict, List, Tuple
from src.models.legal_analysis import (
    LegalAnalysis, PermissionAnalysis, PermissionLevel, WebsiteCategory
)
from src.classifier.category_buckets import (
    CATEGORY_DEFINITIONS, WebsiteCategory
)

PARAM_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "scraping": {
        "allowed": ["scrap", "crawl", "automated", "bot", "spider", "extract data", "web scrap"],
        "forbidden": ["prohibit scraping", "no automated", "against robots.txt", "cannot scrape", "scraping prohibited"]
    },
    "manual_collection": {
        "allowed": ["manual", "copy", "read", "browse", "view"],
        "forbidden": ["cannot copy", "prohibit copying", "copying prohibited"]
    },
    "storing": {
        "allowed": ["store", "cache", "retain", "save", "storage"],
        "forbidden": ["cannot store", "no storage", "do not retain", "storage prohibited"]
    },
    "free_display": {
        "allowed": ["display", "show", "view", "accessible", "public"],
        "forbidden": ["cannot display", "no display", "display prohibited"]
    },
    "subscription_display": {
        "allowed": ["subscription", "paid", "premium", "member", "paid access"],
        "forbidden": ["only free", "no paid access", "subscription not allowed"]
    },
    "free_redistribute": {
        "allowed": ["redistribute", "share", "distribute", "publish", "republish"],
        "forbidden": ["cannot redistribute", "no redistribution", "cannot share", "redistribution prohibited"]
    },
    "subscription_redistribute": {
        "allowed": ["redistribute", "share", "distribute", "sell"],
        "forbidden": ["cannot redistribute", "no redistribution", "resale not allowed"]
    },
}
class LegalClassifier:
    def __init__(self):
        self.param_keywords = PARAM_KEYWORDS

    def classify_permissions(
        self,
        text: str,
        website_url: str,
        website_domain: str,
        robots_txt: str = ""
    ) -> LegalAnalysis:
        permissions = {}
        for param_name in PARAM_KEYWORDS.keys():
            if param_name == "scraping":
                permission = self._classify_from_robots(robots_txt)
            else:
                permission = self._classify_single_parameter(text, param_name)
            permissions[param_name] = permission
        category, category_reasoning = self._determine_category(permissions)
        unique_findings = self._extract_unique_findings(text, permissions)
        summary_text = self._generate_summary(website_domain, category, permissions)
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

    def _classify_from_robots(self, robots_txt: str) -> PermissionAnalysis:
        """Classify scraping permission from robots.txt content."""
        if not robots_txt:
            return PermissionAnalysis(
                parameter_name="scraping",
                permission=PermissionLevel.UNCERTAIN,
                reasoning="No robots.txt found or empty",
                confidence_score=0.0
            )
        
        robots_lower = robots_txt.lower()
        # Check for complete disallow
        disallow_all = False
        allow_all = False
        
        lines = robots_lower.split(chr(10))
        has_user_agent_star = False
        found_disallow = []
        found_allow = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('user-agent:'):
                ua = line.split(':', 1)[1].strip()
                if ua == '*':
                    has_user_agent_star = True
            elif line.startswith('disallow:'):
                path = line.split(':', 1)[1].strip()
                if has_user_agent_star:
                    found_disallow.append(path)
            elif line.startswith('allow:'):
                path = line.split(':', 1)[1].strip()
                if has_user_agent_star:
                    found_allow.append(path)
        
        # Determine if basically disallowed or allowed
        for disallow in found_disallow:
            if disallow == '/' or disallow == '':
                disallow_all = True
            if disallow == '/':
                disallow_all = True
        
        for allow in found_allow:
            if allow == '/' and found_disallow == ['/']:
                allow_all = False  # allow / but also disallow /, so blocked
            elif allow == '/':
                allow_all = True
        
        if disallow_all:
            return PermissionAnalysis(
                parameter_name="scraping",
                permission=PermissionLevel.NOT_ALLOWED,
                reasoning="robots.txt disallows all crawlers (Disallow: /)",
                confidence_score=1.0
            )
        elif found_disallow:
            return PermissionAnalysis(
                parameter_name="scraping",
                permission=PermissionLevel.ALLOWED,
                reasoning=f"robots.txt allows crawling with some restrictions: {found_disallow}",
                confidence_score=0.8
            )
        else:
            return PermissionAnalysis(
                parameter_name="scraping",
                permission=PermissionLevel.ALLOWED,
                reasoning="robots.txt exists with no disallow rules for *",
                confidence_score=0.9
            )

    def _classify_single_parameter(
        self,
        text: str,
        param_name: str
    ) -> PermissionAnalysis:
        text_lower = text.lower()
        keywords = self.param_keywords.get(param_name, {})
        allowed_keywords = keywords.get("allowed", [])
        forbidden_keywords = keywords.get("forbidden", [])
        allowed_count = sum(1 for kw in allowed_keywords if kw.lower() in text_lower)
        forbidden_count = sum(1 for kw in forbidden_keywords if kw.lower() in text_lower)
        excerpts = self._extract_excerpts(text, allowed_keywords + forbidden_keywords)
        if forbidden_count > allowed_count:
            permission = PermissionLevel.NOT_ALLOWED
            reasoning = f"Found {forbidden_count} prohibition indicators vs {allowed_count} permission indicators"
        elif allowed_count > 0:
            permission = PermissionLevel.ALLOWED
            reasoning = f"Found {allowed_count} permission indicators"
        else:
            permission = PermissionLevel.UNCERTAIN
            reasoning = "No explicit indicators found in document"
        total = allowed_count + forbidden_count
        confidence = max(allowed_count, forbidden_count) / max(total, 1) if total > 0 else 0.0
        return PermissionAnalysis(
            parameter_name=param_name,
            permission=permission,
            reasoning=reasoning,
            relevant_excerpts=excerpts,
            confidence_score=confidence
        )
    def _extract_excerpts(
        self,
        text: str,
        keywords: List[str],
        context_chars: int = 150,
        max_excerpts: int = 3
    ) -> List[str]:
        excerpts = []
        text_lower = text.lower()
        for keyword in keywords:
            keyword_lower = keyword.lower()
            start = 0
            while len(excerpts) < max_excerpts:
                idx = text_lower.find(keyword_lower, start)
                if idx == -1:
                    break
                context_start = max(0, idx - context_chars)
                context_end = min(len(text), idx + len(keyword) + context_chars)
                excerpt = text[context_start:context_end].strip()
                excerpts.append(f"...{excerpt}...")
                start = idx + len(keyword)
        return excerpts[:max_excerpts]

    def _determine_category(
        self,
        permissions: Dict[str, PermissionAnalysis]
    ) -> Tuple[WebsiteCategory, str]:
        bucket_scores = {
            WebsiteCategory.BUCKET_1: {"allowed": 0, "forbidden": 0},
            WebsiteCategory.BUCKET_2: {"allowed": 0, "forbidden": 0},
            WebsiteCategory.BUCKET_3: {"allowed": 0, "forbidden": 0},
            WebsiteCategory.BUCKET_4: {"allowed": 0, "forbidden": 0},
        }
        for param in ["scraping", "storing", "free_display", "free_redistribute"]:
            self._score_param(permissions, param, bucket_scores[WebsiteCategory.BUCKET_1])
        for param in ["scraping", "storing", "free_display"]:
            self._score_param(permissions, param, bucket_scores[WebsiteCategory.BUCKET_2])
        for param in ["scraping", "storing"]:
            self._score_param(permissions, param, bucket_scores[WebsiteCategory.BUCKET_3])
        self._score_param(permissions, "manual_collection", bucket_scores[WebsiteCategory.BUCKET_4])
        best_bucket = WebsiteCategory.BUCKET_4
        best_score = 0
        for bucket, scores in bucket_scores.items():
            total_score = scores["allowed"] - scores["forbidden"]
            if total_score > best_score:
                best_score = total_score
                best_bucket = bucket
        definitions = CATEGORY_DEFINITIONS[best_bucket]
        reasoning = f"Website matches {definitions.name} profile with score {best_score}"
        return best_bucket, reasoning
    def _score_param(
        self,
        permissions: Dict[str, PermissionAnalysis],
        param_name: str,
        bucket_score: Dict[str, int]
    ):
        if param_name not in permissions:
            return
        perm = permissions[param_name]
        if perm.permission == PermissionLevel.ALLOWED:
            bucket_score["allowed"] += 1
        elif perm.permission == PermissionLevel.NOT_ALLOWED:
            bucket_score["forbidden"] += 1

    def _extract_unique_findings(
        self,
        text: str,
        permissions: Dict[str, PermissionAnalysis]
    ) -> List[str]:
        findings = []
        text_lower = text.lower()
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
        permissions: Dict[str, PermissionAnalysis]
    ) -> str:
        definition = CATEGORY_DEFINITIONS[category]
        summary_parts = [
            f"Website: {website_domain}",
            f"Category: Bucket {category.value} - {definition.name}",
            "",
            "Permission Summary:",
        ]
        for param_name, perm in permissions.items():
            emoji = "YES" if perm.permission == PermissionLevel.ALLOWED else "NO" if perm.permission == PermissionLevel.NOT_ALLOWED else "?"
            summary_parts.append(f"  [{emoji}] {param_name}: {perm.permission.value}")
        return "\n".join(summary_parts)