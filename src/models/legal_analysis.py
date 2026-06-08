"""Data models for legal analysis results."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

class PermissionLevel(Enum):
    ALLOWED = "allowed"
    NOT_ALLOWED = "not_allowed"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"

class WebsiteCategory(Enum):
    BUCKET_1 = 1 # scrape, store, display, redistribute allowed
    BUCKET_2 = 2  # scrape, store, display allowed, no redistribute
    BUCKET_3 = 3  # scrape, store allowed, no display, no redistribute
    BUCKET_4 = 4  # manually collect data only

@dataclass
class PermissionAnalysis:
    """Analysis result for one of the 7 parameters."""
    parameter_name: str
    permission: PermissionLevel
    reasoning: str
    relevant_excerpts: List[str] = field(default_factory=list)
    confidence_score: float = 0.0

@dataclass
class LegalAnalysis:
    """Complete legal analysis for a website."""
    website_url: str
    website_domain: str
    category: WebsiteCategory
    category_reasoning: str
    permissions: Dict[str, PermissionAnalysis] = field(default_factory=dict)
    unique_findings: List[str] = field(default_factory=list)
    summary_text: str = ""
    raw_analysis: str = ""
