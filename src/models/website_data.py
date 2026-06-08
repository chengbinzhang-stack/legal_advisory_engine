"""Data models for website scraped data."""
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

@dataclass
class ScrapedDocument:
    """Represents a scraped legal document."""
    document_type: str  # "terms_of_use", "privacy_policy", "robots_txt"
    url: str
    raw_content: str
    scraped_at: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class WebsiteData:
    """Aggregated data for a single website."""
    url: str
    domain: str
    documents: List[ScrapedDocument] = field(default_factory=list)
    raw_text: str = ""
    processed_at: Optional[datetime] = None
