"""Abstract base class for all legal document scrapers."""
from abc import ABC, abstractmethod
from typing import Optional
import httpx
from src.models.website_data import ScrapedDocument
from datetime import datetime

class BaseScraper(ABC):
    """Abstract base class for all legal document scrapers."""

    def __init__(self, timeout: int = 30, user_agent: str = None):
        self.timeout = timeout
        self.user_agent = user_agent or "LegalAdvisoryBot/1.0 (Research Purpose)"
        self.client = httpx.Client(timeout=timeout)

    @abstractmethod
    def scrape(self, url: str) -> ScrapedDocument:
        """Scrape a document from the given URL."""
        pass

    def _build_headers(self) -> dict:
        """Build request headers."""
        return {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    def _is_accessible(self, status_code: int) -> bool:
        """Check if HTTP status code indicates accessibility."""
        return 200 <= status_code < 400

    def _fetch(self, url: str) -> tuple[str, int]:
        """Fetch URL content and return (content, status_code)."""
        response = self.client.get(url, headers=self._build_headers())
        return response.text, response.status_code

    def __del__(self):
        """Close HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
