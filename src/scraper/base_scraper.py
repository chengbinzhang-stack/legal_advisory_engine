"""Abstract base class for all legal document scrapers."""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, TYPE_CHECKING
import httpx
from bs4 import BeautifulSoup
from src.models.website_data import ScrapedDocument
from datetime import datetime

if TYPE_CHECKING:
    from src.scraper.browserless_session import BrowserlessSessionManager


# SPA shell detection
MIN_CONTENT_LENGTH_FOR_SPA = 1000  # If content is smaller than this, likely an SPA shell


def is_spa_shell(html_content: str) -> bool:
    """Detect if HTML content is likely an SPA shell (JS-rendered page)."""
    if len(html_content) < MIN_CONTENT_LENGTH_FOR_SPA:
        return True

    # Common SPA framework markers
    spa_markers = [
        '<script type="module"',           # Vite/React/Vue
        'id="root"',                        # React typical
        'id="app"',                         # Vue typical
        'crossorigin',                      # Vite builds
        '/assets/',                         # Vite asset paths
        '__NEXT_DATA__',                    # Next.js
        '__NUXT__',                         # Nuxt.js
        '"react"',                          # React apps
        '"vue"',                            # Vue apps
    ]

    content_lower = html_content.lower()
    marker_count = sum(1 for marker in spa_markers if marker.lower() in content_lower)

    # If content is small AND has SPA markers, definitely SPA
    if len(html_content) < 2000 and marker_count >= 1:
        return True

    return False


class BaseScraper(ABC):
    """Abstract base class for all legal document scrapers."""

    def __init__(self, timeout: int = 30, user_agent: str = None,
                 browserless_api_key: str = None,
                 browserless_session: "BrowserlessSessionManager" = None):
        self.timeout = timeout
        self.user_agent = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        self.client = httpx.Client(timeout=timeout)
        self.browserless_api_key = browserless_api_key
        self.browserless_session = browserless_session

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

    def _fetch_with_browserless(self, url: str) -> Tuple[str, int]:
        """Fetch URL using Browserless API for JS-rendered content."""
        # Use session if available (more efficient), otherwise fall back to direct API
        if self.browserless_session:
            return self.browserless_session.fetch(url)

        if not self.browserless_api_key:
            return "", 503

        try:
            response = httpx.post(
                f"https://chrome.browserless.io/content?token={self.browserless_api_key}",
                json={
                    "url": url,
                    "gotoOptions": {
                        "waitUntil": "networkidle2"
                    }
                },
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            if response.status_code == 200:
                return response.text, 200
            return "", response.status_code
        except Exception:
            return "", 503

    @staticmethod
    def _extract_text(html_content) -> str:
        """Extract text from HTML string or BeautifulSoup object."""
        if isinstance(html_content, BeautifulSoup):
            soup = html_content
        else:
            soup = BeautifulSoup(html_content, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()
        return soup.get_text(separator="\n", strip=True)

    def __del__(self):
        """Close HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
