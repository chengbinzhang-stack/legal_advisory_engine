"""Scraper for robots.txt files."""
import httpx
from src.models.website_data import ScrapedDocument
from src.scraper.base_scraper import BaseScraper
from datetime import datetime

class RobotsScraper(BaseScraper):
    """Scraper for robots.txt files."""

    def scrape(self, url: str) -> ScrapedDocument:
        """Scrape robots.txt from a website."""
        robots_url = self._get_robots_url(url)
        return self._try_scrape(robots_url)

    def _get_robots_url(self, url: str) -> str:
        """Get robots.txt URL from base URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    def _try_scrape(self, url: str) -> ScrapedDocument:
        """Try to scrape robots.txt."""
        try:
            response = httpx.get(url, headers=self._build_headers(), timeout=self.timeout)
            if response.status_code == 200:
                return ScrapedDocument(
                    document_type="robots_txt",
                    url=str(response.url),
                    raw_content=response.text,
                    scraped_at=datetime.now(),
                    success=True
                )
            elif response.status_code == 404:
                return ScrapedDocument(
                    document_type="robots_txt",
                    url=url,
                    raw_content="",
                    scraped_at=datetime.now(),
                    success=False,
                    error_message="robots.txt not found (404)"
                )
            else:
                return ScrapedDocument(
                    document_type="robots_txt",
                    url=url,
                    raw_content="",
                    scraped_at=datetime.now(),
                    success=False,
                    error_message=f"HTTP {response.status_code}"
                )
        except Exception as e:
            return ScrapedDocument(
                document_type="robots_txt",
                url=url,
                raw_content="",
                scraped_at=datetime.now(),
                success=False,
                error_message=str(e)
            )
