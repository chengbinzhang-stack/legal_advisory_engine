"""Scraper for Privacy Policy pages."""
import httpx
from bs4 import BeautifulSoup
from src.models.website_data import ScrapedDocument
from src.scraper.base_scraper import BaseScraper, is_spa_shell
from datetime import datetime


class PrivacyScraper(BaseScraper):
    """Scraper for Privacy Policy pages."""

    COMMON_PATHS = [
        "/privacy-policy",
        "/privacy",
        "/legal/privacy",
        "/legal/privacy-policy",
        "/data-protection",
        "/personal-information",
        "/privacy-statement",
        "/en/privacy-policy",
        "/about/privacy",
        "/about/legal/privacy",
        "/legal/privacy-notice",
        "/privacy-notice",
        "/privacy-statement",
        "/cookies-policy",
        "/cookie-policy",
        "/data-privacy",
        "/data-privacy-policy",
        "/personal-data",
        "/information-collection",
        "/privacy-and-cookies",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def scrape(self, url: str) -> ScrapedDocument:
        if self._is_likely_privacy_url(url):
            result = self._try_scrape(url)
            if result.success:
                return result
        base_url = self._get_base_url(url)
        sitemap_result = self._try_sitemap(base_url, ["privacy", "data", "cookie", "personal"])
        if sitemap_result:
            return sitemap_result
        homepage_links = self._find_legal_links(base_url, ["privacy", "data-protection", "cookie"])
        for link in homepage_links[:5]:
            result = self._try_scrape(link)
            if result.success:
                return result
        for path in self.COMMON_PATHS:
            privacy_url = base_url.rstrip("/") + path
            result = self._try_scrape(privacy_url)
            if result.success:
                return result
        return ScrapedDocument(
            document_type="privacy_policy", url=url, raw_content="",
            scraped_at=datetime.now(), success=False,
            error_message="Privacy policy page not found"
        )

    def _try_sitemap(self, base_url: str, keywords: list) -> ScrapedDocument:
        sitemap_url = base_url.rstrip("/") + "/sitemap.xml"
        try:
            response = httpx.get(sitemap_url, headers=self._build_headers(), timeout=self.timeout, follow_redirects=True)
            if response.status_code != 200:
                return None
            soup = BeautifulSoup(response.text, "html.parser")
            locs = soup.find_all("loc")
            privacy_locs = [loc.get_text().strip() for loc in locs
                           if any(kw in loc.get_text().lower() for kw in keywords)]
            for privacy_url in privacy_locs[:3]:
                result = self._try_scrape(privacy_url)
                if result.success:
                    return result
        except Exception:
            pass
        return None

    def _find_legal_links(self, base_url: str, keywords: list) -> list:
        links = []
        try:
            response = httpx.get(base_url, headers=self._build_headers(), timeout=self.timeout, follow_redirects=True)
            if response.status_code != 200:
                return links
            soup = BeautifulSoup(response.text, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if not href.startswith("http"):
                    href = base_url.rstrip("/") + ("/" if not href.startswith("/") else "") + href
                if any(kw in href.lower() for kw in keywords) and href not in links:
                    links.append(href)
        except Exception:
            pass
        return links

    def _try_scrape(self, url: str) -> ScrapedDocument:
        """Try httpx first, fallback to Browserless if SPA detected."""
        try:
            response = httpx.get(url, headers=self._build_headers(), timeout=self.timeout, follow_redirects=True)
            if response.status_code != 200:
                return ScrapedDocument(
                    document_type="privacy_policy", url=url, raw_content="",
                    scraped_at=datetime.now(), success=False,
                    error_message=f"HTTP {response.status_code}"
                )

            html_content = response.text

            # Check if it's an SPA shell - if so, try Browserless
            if is_spa_shell(html_content) and self.browserless_api_key:
                browserless_content, status = self._fetch_with_browserless(url)
                if status == 200 and len(browserless_content) > 500:
                    soup = BeautifulSoup(browserless_content, "html.parser")
                    text = self._extract_text(soup)
                    return ScrapedDocument(
                        document_type="privacy_policy", url=url,
                        raw_content=text, scraped_at=datetime.now(), success=True
                    )

            # Normal httpx path
            text = self._extract_text(html_content)
            return ScrapedDocument(
                document_type="privacy_policy", url=str(response.url),
                raw_content=text, scraped_at=datetime.now(), success=True
            )

        except Exception as e:
            # If httpx failed and we have Browserless, try it
            if self.browserless_api_key:
                browserless_content, status = self._fetch_with_browserless(url)
                if status == 200:
                    soup = BeautifulSoup(browserless_content, "html.parser")
                    text = self._extract_text(soup)
                    return ScrapedDocument(
                        document_type="privacy_policy", url=url,
                        raw_content=text, scraped_at=datetime.now(), success=True
                    )
            return ScrapedDocument(
                document_type="privacy_policy", url=url, raw_content="",
                scraped_at=datetime.now(), success=False, error_message=str(e)
            )

    def _extract_text(self, html_content) -> str:
        """Extract text from HTML string or BeautifulSoup object."""
        if isinstance(html_content, BeautifulSoup):
            soup = html_content
        else:
            soup = BeautifulSoup(html_content, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()
        return soup.get_text(separator="\n", strip=True)

    def _is_likely_privacy_url(self, url: str) -> bool:
        url_lower = url.lower()
        privacy_keywords = ["privacy", "policy", "data-protection", "personal", "cookie"]
        return any(kw in url_lower for kw in privacy_keywords)

    def _get_base_url(self, url: str) -> str:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
