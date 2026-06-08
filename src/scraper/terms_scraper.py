"""Scraper for Terms of Use pages."""
from typing import List, Optional
import httpx
from bs4 import BeautifulSoup
from src.models.website_data import ScrapedDocument
from src.scraper.base_scraper import BaseScraper
from src.scraper.site_explorer import SiteExplorer
from datetime import datetime


class TermsScraper(BaseScraper):
    """Scraper for Terms of Use pages."""

    COMMON_PATHS = [
        "/terms-of-use",
        "/terms-of-service",
        "/terms",
        "/legal/terms",
        "/legal/terms-of-use",
        "/legal/terms-and-conditions",
        "/conditions",
        "/user-agreement",
        "/content/terms",
        "/terms-and-conditions",
        "/en/terms-of-use",
        "/en/terms-of-service",
        "/about/terms",
        "/about/legal/terms",
        "/site/terms",
        "/website-terms",
        "/legal-notice",
        "/legal/conditions",
        "/legal/docs/terms",
        "/about/legal",
        "/legal",
        "/statement",
        "/guidelines",
        "/rules",
        "/acceptable-use",
        "/use-policy",
    ]

    DOMAIN_SPECIFIC_PATHS = {
        "worldbank.org": [
            "/en/about/legal/terms-and-conditions",
            "/en/about/legal",
            "/content/website-terms-conditions",
        ],
        "parivahan.gov.in": [
            "/terms-conditions",
            "/disclaimer",
        ],
        "dataworld": [
            "/terms",
        ],
        "fred.stlouisfed.org": [
            "/legal",
        ],
    }

    MIN_CONTENT_LENGTH = 500

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def scrape(self, url: str) -> ScrapedDocument:
        # 1. If URL itself is a terms URL, scrape it directly
        if self._is_likely_terms_url(url):
            result = self._try_scrape(url)
            if result.success and len(result.raw_content) >= self.MIN_CONTENT_LENGTH:
                return result
            # If 404 or thin content, try common URL variations
            if not result.success or len(result.raw_content) < self.MIN_CONTENT_LENGTH:
                variations = self._url_variations(url)
                for var_url in variations:
                    var_result = self._try_scrape(var_url)
                    if var_result.success and len(var_result.raw_content) >= self.MIN_CONTENT_LENGTH:
                        return var_result

        base_url = self._get_base_url(url)
        domain = self._get_domain(url)

        # 2. Sitemap discovery - find best terms page from sitemap
        sitemap_result = self._try_sitemap(base_url, ["terms", "conditions", "agreement", "legal"])
        if sitemap_result and len(sitemap_result.raw_content) >= self.MIN_CONTENT_LENGTH:
            return sitemap_result

        # 3. Homepage link discovery - find legal links from homepage
        homepage_links = self._find_legal_links(base_url, ["terms", "conditions", "agreement"])
        for link in homepage_links[:5]:
            result = self._try_scrape(link)
            if result.success and len(result.raw_content) >= self.MIN_CONTENT_LENGTH:
                return result

        # 4. Domain-specific known paths (like worldbank.org legal pages)
        if domain in self.DOMAIN_SPECIFIC_PATHS:
            for path in self.DOMAIN_SPECIFIC_PATHS[domain]:
                terms_url = base_url.rstrip("/") + path
                result = self._try_scrape(terms_url)
                if result.success and len(result.raw_content) >= self.MIN_CONTENT_LENGTH:
                    return result

        # 5. Common fallback paths
        for path in self.COMMON_PATHS:
            terms_url = base_url.rstrip("/") + path
            result = self._try_scrape(terms_url)
            if result.success and len(result.raw_content) >= self.MIN_CONTENT_LENGTH:
                return result

        # 6. Site exploration - crawl the site to discover legal pages
        explorer = SiteExplorer(max_depth=2, max_links=30, timeout=self.timeout, user_agent=self.user_agent)
        discovery = explorer.discover_legal_pages(base_url)
        for terms_url in discovery.get("terms_of_use", [])[:10]:
            result = self._try_scrape(terms_url)
            if result.success and len(result.raw_content) >= self.MIN_CONTENT_LENGTH:
                return result
        # Also try other_legal pages (disclaimers, notices, etc.)
        for legal_url in discovery.get("other_legal", [])[:10]:
            result = self._try_scrape(legal_url)
            if result.success and len(result.raw_content) >= self.MIN_CONTENT_LENGTH:
                return result

        return ScrapedDocument(
            document_type="terms_of_use", url=url, raw_content="",
            scraped_at=datetime.now(), success=False,
            error_message="Terms of use page not found"
        )

    def _try_sitemap(self, base_url: str, keywords: List[str]) -> Optional[ScrapedDocument]:
        sitemap_url = base_url.rstrip("/") + "/sitemap.xml"
        best_result = None
        try:
            response = httpx.get(sitemap_url, headers=self._build_headers(), timeout=self.timeout, follow_redirects=True)
            if response.status_code != 200:
                return None
            soup = BeautifulSoup(response.text, "html.parser")
            locs = soup.find_all("loc")
            # Strict URL pattern matching - only accept terms-like URLs
            strict_patterns = [
                "terms-of-use", "terms-of-service", "terms_and_conditions",
                "terms-and-conditions", "terms_conditions", "conditions",
                "user-agreement", "legal-notice", "legal-notice",
                "acceptable-use", "use-policy", "site-terms",
            ]
            terms_locs = []
            for loc in locs:
                loc_url = loc.get_text().strip().lower()
                if any(pat in loc_url for pat in strict_patterns):
                    terms_locs.append(loc.get_text().strip())
            # Fall back to keyword match if no strict patterns found
            if not terms_locs:
                terms_locs = [loc.get_text().strip() for loc in locs
                              if any(kw in loc.get_text().lower() for kw in keywords)]
            for terms_url in terms_locs[:10]:
                result = self._try_scrape(terms_url)
                if result.success and len(result.raw_content) >= self.MIN_CONTENT_LENGTH:
                    if best_result is None or len(result.raw_content) > len(best_result.raw_content):
                        best_result = result
        except Exception:
            pass
        return best_result

    def _find_legal_links(self, base_url: str, keywords: List[str]) -> List[str]:
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
                href_lower = href.lower()
                # Strict URL pattern matching
                strict_patterns = [
                    "terms-of-use", "terms-of-service", "terms_and_conditions",
                    "terms-and-conditions", "terms_conditions", "conditions",
                    "user-agreement", "legal-notice", "acceptable-use",
                    "use-policy", "site-terms", "legal/terms",
                ]
                matched = any(pat in href_lower for pat in strict_patterns)
                if not matched:
                    matched = any(kw in href_lower for kw in keywords) and any(kw in href_lower for kw in ["terms", "conditions", "agreement"])
                if matched and href not in links:
                    links.append(href)
        except Exception:
            pass
        return links

    def _try_scrape(self, url: str) -> ScrapedDocument:
        try:
            response = httpx.get(url, headers=self._build_headers(), timeout=self.timeout, follow_redirects=True)
            if response.status_code == 200:
                content = self._extract_text(response.text)
                return ScrapedDocument(document_type="terms_of_use", url=str(response.url),
                    raw_content=content, scraped_at=datetime.now(), success=True)
            return ScrapedDocument(document_type="terms_of_use", url=url, raw_content="",
                scraped_at=datetime.now(), success=False, error_message=f"HTTP {response.status_code}")
        except Exception as e:
            return ScrapedDocument(document_type="terms_of_use", url=url, raw_content="",
                scraped_at=datetime.now(), success=False, error_message=str(e))

    def _extract_text(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()
        return soup.get_text(separator="\n", strip=True)

    def _is_likely_terms_url(self, url: str) -> bool:
        url_lower = url.lower()
        terms_keywords = ["terms", "conditions", "agreement", "legal", "legal-notice", "notice", "use-policy", "acceptable"]
        return any(kw in url_lower for kw in terms_keywords)

    def _get_base_url(self, url: str) -> str:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _get_domain(self, url: str) -> str:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc

    def _url_variations(self, url: str) -> list:
        """Generate URL variations to try when the original URL fails."""
        variations = []
        url_lower = url.lower()

        # Common plural/singular swaps: dataset/datasets, term/terms, etc.
        import re
        # Try adding/removing 's' at the end of meaningful path segments
        # e.g. .../terms-of-use-for-dataset -> .../terms-of-use-for-datasets
        # e.g. .../terms-of-use-for-datasets -> .../terms-of-use-for-dataset
        for singular, plural in [
            ("dataset", "datasets"),
            ("term", "terms"),
            ("condition", "conditions"),
            ("policy", "policies"),
        ]:
            if singular in url_lower and plural not in url_lower:
                var = url_lower.replace(singular, plural, 1)
                variations.append(var)
            elif plural in url_lower and singular not in url_lower:
                var = url_lower.replace(plural, singular, 1)
                variations.append(var)

        # Try appending/removing common suffixes
        if not url_lower.endswith("/"):
            variations.append(url.rstrip("/") + "/")
        if url_lower.endswith("/"):
            variations.append(url.rstrip("/"))

        # Try with/without trailing slash variations already covered above
        return variations