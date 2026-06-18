"""Site explorer for discovering legal document pages by crawling site navigation."""
from typing import List, Optional, Dict, Set
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup
from src.scraper.base_scraper import BaseScraper, is_spa_shell


class SiteExplorer(BaseScraper):
    """
    Explore a website to discover legal document pages.

    Starting from a homepage, prioritizes links whose URL or text
    contains legal keywords (terms, privacy, about, legal, etc.)
    to quickly find the relevant pages without exhaustive crawling.
    """

    LEGAL_URL_PATTERNS = [
        "terms", "condition", "agreement", "legal", "privacy",
        "policy", "notice", "disclaimer", "guideline", "rule",
        "use-policy", "acceptable-use", "user-agreement", "statement",
        "permission", "data-policy", "intellectual-property",
    ]

    LEGAL_LINK_TEXT_KEYWORDS = [
        "terms", "conditions", "privacy", "legal", "about",
        "disclaimer", "notice", "policy", "agreement", "permission",
        "data use", "data policy", "intellectual", "copyright",
    ]

    def __init__(self, max_depth: int = 2, max_links: int = 30, *args, **kwargs):
        """
        Initialize SiteExplorer.

        Args:
            max_depth: Maximum crawl depth from starting URL
            max_links: Maximum number of candidate links to check per page
        """
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth
        self.max_links = max_links
        self._visited: Set[str] = set()
        self._root_domain: str = ""

    def discover_legal_pages(self, start_url: str) -> Dict[str, List[str]]:
        """
        Discover legal-related pages starting from a URL.

        Returns a dict mapping category to list of discovered URLs.
        """
        self._root_domain = self._get_root_domain(start_url)
        results = {
            "terms_of_use": [],
            "privacy_policy": [],
            "other_legal": [],
            "about_pages": [],
        }

        self._visited.clear()
        self._crawl(start_url, depth=0, results=results)

        for key in results:
            results[key] = list(set(results[key]))

        return results

    def _crawl(self, url: str, depth: int, results: Dict[str, List[str]]) -> None:
        """Recursively crawl URL to find legal pages."""
        if depth > self.max_depth:
            return

        normalized = self._normalize_url(url)
        if normalized in self._visited:
            return
        self._visited.add(normalized)

        try:
            html_content = ""
            final_url = url

            # Try httpx first
            response = httpx.get(url, headers=self._build_headers(),
                                timeout=self.timeout, follow_redirects=True)
            if response.status_code == 200:
                html_content = response.text
                final_url = str(response.url)

            # If SPA detected and Browserless available, use it
            if is_spa_shell(html_content) and self.browserless_session:
                browserless_content, status = self._fetch_with_browserless(url)
                if status == 200:
                    html_content = browserless_content

            if not html_content:
                return

            soup = BeautifulSoup(html_content, "html.parser")

            # Classify current page
            page_category = self._classify_url(final_url)
            if page_category:
                results[page_category].append(final_url)

            # Extract and follow legal-relevant links
            if depth < self.max_depth:
                links = self._find_legal_links(soup, final_url)
                for link in links[:self.max_links]:
                    self._crawl(link, depth + 1, results)

        except Exception:
            pass

    def _find_legal_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Find links on the page that point to potential legal documents.

        Checks BOTH the URL path AND the visible link text for legal keywords.
        Returns deduplicated list of matching absolute URLs.
        """
        candidates = []
        seen = set()

        for a_tag in soup.find_all("a", href=True):
            href = a_tag.get("href", "")
            link_text = a_tag.get_text(strip=True).lower()

            resolved = self._resolve_url(href, base_url)
            if not resolved:
                continue

            normalized = self._normalize_url(resolved)
            if normalized in seen:
                continue
            if not self._is_same_root_domain(resolved, base_url):
                continue

            # Check URL path for legal keywords
            url_match = any(pat in resolved.lower() for pat in self.LEGAL_URL_PATTERNS)
            # Check link text for legal keywords
            text_match = any(kw in link_text for kw in self.LEGAL_LINK_TEXT_KEYWORDS)

            if url_match or text_match:
                seen.add(normalized)
                candidates.append(resolved)

        return candidates

    def _classify_url(self, url: str) -> Optional[str]:
        """Classify a URL by its path."""
        url_lower = url.lower()

        if any(k in url_lower for k in ["terms-of-use", "terms-of-service",
                                         "terms-and-conditions", "conditions",
                                         "user-agreement", "acceptable-use",
                                         "use-policy", "site-terms", "legal/terms",
                                         "summary-terms", "/terms"]):
            return "terms_of_use"

        if any(k in url_lower for k in ["privacy", "privacy-policy", "privacy-notice"]):
            return "privacy_policy"

        if any(k in url_lower for k in ["about", "about-us"]):
            return "about_pages"

        if any(k in url_lower for k in ["legal", "disclaimer", "notice",
                                         "guideline", "rule", "statement"]):
            return "other_legal"

        return None

    def _resolve_url(self, href: str, base_url: str) -> Optional[str]:
        """Resolve a href to a full URL."""
        if not href or href.startswith("#") or href.startswith("javascript:"):
            return None

        if href.startswith("http://") or href.startswith("https://"):
            return href

        return urljoin(base_url, href)

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison."""
        parsed = urlparse(url)
        path = parsed.path.rstrip("/").lower()
        return f"{parsed.scheme}://{parsed.netloc}{path}"

    def _get_root_domain(self, url: str) -> str:
        """Get the root domain (e.g., worldbank.org from data360.worldbank.org)."""
        parsed = urlparse(url)
        parts = parsed.netloc.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return parsed.netloc

    def _is_same_root_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs share the same root domain (e.g., worldbank.org)."""
        p1 = urlparse(url1)
        p2 = urlparse(url2)
        root1 = self._get_root_domain(f"https://{p1.netloc}")
        root2 = self._get_root_domain(f"https://{p2.netloc}")
        return root1 == root2

    def find_legal_links_from_url(self, start_url: str) -> Dict[str, str]:
        """
        Find legal document links starting from any URL.

        Returns best guess of terms/privacy URLs found.
        """
        discovery = self.discover_legal_pages(start_url)
        terms = discovery.get("terms_of_use", [])
        privacy = discovery.get("privacy_policy", [])
        other = discovery.get("other_legal", [])

        return {
            "terms_of_use": terms[0] if terms else None,
            "privacy_policy": privacy[0] if privacy else None,
            "other_legal": other[:5],
        }

    def scrape(self, url: str) -> "ScrapedDocument":
        """
        Scrape terms of use by exploring the site to discover the best candidate.

        Uses site exploration to find legal pages starting from the given URL.
        Returns the most promising terms-of-use page found.
        """
        from src.models.website_data import ScrapedDocument
        from datetime import datetime

        discovery = self.discover_legal_pages(url)
        terms_urls = discovery.get("terms_of_use", [])
        other_urls = discovery.get("other_legal", [])

        all_candidates = terms_urls + other_urls
        for candidate_url in all_candidates:
            result = self._try_scrape(candidate_url)
            if result.success and len(result.raw_content) >= 500:
                return result

        return ScrapedDocument(
            document_type="terms_of_use",
            url=url,
            raw_content="",
            scraped_at=datetime.now(),
            success=False,
            error_message="SiteExplorer: no terms page found through site exploration"
        )

    def _try_scrape(self, url: str) -> "ScrapedDocument":
        """Try httpx first, fallback to Browserless if SPA detected."""
        from src.models.website_data import ScrapedDocument
        from datetime import datetime

        try:
            response = httpx.get(url, headers=self._build_headers(),
                                timeout=self.timeout, follow_redirects=True)
            if response.status_code != 200:
                return ScrapedDocument(
                    document_type="terms_of_use", url=url, raw_content="",
                    scraped_at=datetime.now(), success=False,
                    error_message=f"HTTP {response.status_code}"
                )

            html_content = response.text

            # Check if it's an SPA shell - if so, try Browserless
            if is_spa_shell(html_content) and self.browserless_api_key:
                browserless_content, status = self._fetch_with_browserless(url)
                if status == 200 and len(browserless_content) > 500:
                    soup = BeautifulSoup(browserless_content, "html.parser")
                    for elem in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                        elem.decompose()
                    content = soup.get_text(separator="\n", strip=True)
                    return ScrapedDocument(
                        document_type="terms_of_use", url=url,
                        raw_content=content, scraped_at=datetime.now(), success=True
                    )

            # Normal httpx path
            soup = BeautifulSoup(html_content, "html.parser")
            for elem in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                elem.decompose()
            content = soup.get_text(separator="\n", strip=True)
            return ScrapedDocument(
                document_type="terms_of_use", url=str(response.url),
                raw_content=content, scraped_at=datetime.now(), success=True
            )

        except Exception as e:
            # If httpx failed and we have Browserless, try it
            if self.browserless_api_key:
                browserless_content, status = self._fetch_with_browserless(url)
                if status == 200:
                    soup = BeautifulSoup(browserless_content, "html.parser")
                    for elem in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                        elem.decompose()
                    content = soup.get_text(separator="\n", strip=True)
                    return ScrapedDocument(
                        document_type="terms_of_use", url=url,
                        raw_content=content, scraped_at=datetime.now(), success=True
                    )
            return ScrapedDocument(
                document_type="terms_of_use", url=url, raw_content="",
                scraped_at=datetime.now(), success=False, error_message=str(e)
            )
