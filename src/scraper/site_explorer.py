"""Site explorer for discovering legal document pages by crawling site navigation."""
from typing import List, Optional, Dict, Set
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup
from src.scraper.base_scraper import BaseScraper


class SiteExplorer(BaseScraper):
    """
    Explore a website to discover legal document pages.

    Given a starting URL (e.g., about page), crawls navigation links
    to find terms of use, privacy policy, and other legal pages.
    """

    LEGAL_PATTERNS = [
        "terms", "condition", "agreement", "legal", "privacy",
        "policy", "notice", "disclaimer", "guideline", "rule",
        "use-policy", "acceptable-use", "user-agreement", "statement",
    ]

    NAVIGATION_TAGS = ["nav", "header", "footer", "aside", "menu", "sidebar"]

    def __init__(self, max_depth: int = 2, max_links: int = 50, *args, **kwargs):
        """
        Initialize SiteExplorer.

        Args:
            max_depth: Maximum crawl depth from starting URL
            max_links: Maximum number of links to collect per page
        """
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth
        self.max_links = max_links
        self._visited: Set[str] = set()

    def discover_legal_pages(self, start_url: str) -> Dict[str, List[str]]:
        """
        Discover legal-related pages starting from a URL.

        Returns a dict mapping category to list of discovered URLs.
        """
        results = {
            "terms_of_use": [],
            "privacy_policy": [],
            "other_legal": [],
            "about_pages": [],
        }

        self._visited.clear()
        self._crawl(start_url, depth=0, results=results)

        # Dedupe
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
            response = httpx.get(url, headers=self._build_headers(),
                                timeout=self.timeout, follow_redirects=True)
            if response.status_code != 200:
                return

            soup = BeautifulSoup(response.text, "html.parser")
            final_url = str(response.url)

            # Classify current page
            page_category = self._classify_url(final_url)
            if page_category:
                results[page_category].append(final_url)

            # Find and crawl navigation links
            if depth < self.max_depth:
                nav_links = self._extract_nav_links(soup, final_url)
                for link in nav_links[:self.max_links]:
                    self._crawl(link, depth + 1, results)

        except Exception:
            pass

    def _extract_nav_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract navigation links from the page."""
        links = []
        seen = set()

        # Extract from navigation elements first
        for tag in self.NAVIGATION_TAGS:
            for element in soup.find_all(tag):
                for a_tag in element.find_all("a", href=True):
                    href = a_tag["href"]
                    resolved = self._resolve_url(href, base_url)
                    if resolved and resolved not in seen:
                        normalized = self._normalize_url(resolved)
                        if self._is_same_domain(normalized, base_url):
                            seen.add(normalized)
                            links.append(resolved)

        # Fall back to all links if navigation is sparse
        if len(links) < 5:
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                resolved = self._resolve_url(href, base_url)
                if resolved and resolved not in seen:
                    normalized = self._normalize_url(resolved)
                    if self._is_same_domain(normalized, base_url):
                        seen.add(normalized)
                        links.append(resolved)

        return links

    def _classify_url(self, url: str) -> Optional[str]:
        """Classify a URL by its path."""
        url_lower = url.lower()

        if any(k in url_lower for k in ["terms-of-use", "terms-of-service",
                                         "terms-and-conditions", "conditions",
                                         "user-agreement", "acceptable-use",
                                         "use-policy", "site-terms", "legal/terms"]):
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
            parsed = urlparse(href)
            if not self._is_same_domain(parsed.netloc, base_url):
                return None
            return href

        return urljoin(base_url, href)

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison."""
        parsed = urlparse(url)
        path = parsed.path.rstrip("/").lower()
        return f"{parsed.scheme}://{parsed.netloc}{path}"

    def _is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs are from the same domain."""
        p1 = urlparse(url1)
        p2 = urlparse(url2)
        return p1.netloc == p2.netloc or (
            p1.netloc.endswith(p2.netloc) or p2.netloc.endswith(p1.netloc)
        )

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
