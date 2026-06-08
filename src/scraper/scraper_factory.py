"""Factory for creating scraper instances."""
from typing import Dict, Type
from src.scraper.base_scraper import BaseScraper
from src.scraper.terms_scraper import TermsScraper
from src.scraper.privacy_scraper import PrivacyScraper
from src.scraper.robots_scraper import RobotsScraper
from src.scraper.site_explorer import SiteExplorer

class ScraperFactory:
    """Factory for creating appropriate scraper instances."""

    SCRAPER_TYPES: Dict[str, Type[BaseScraper]] = {
        "terms_of_use": TermsScraper,
        "privacy_policy": PrivacyScraper,
        "robots_txt": RobotsScraper,
        "site_explorer": SiteExplorer,
    }

    @classmethod
    def create(cls, document_type: str, **kwargs) -> BaseScraper:
        """Create a scraper for the specified document type."""
        scraper_class = cls.SCRAPER_TYPES.get(document_type)
        if not scraper_class:
            raise ValueError(f"Unknown document type: {document_type}")
        return scraper_class(**kwargs)

    @classmethod
    def create_all(cls, **kwargs) -> Dict[str, BaseScraper]:
        """Create all scraper types."""
        return {doc_type: cls.create(doc_type, **kwargs)
                for doc_type in cls.SCRAPER_TYPES}
