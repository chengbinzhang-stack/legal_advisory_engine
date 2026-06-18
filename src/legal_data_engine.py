"""Legal Data Engine - Main orchestration class."""
import json
import os
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from datetime import datetime

from config import EngineConfig
from src.scraper.terms_scraper import TermsScraper
from src.scraper.privacy_scraper import PrivacyScraper
from src.scraper.robots_scraper import RobotsScraper
from src.scraper.browserless_session import BrowserlessSessionManager
from src.models.website_data import WebsiteData
from src.models.legal_analysis import LegalAnalysis
from src.classifier.legal_classifier import LegalClassifier
from src.parser.text_chunker import TextChunker
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.chroma_client import ChromaClient
from src.rag.document_store import DocumentStore

class LegalDataEngine:
    def __init__(self, config: EngineConfig = None):
        self.config = config or EngineConfig()
        self.classifier = LegalClassifier(
            api_key=self.config.minimax_api_key,
            base_url=self.config.minimax_base_url
        )
        self.text_chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size
        )
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.config.embedding_model,
            api_key=self.config.minimax_api_key,
            group_id=self.config.minimax_group_id,
            base_url=self.config.minimax_base_url,
            batch_size=self.config.embedding_batch_size,
            device=self.config.embedding_device,
        )
        self.chroma_client = ChromaClient(
            persist_directory=self.config.chroma_persist_directory,
            collection_name=self.config.chroma_collection_name
        )
        self.document_store = DocumentStore(
            self.embedding_generator,
            self.chroma_client,
            self.text_chunker
        )
        self._analysis_cache: Dict[str, LegalAnalysis] = {}

    def process_website(self, url: str) -> LegalAnalysis:
        domain = self._extract_domain(url)

        # Check cache first - return cached result if valid and not expired
        cached = self._get_cached_analysis(domain)
        if cached:
            return cached

        # No cache or expired - scrape the website
        scraped_data = self._scrape_website(url)
        combined_text = self._combine_documents(scraped_data)
        robots_txt = ''
        for doc in scraped_data.documents:
            if doc.document_type == 'robots_txt' and doc.success:
                robots_txt = doc.raw_content
                break

        # Build document_type -> url mapping for proper citation
        document_urls = {
            doc.document_type: doc.url
            for doc in scraped_data.documents if doc.success
        }

        # If terms and privacy both failed, skip LLM - insufficient data
        terms_ok = any(d.document_type == 'terms_of_use' and d.success for d in scraped_data.documents)
        privacy_ok = any(d.document_type == 'privacy_policy' and d.success for d in scraped_data.documents)

        if not terms_ok and not privacy_ok:
            # Cannot analyze - return failed result
            from src.models.legal_analysis import LegalAnalysis, PermissionAnalysis, PermissionLevel, WebsiteCategory
            error_analysis = LegalAnalysis(
                website_url=url,
                website_domain=domain,
                category=WebsiteCategory.UNKNOWN,
                category_reasoning="Failed to scrape terms and privacy pages - insufficient data for analysis",
                permissions={},
                unique_findings=[],
                summary_text=f"Analysis failed: could not retrieve terms_of_use or privacy_policy pages. Terms scraper status: {[d.success for d in scraped_data.documents if d.document_type in ['terms_of_use','privacy_policy']]}"
            )
            self._save_analysis(error_analysis)
            self._analysis_cache[domain] = error_analysis
            return error_analysis

        analysis = self.classifier.classify_permissions(
            text=combined_text,
            website_url=url,
            website_domain=domain,
            robots_txt=robots_txt,
            document_urls=document_urls
        )
        self.document_store.store_website_data(scraped_data)
        self._save_analysis(analysis)
        self._analysis_cache[domain] = analysis
        return analysis

    def _get_cached_analysis(self, domain: str) -> Optional[LegalAnalysis]:
        """Get cached analysis if it exists and is not expired."""
        if self.config.cache_duration_hours <= 0:
            return None  # Cache disabled

        # Check in-memory cache first
        if domain in self._analysis_cache:
            return self._analysis_cache[domain]

        # Check summary file
        summary_path = os.path.join(
            self.config.summaries_directory,
            f"summary_{domain.replace('.', '_')}.json"
        )
        if not os.path.exists(summary_path):
            return None

        try:
            with open(summary_path, "r") as f:
                data = json.load(f)

            # Check if within cache duration
            processed_at = datetime.fromisoformat(data["processed_at"])
            age_hours = (datetime.now() - processed_at).total_seconds() / 3600
            if age_hours > self.config.cache_duration_hours:
                return None  # Cache expired

            return self._load_analysis_from_json(data)
        except Exception:
            return None

    def _scrape_website(self, url: str) -> WebsiteData:
        # Create a shared Browserless session for all scrapers (saves API units)
        session_manager = None
        if self.config.browserless_api_key:
            session_manager = BrowserlessSessionManager(self.config.browserless_api_key)
            session_manager.create_session()

        terms_scraper = TermsScraper(
            timeout=self.config.timeout,
            browserless_api_key=self.config.browserless_api_key,
            browserless_session=session_manager
        )
        privacy_scraper = PrivacyScraper(
            timeout=self.config.timeout,
            browserless_api_key=self.config.browserless_api_key,
            browserless_session=session_manager
        )
        robots_scraper = RobotsScraper(timeout=self.config.timeout)
        domain = self._extract_domain(url)
        website_data = WebsiteData(url=url, domain=domain)

        try:
            print(f"[LegalDataEngine] Starting terms_scraper.scrape: {url}", flush=True)
            website_data.documents.append(terms_scraper.scrape(url))
            print(f"[LegalDataEngine] Done terms_scraper.scrape", flush=True)

            print(f"[LegalDataEngine] Starting privacy_scraper.scrape: {url}", flush=True)
            website_data.documents.append(privacy_scraper.scrape(url))
            print(f"[LegalDataEngine] Done privacy_scraper.scrape", flush=True)

            print(f"[LegalDataEngine] Starting robots_scraper.scrape: {url}", flush=True)
            website_data.documents.append(robots_scraper.scrape(url))
            print(f"[LegalDataEngine] Done robots_scraper.scrape", flush=True)
        finally:
            # Always close the session when done
            if session_manager:
                session_manager.close()

        website_data.processed_at = datetime.now()
        return website_data

    def _combine_documents(self, website_data: WebsiteData) -> str:
        texts = []
        for doc in website_data.documents:
            if doc.success and doc.raw_content:
                texts.append(doc.raw_content)
        return "\n\n".join(texts)

    def _extract_domain(self, url: str) -> str:
        parsed = urlparse(url)
        domain = parsed.netloc
        # Strip leading www. to normalize: www.example.com → example.com
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    def _save_analysis(self, analysis: LegalAnalysis):
        summary_path = os.path.join(
            self.config.summaries_directory,
            f"summary_{analysis.website_domain.replace('.', '_')}.json"
        )
        data = {
            "website_url": analysis.website_url,
            "website_domain": analysis.website_domain,
            "category": analysis.category.value,
            "category_name": analysis.category.name,
            "processed_at": datetime.now().isoformat(),
            "permissions": {
                k: {
                    "level": v.permission.value,
                    "confidence": v.confidence_score,
                    "reasoning": v.reasoning,
                    "relevant_excerpts": v.relevant_excerpts,
                    "source_documents": v.source_documents
                } for k, v in analysis.permissions.items()
            },
            "unique_findings": analysis.unique_findings,
            "summary_text": analysis.summary_text
        }
        with open(summary_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_website_analysis(self, domain: str) -> Optional[LegalAnalysis]:
        if domain in self._analysis_cache:
            return self._analysis_cache[domain]
        summary_path = os.path.join(
            self.config.summaries_directory,
            f"summary_{domain.replace('.', '_')}.json"
        )
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                data = json.load(f)
            return self._load_analysis_from_json(data)
        return None

    def _load_analysis_from_json(self, data: Dict) -> LegalAnalysis:
        from src.models.legal_analysis import PermissionAnalysis, PermissionLevel, WebsiteCategory
        permissions = {}
        for k, v in data.get("permissions", {}).items():
            permissions[k] = PermissionAnalysis(
                parameter_name=k,
                permission=PermissionLevel(v["level"]),
                reasoning=v.get("reasoning", ""),
                relevant_excerpts=v.get("relevant_excerpts", []),
                source_documents=v.get("source_documents", []),
                confidence_score=v.get("confidence", 0.0)
            )
        return LegalAnalysis(
            website_url=data["website_url"],
            website_domain=data["website_domain"],
            category=WebsiteCategory(data["category"]),
            category_reasoning="",
            permissions=permissions,
            unique_findings=data.get("unique_findings", []),
            summary_text=data.get("summary_text", "")
        )
