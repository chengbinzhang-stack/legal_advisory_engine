"""
Configuration settings for Legal Advisory Engine
"""
import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EngineConfig:
    """Main configuration for the Legal Data Protection Engine."""

    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5
    min_request_delay: float = 2.0
    user_agent: str = "LegalAdvisoryBot/1.0 (+https://example.com/bot)"
    respect_robots_txt: bool = True
    javascript_rendering: bool = False

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_device: Optional[str] = None

    chroma_persist_directory: str = "./chroma_db"
    chroma_collection_name: str = "legal_documents"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100

    anthropic_api_key: Optional[str] = None
    minimax_api_key: Optional[str] = "sk-cp-qzt0Jl0fXRCv5d460DOTSfJz6FWAHj_oDHMKIYm0HCbGPSLcNSgijNr2vrm2Hy1CFINZK8i2bbDE_tGkJFWRT1u1Ary2lcXTjQAQ_kmGgZjT2a78bdZ0Q2Q"
    minimax_model: str = "MiniMax-Text-01"
    minimax_base_url: str = "https://api.minimax.chat/v1"
    llm_model: str = "MiniMax-Text-01"
    llm_max_tokens: int = 2048

    max_workers: int = 4
    log_level: str = "INFO"

    data_directory: str = "./data"
    summaries_directory: str = "./data/summaries"

    example_websites: list = field(default_factory=lambda: [
        "https://fada.in/",
        "https://vahan.parivahan.gov.in/vahan4dashboard/",
        "https://data360.worldbank.org/en/api",
        "https://fred.stlouisfed.org/categories"
    ])

    def __post_init__(self):
        if api_key := os.environ.get("ANTHROPIC_API_KEY"):
            self.anthropic_api_key = api_key
        if api_key := os.environ.get("MINIMAX_API_KEY"):
            self.minimax_api_key = api_key
        if device := os.environ.get("EMBEDDING_DEVICE"):
            self.embedding_device = device
        os.makedirs(self.chroma_persist_directory, exist_ok=True)
        os.makedirs(self.summaries_directory, exist_ok=True)