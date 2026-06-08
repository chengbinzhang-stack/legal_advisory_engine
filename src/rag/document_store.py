"""Handles document storage and retrieval in the RAG system."""
from typing import List, Dict, Any, Optional
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.chroma_client import ChromaClient
from src.models.website_data import WebsiteData
from src.parser.text_chunker import TextChunker

class DocumentStore:
    """Handles document storage and retrieval in the RAG system."""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        chroma_client: ChromaClient,
        text_chunker: Optional[TextChunker] = None
    ):
        self.embedding_generator = embedding_generator
        self.chroma_client = chroma_client
        self.text_chunker = text_chunker or TextChunker()

    def store_website_data(
        self,
        website_data: WebsiteData,
        document_types: List[str] = None
    ) -> Dict[str, List[str]]:
        stored_chunks = {}
        for doc in website_data.documents:
            if document_types and doc.document_type not in document_types:
                continue
            metadata = {
                "website_url": website_data.url,
                "website_domain": website_data.domain,
                "document_type": doc.document_type,
                "scraped_at": doc.scraped_at.isoformat(),
            }
            chunks = self.text_chunker.chunk_text(
                doc.raw_content,
                metadata=metadata
            )
            if not chunks:
                continue
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_generator.encode(texts)
            chunk_ids = self.chroma_client.add_documents(
                documents=texts,
                embeddings=embeddings,
                metadatas=[chunk["metadata"] for chunk in chunks]
            )
            stored_chunks[doc.document_type] = chunk_ids
        return stored_chunks