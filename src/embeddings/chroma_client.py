"""ChromaDB client wrapper for vector storage."""
import chromadb
from chromadb.config import Settings
from typing import List, Optional, Dict, Any
import uuid

class ChromaClient:
    """Wrapper for ChromaDB operations."""

    COLLECTION_NAME = "legal_documents"

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = COLLECTION_NAME
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        # Use in-memory mode to avoid inotify limits on Streamlit Cloud
        self.client = chromadb.PersistentClient(
            path=":memory:",
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Legal documents for RAG"}
        )

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the collection."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        return ids

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query the collection."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        return results

    def get_by_website(
        self,
        website_domain: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all documents for a specific website."""
        results = self.collection.query(
            query_embeddings=[None],
            n_results=limit,
            where={"website_domain": website_domain}
        )
        return results

    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)

    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
