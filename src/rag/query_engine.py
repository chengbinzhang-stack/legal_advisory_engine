"""RAG query engine for retrieving relevant legal information."""
from typing import List, Dict, Any, Optional
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.chroma_client import ChromaClient

# Maximum cosine distance for a result to be considered relevant
# Cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
# Results with distance > this threshold are filtered out
MAX_SIMILARITY_DISTANCE = 1.0


class QueryEngine:
    """RAG query engine for retrieving relevant legal information."""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        chroma_client: ChromaClient
    ):
        self.embedding_generator = embedding_generator
        self.chroma_client = chroma_client

    def query(
        self,
        query_text: str,
        website_domain: Optional[str] = None,
        document_types: Optional[List[str]] = None,
        n_results: int = 10
    ) -> Dict[str, Any]:
        query_embedding = self.embedding_generator.encode_query(query_text)
        where_clause = {}
        if website_domain:
            where_clause["website_domain"] = website_domain
        if document_types:
            where_clause["document_type"] = {"$in": document_types}
        if where_clause:
            results = self.chroma_client.query(
                query_embedding=query_embedding,
                n_results=n_results,
                where=where_clause
            )
        else:
            results = self.chroma_client.query(
                query_embedding=query_embedding,
                n_results=n_results
            )
        formatted_results = self._format_results(results)
        context = self._build_context(formatted_results)
        return {
            "query": query_text,
            "results": formatted_results,
            "context": context,
            "total_results": len(formatted_results)
        }

    def _format_results(
        self,
        raw_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter out results with low similarity (high distance)."""
        formatted = []
        if not raw_results.get("documents"):
            return formatted
        for i, doc in enumerate(raw_results["documents"][0]):
            distance = raw_results["distances"][0][i] if raw_results.get("distances") else None
            # Skip results with distance > threshold (too dissimilar)
            if distance is not None and distance > MAX_SIMILARITY_DISTANCE:
                continue
            formatted.append({
                "text": doc,
                "metadata": raw_results["metadatas"][0][i] if raw_results.get("metadatas") else {},
                "distance": distance,
            })
        return formatted

    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        context_parts = ["Context from legal documents:"]
        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            doc_type = metadata.get("document_type", "unknown")
            domain = metadata.get("website_domain", "unknown")
            header = "--- Document %d (%s from %s) ---" % (i, doc_type, domain)
            context_parts.append(header)
            context_parts.append(result["text"][:1000])
        return "".join(context_parts)
