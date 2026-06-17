"""Generates embeddings using MiniMax embo-01 API."""
import os
from typing import List, Optional
import httpx


class EmbeddingGenerator:
    """
    Generates embeddings using MiniMax embo-01 model via API.
    Requires MINIMAX_API_KEY and MINIMAX_GROUP_ID environment variables,
    or pass them directly via api_key and group_id.
    """

    EMBEDDING_API_URL = "https://api.minimax.chat/v1/embeddings"
    BATCH_SIZE = 20  # safe batch size for API calls

    def __init__(
        self,
        model_name: str = "embo-01",
        api_key: Optional[str] = None,
        group_id: Optional[str] = None,
        base_url: str = "https://api.minimax.chat/v1",
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.batch_size = batch_size
        self.device = device or "cpu"
        self.http_client = httpx.Client(timeout=60.0)
        # Prefer explicit args > env var > Streamlit secrets
        self.api_key = (
            api_key
            or os.environ.get("MINIMAX_API_KEY")
            or self._safe_get_st_secrets("MINIMAX_API_KEY")
        )
        self.group_id = (
            group_id
            or os.environ.get("MINIMAX_GROUP_ID")
            or self._safe_get_st_secrets("MINIMAX_GROUP_ID")
        )

    @staticmethod
    def _safe_get_st_secrets(key: str) -> Optional[str]:
        """Try to get a value from Streamlit secrets, silently fail if not in Streamlit context."""
        try:
            import streamlit as st
            return st.secrets.get(key)
        except Exception:
            return None

    def _call_api(self, texts: List[str], embedding_type: str = "db") -> List[List[float]]:
        """Call MiniMax embeddings API for a batch of texts.
        type: "db" for document storage, "query" for user query retrieval.
        """
            raise ValueError(
                "MINIMAX_API_KEY and MINIMAX_GROUP_ID must be set. "
                "Set them as environment variables or pass api_key/group_id to constructor."
            )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "texts": texts,
            "type": embedding_type,
        }
        response = self.http_client.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=payload,
        )
        result = response.json()
        if "data" not in result:
            base_resp = result.get("base_resp", {})
            status_code = base_resp.get("status_code", "unknown")
            error_msg = base_resp.get("status_msg", str(result))
            raise ValueError(f"MiniMax embeddings API error ({status_code}): {error_msg}")
        response.raise_for_status()
        embeddings = sorted(result["data"], key=lambda x: x["index"])
        return [e["embedding"] for e in embeddings]

    def encode(self, texts: List[str], batch_size: Optional[int] = None, show_progress: bool = False) -> List[List[float]]:
        """
        Encode a list of texts into embeddings (batch, for document storage).
        Automatically batches requests to the API.
        """
        batch_size = batch_size or self.BATCH_SIZE
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(self._call_api(batch))
        return all_embeddings

    def encode_query(self, query: str) -> List[float]:
        """Encode a single query string into an embedding vector (type=query for retrieval)."""
        return self._call_api([query], embedding_type="query")[0]

    def similarity(self, query_vec: List[float], doc_vecs: List[List[float]]) -> List[float]:
        """Compute cosine similarity between query vector and document vectors."""
        # Import here to avoid hard dependency if not needed
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        q = np.array(query_vec).reshape(1, -1)
        d = np.array(doc_vecs)
        return cosine_similarity(q, d)[0].tolist()

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (embo-01 returns 1024-dim vectors)."""
        return 1024