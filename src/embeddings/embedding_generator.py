"""Generates embeddings using sentence-transformers."""
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import torch

class EmbeddingGenerator:
    """Generates embeddings using sentence-transformers."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        batch_size = batch_size or self.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()

    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings."""
        return self.model.get_sentence_embedding_dimension()
