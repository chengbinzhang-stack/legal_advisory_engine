"""Generates embeddings using TF-IDF (lightweight, no heavy deps)."""
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class EmbeddingGenerator:
    """
    Generates embeddings using TF-IDF vectorizer.
    Lightweight alternative to sentence-transformers for Streamlit Cloud deployment.
    """

    def __init__(self, model_name: str = "tfidf", device: Optional[str] = None, batch_size: int = 32):
        self.device = device or "cpu"
        self.batch_size = batch_size
        self.vectorizer = TfidfVectorizer(max_features=384, min_df=1, max_df=0.95, ngram_range=(1, 3), stop_words="english")
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: List[str], batch_size: Optional[int] = None, show_progress: bool = False) -> List[List[float]]:
        if not self._fitted:
            self.fit(texts)
        return self.vectorizer.transform(texts).toarray().tolist()

    def encode_query(self, query: str) -> List[float]:
        if not self._fitted:
            return np.zeros(384).tolist()
        return self.vectorizer.transform([query]).toarray()[0].tolist()

    def similarity(self, query_vec: List[float], doc_vecs: List[List[float]]) -> List[float]:
        q = np.array(query_vec).reshape(1, -1)
        d = np.array(doc_vecs)
        max_dim = max(q.shape[1], d.shape[1])
        q = np.pad(q, ((0, 0), (0, max_dim - q.shape[1])))
        d = np.pad(d, ((0, 0), (0, max_dim - d.shape[1])))
        return cosine_similarity(q, d)[0].tolist()

    @property
    def embedding_dim(self) -> int:
        return 384