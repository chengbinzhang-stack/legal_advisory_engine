"""Chunks text into overlapping segments for embedding."""
from typing import List, Dict, Any, Optional
import re

class TextChunker:
    """Chunks text into overlapping segments for embedding."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(
        self,
        text: str,
        metadata: Optional[dict] = None
    ) -> List[Dict[str, Any]]:
        """Chunk text into overlapping segments."""
        paragraphs = self._split_into_paragraphs(text)
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)
            if current_size + para_size > self.chunk_size:
                if current_size >= self.min_chunk_size:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk_dict = {
                        "text": chunk_text,
                        "metadata": (metadata.copy() if metadata else {})
                    }
                    chunk_dict["metadata"]["chunk_size"] = len(chunk_text)
                    chunks.append(chunk_dict)
                overlap_text = current_chunk[-1:] if current_chunk else []
                current_chunk = overlap_text + [para]
                current_size = sum(len(p) for p in current_chunk)
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk and current_size >= self.min_chunk_size:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_dict = {
                "text": chunk_text,
                "metadata": (metadata.copy() if metadata else {})
            }
            chunk_dict["metadata"]["chunk_size"] = len(chunk_text)
            chunks.append(chunk_dict)

        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n|(?<=\n)(?=[A-Z])', text)
        return [p.strip() for p in paragraphs if p.strip()]
