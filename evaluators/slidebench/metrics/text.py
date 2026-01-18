"""Text similarity metric using sentence embeddings."""
from typing import Optional

from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Lazy-loaded model
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Get or initialize the sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def get_text_similarity_prev(text1: str, text2: str) -> Optional[float]:
    """Calculate text similarity using character overlap (SequenceMatcher).

    Args:
        text1: First text string.
        text2: Second text string.

    Returns:
        Similarity ratio between 0.0 and 1.0, or None if either input is None.
    """
    if (text1 is None) or (text2 is None):
        return None
    return SequenceMatcher(None, text1, text2).ratio()


def get_text_similarity(text1: str, text2: str) -> Optional[float]:
    """Calculate text similarity using sentence embeddings.

    Uses a pretrained sentence-transformer model to compute cosine similarity
    between text embeddings.

    Args:
        text1: First text string.
        text2: Second text string.

    Returns:
        Cosine similarity score, or None if either input is empty/None.
    """
    if not text1 or not text2:
        return None

    model = _get_model()
    embeddings = model.encode([text1, text2], convert_to_tensor=True).cpu()
    similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))

    return similarity[0][0]


if __name__ == "__main__":
    # Test
    text1 = "Language for Robots Circa 2018"
    text2 = "Language for Robots Circa 2020"
    print(get_text_similarity(text1, text2))
