"""Deduplicate tweets — by ID, then by TF-IDF cosine similarity on text."""

import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from test.twitter.models import Tweet

logger = logging.getLogger(__name__)


def deduplicate(tweets: list[Tweet], threshold: float = 0.7) -> list[Tweet]:
    """Remove exact ID duplicates, then near-duplicate texts via TF-IDF."""
    if len(tweets) <= 1:
        return tweets

    seen_ids: set[str] = set()
    unique: list[Tweet] = []
    for t in tweets:
        if t.id not in seen_ids:
            seen_ids.add(t.id)
            unique.append(t)

    if len(unique) <= 1:
        return unique

    texts = [t.text for t in unique]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        kept_indices: list[int] = []
        removed: set[int] = set()

        for i in range(len(texts)):
            if i in removed:
                continue
            kept_indices.append(i)
            for j in range(i + 1, len(texts)):
                if similarity_matrix[i, j] >= threshold:
                    removed.add(j)

        result = [unique[i] for i in kept_indices]

        logger.info(
            "Tweet dedup complete: %d → %d → %d",
            len(tweets),
            len(unique),
            len(result),
        )

        return result

    except Exception as e:
        logger.error("Tweet dedup failed: %s, falling back to ID-only", e)
        return unique
