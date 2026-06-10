"""News search client — wraps SerpApiClient for Google News."""

import logging

from test.news.models import Article, NewsResult
from test.serp_client import SerpApiClient

logger = logging.getLogger(__name__)


async def search_news(query: str, num: int = 20) -> NewsResult:
    """Search Google News via SerpAPI, return structured articles."""
    async with SerpApiClient.from_settings() as client:
        raw = await client.search_google_news(query, num=num)

    articles = [Article(**a, query=query) for a in raw]

    logger.info("News search complete: %s — %d articles", query[:60], len(articles))
    return NewsResult(query=query, articles=articles, count=len(articles))
