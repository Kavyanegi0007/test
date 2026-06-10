"""X (Twitter) API v2 client — recent search and user lookup. No monitor.* dependencies."""

import logging
import os

import aiohttp

from test.twitter.models import AuthorInfo, SearchResult, Tweet

logger = logging.getLogger(__name__)

_SEARCH_ENDPOINT = "https://api.x.com/2/tweets/search/recent"
_USER_ENDPOINT = "https://api.x.com/2/users"
_TWEET_FIELDS = "created_at,public_metrics,author_id"
_USER_FIELDS = "public_metrics,verified,description"
_TWEET_EXPANSIONS = "author_id"


def _headers() -> dict:
    token = os.getenv("X_BEARER_TOKEN", "")
    return {"Authorization": f"Bearer {token}"}


async def search_recent(query: str, max_results: int = 20) -> SearchResult:
    """Search recent tweets matching a query (last 7 days)."""
    params = {
        "query": query,
        "max_results": str(min(max_results, 100)),
        "tweet.fields": _TWEET_FIELDS,
        "expansions": _TWEET_EXPANSIONS,
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(
            _SEARCH_ENDPOINT,
            headers=_headers(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 429:
                logger.warning("X API rate limited: %s", query[:60])
                return SearchResult(tweets=[], authors={}, query=query, count=0)

            resp.raise_for_status()
            data = await resp.json()

    authors: dict[str, AuthorInfo] = {}
    for user in data.get("includes", {}).get("users", []):
        metrics = user.get("public_metrics", {})
        authors[user["id"]] = AuthorInfo(
            id=user["id"],
            name=user.get("name", ""),
            username=user.get("username", ""),
            followers_count=metrics.get("followers_count", 0),
            following_count=metrics.get("following_count", 0),
            tweet_count=metrics.get("tweet_count", 0),
            verified=user.get("verified", False),
            description=user.get("description"),
        )

    tweets: list[Tweet] = []
    for raw in data.get("data", []):
        metrics = raw.get("public_metrics", {})
        author = authors.get(raw.get("author_id", ""))
        tweets.append(
            Tweet(
                id=raw["id"],
                text=raw.get("text", ""),
                created_at=raw.get("created_at"),
                author_id=raw.get("author_id"),
                author_name=author.name if author else None,
                author_username=author.username if author else None,
                like_count=metrics.get("like_count", 0),
                retweet_count=metrics.get("retweet_count", 0),
                reply_count=metrics.get("reply_count", 0),
                query=query,
            )
        )

    logger.info("X search complete: %s — %d results", query[:60], len(tweets))
    return SearchResult(tweets=tweets, authors=authors, query=query, count=len(tweets))


async def lookup_user(user_id: str) -> AuthorInfo | None:
    """Look up a single user by ID."""
    url = f"{_USER_ENDPOINT}/{user_id}"
    params = {"user.fields": _USER_FIELDS}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=_headers(), params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()

    raw = data.get("data", {})
    if not raw:
        return None

    metrics = raw.get("public_metrics", {})
    return AuthorInfo(
        id=raw["id"],
        name=raw.get("name", ""),
        username=raw.get("username", ""),
        followers_count=metrics.get("followers_count", 0),
        following_count=metrics.get("following_count", 0),
        tweet_count=metrics.get("tweet_count", 0),
        verified=raw.get("verified", False),
        description=raw.get("description"),
    )
