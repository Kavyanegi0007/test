"""Standalone async SerpAPI client — no monitor.* dependencies."""

import logging
import os

import aiohttp

logger = logging.getLogger(__name__)

_SERPAPI_URL = "https://serpapi.com/search"


class SerpApiClient:
    def __init__(
        self,
        api_key: str,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._api_key = api_key
        self._owns_session = session is None
        self._session: aiohttp.ClientSession = session or aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=20),
        )

    @classmethod
    def from_settings(cls) -> "SerpApiClient":
        return cls(api_key=os.getenv("SERPAPI_API_KEY", ""))

    async def close(self) -> None:
        if self._owns_session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "SerpApiClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def search(self, params: dict) -> dict:
        async with self._session.get(
            _SERPAPI_URL,
            params={**params, "api_key": self._api_key},
            headers={"Accept-Encoding": "gzip, deflate"},
        ) as resp:
            resp.raise_for_status()
            return await resp.json(content_type=None)

    async def search_google_news(
        self,
        query: str,
        num: int = 20,
        gl: str = "us",
        hl: str = "en",
    ) -> list[dict]:
        params: dict[str, object] = {
            "engine": "google_news",
            "q": query,
            "gl": gl,
            "hl": hl,
        }
        data = await self.search(params)

        news_results = data.get("news_results", [])
        if not isinstance(news_results, list):
            return []

        articles: list[dict] = []
        seen_urls: set[str] = set()

        for item in news_results:
            if len(articles) >= num:
                break

            stories = item.get("stories")
            if stories and isinstance(stories, list):
                for story in stories:
                    if len(articles) >= num:
                        break
                    url = story.get("link", "")
                    title = story.get("title", "")
                    if not title or not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    source = story.get("source", {})
                    articles.append(
                        {
                            "title": title,
                            "snippet": story.get("snippet", ""),
                            "link": url,
                            "source": source.get("name", "") if isinstance(source, dict) else str(source),
                            "date": story.get("date", ""),
                            "iso_date": story.get("iso_date", ""),
                        }
                    )
            else:
                url = item.get("link", "")
                title = item.get("title", "")
                if not title or not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                source = item.get("source", {})
                articles.append(
                    {
                        "title": title,
                        "snippet": item.get("snippet", ""),
                        "link": url,
                        "source": source.get("name", "") if isinstance(source, dict) else str(source),
                        "date": item.get("date", ""),
                        "iso_date": item.get("iso_date", ""),
                    }
                )

        logger.info("SerpAPI news search: %s — %d articles", query[:60], len(articles))
        return articles[:num]

    async def search_google(
        self,
        query: str,
        num: int = 10,
        gl: str | None = None,
        hl: str | None = None,
    ) -> list[dict]:
        params: dict[str, object] = {
            "engine": "google",
            "q": query,
            "num": num,
        }
        if gl is not None:
            params["gl"] = gl
        if hl is not None:
            params["hl"] = hl
        data = await self.search(params)

        seen: set[str] = set()
        results: list[dict] = []

        sections_to_try = [
            "related_questions",
            "events_results",
            "short_videos",
            "organic_results",
            "inline_videos",
            "latest_posts",
            "perspectives",
            "knowledge_graph",
        ]

        for section in sections_to_try:
            section_data = data.get(section, [])
            if not section_data or not isinstance(section_data, list):
                continue

            for r in section_data[:num]:
                title = r.get("title")
                if not title:
                    continue

                link = r.get("link", r.get("image", ""))
                if link and link not in seen and title.strip():
                    seen.add(link)
                    results.append(
                        {
                            "title": title,
                            "snippet": r.get("snippet", ""),
                            "link": link,
                            "source": r.get("source", ""),
                            "date": r.get("date", ""),
                            "section": section,
                        }
                    )

                if len(results) >= num:
                    break
            if len(results) >= num:
                break

        logger.info("SerpAPI search: %s — %d results", query[:60], len(results))
        return results[:num]
