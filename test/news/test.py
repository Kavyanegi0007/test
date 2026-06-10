import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from test.llm.questions import QueryDisambiguator
from test.news.client import search_news
from test.news.models import EntityNews
from test.query_builder import (
    ClientSearchProfile,
    QueryDisambiguationInput,
    QueryAugmentationInput,
    QueryCombineInput,
    query_disambiguation,
    query_augmentation,
    combine,
)


load_dotenv("test/.env")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


async def main():
    profiles = QueryDisambiguator.load_profile("test/llm/profile.json")
    profile = profiles[0]
    entity_name = profile.entity_name

    cp = profile.client_profile
    client_search_profile = ClientSearchProfile(
        category=cp.category,
        industry=cp.industry,
        legal_name=cp.name if cp.name != entity_name else None,
        additional_context=cp.additional_context or "",
        sensitivities=cp.sensitivities or [],
    )

    disambig = await query_disambiguation(QueryDisambiguationInput(entity_name=entity_name))
    augmented = await query_augmentation(
        QueryAugmentationInput(
            query=disambig.query,
            num_queries=10,
            client_profile=client_search_profile,
        )
    )
    bundle = await combine(
        QueryCombineInput(
            canonical=entity_name,
            disambiguated_query=disambig.query,
            orm_queries=augmented.queries,
            entity_name=entity_name,
            client_profile=client_search_profile,
        )
    )
    queries = bundle.all_queries
    logging.info("Generated %d queries from query_builder", len(queries))

    all_articles = []
    seen_urls: set[str] = set()

    for q in queries:
        result = await search_news(q, num=5)
        print(f"  {q}: {result.count} articles")
        for a in result.articles:
            if a.link and a.link not in seen_urls:
                seen_urls.add(a.link)
                all_articles.append(a)

    print(f"\nTotal fetched (deduped by URL): {len(all_articles)}")

    output = EntityNews(
        entity_name=entity_name,
        articles=all_articles,
        total_fetched=len(all_articles),
        queries=queries,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )

    output_dir = Path("test/news/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"news_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    asyncio.run(main())
