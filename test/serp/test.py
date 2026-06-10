import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from test.llm.questions import QueryDisambiguator
from test.query_builder import (
    ClientSearchProfile,
    QueryCombineInput,
    QueryDisambiguationInput,
    QueryAugmentationInput,
    combine,
    query_augmentation,
    query_disambiguation,
)
from test.serp.models import EntitySerp, OrganicResult, SerpResult
from test.serp_client import SerpApiClient

load_dotenv("test/.env")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


async def search_serp(query: str, num: int = 10) -> SerpResult:
    async with SerpApiClient.from_settings() as client:
        raw = await client.search_google(query, num=num)
    results = [OrganicResult(**r, query=query) for r in raw]
    return SerpResult(query=query, results=results, count=len(results))


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

    all_results = []
    seen_links: set[str] = set()

    for q in queries:
        result = await search_serp(q, num=10)
        print(f"  {q}: {result.count} results")
        for r in result.results:
            if r.link and r.link not in seen_links:
                seen_links.add(r.link)
                all_results.append(r)

    print(f"\nTotal fetched (deduped by link): {len(all_results)}")

    output = EntitySerp(
        entity_name=entity_name,
        results=all_results,
        total_fetched=len(all_results),
        queries=queries,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )

    output_dir = Path("test/serp/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"serp_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    asyncio.run(main())
