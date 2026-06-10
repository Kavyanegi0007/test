import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from test.twitter.client import search_recent
from test.twitter.dedup import deduplicate
from test.twitter.models import EntityTweets


load_dotenv("test/.env")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


async def main():
    queries = [
        "Rick Caruso",
        "Rick Caruso USC",
        "Rick Caruso Palisades Fire",
    ]

    all_tweets = []

    for q in queries:
        try:
            result = await search_recent(q, max_results=10)
            print(f"  {q}: {result.count} tweets")
            all_tweets.extend(result.tweets)
        except Exception as e:
            print(f"  {q}: ERROR — {e}")
            continue

    print(f"\nTotal fetched: {len(all_tweets)}")
    deduped = deduplicate(all_tweets)
    print(f"After dedup: {len(deduped)}")

    output = EntityTweets(
        entity_name="Rick Caruso",
        tweets=deduped,
        total_fetched=len(all_tweets),
        unique_after_dedup=len(deduped),
        queries=queries,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )

    output_dir = Path("test/twitter/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"tweets_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    asyncio.run(main())
