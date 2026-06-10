import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from test.llm.questions import QueryDisambiguator
from test.llm.client import probe_questions
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
    key = os.getenv("ANTHROPIC_API_KEY", "")
    model_name = os.getenv("ANTHROPIC_MODEL_NAME", "claude-haiku-4-5")
    model = AnthropicModel(model_name, provider=AnthropicProvider(api_key=key))
    agent = Agent(model=model)
    disambiguator = QueryDisambiguator(agent=agent)

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

    questions = await disambiguator.generate_questions(queries, profile)
    print(f"\nGenerated {len(questions)} questions:\n")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")

    industry = cp.industry or ""
    audit_text = cp.additional_context or ""

    print(f"\n--- Probing {len(questions)} questions ---\n")
    results = await probe_questions(
        questions=questions,
        client_name=entity_name,
        industry=industry,
        audit_text=audit_text,
    )

    for provider, answers in results.providers.items():
        print(f"  {provider}: {len(answers)} answers")

    output = results.model_dump()
    output["generated_at"] = datetime.now(timezone.utc).isoformat()
    output["queries"] = queries

    output_dir = Path("test/llm/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"probe_results_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
