"""
Query Builder Demo — independent smoke test with realistic mock data.

Run::

    python -m monitor.services.serp.query_builder_demo

Each function is tested independently with the same Rick Caruso mock data.
Live API calls (LLM, Google Suggest) are attempted but gracefully
skipped if the endpoint is unavailable.
"""

import asyncio
import json
import sys
from pathlib import Path

from monitor.integrations.google.knowledge_graph import KgCandidate
from monitor.services.serp.query_builder import (
    ClientSearchProfile,
    QueryAugmentationInput,
    QueryBundle,
    QueryCombineInput,
    QueryDisambiguationInput,
    RegistryEntityContext,
    combine,
    query_augmentation,
    query_disambiguation,
)

# ── Loaders ────────────────────────────────────────────

_JSON_PATH = Path(__file__).parent / "query_builder_mock_data_2.json"


def _load_mock_data() -> dict:
    """Load mock data from the companion JSON file."""
    with open(_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


DATA = _load_mock_data()

MOCK_KG = KgCandidate.model_validate(DATA["kg_candidate"])
MOCK_CLIENT_PROFILE = ClientSearchProfile.model_validate(DATA["client_profile"])
MOCK_REGISTRY = RegistryEntityContext.model_validate(DATA["registry_context"])
MOCK_ORM_QUERIES: list[str] = DATA["orm_queries"]


# ── Tests ──────────────────────────────────────────────

async def test_disambiguation() -> None:
    """Resolve 'Rick Caruso' → the single best query for Google Autocomplete."""
    print("─── Function 1: query_disambiguation() ───")
    inp = QueryDisambiguationInput(entity_name="Rick Caruso", kg_entity=MOCK_KG)
    print(f"  Input:    entity_name='{inp.entity_name}', kg_provided=True")
    out = await query_disambiguation(inp)
    print(f"  Output:   query='{out.query}'")
    print()


async def test_augmentation() -> None:
    """Generate reputational-angle queries for Rick Caruso."""
    print("─── Function 2: query_augmentation() ────")
    inp = QueryAugmentationInput(
        query="Rick Caruso",
        client_profile=MOCK_CLIENT_PROFILE,
        registry_context=MOCK_REGISTRY,
    )
    print(f"  Input:    query='{inp.query}', num_queries={inp.num_queries}")
    print(f"            profile_fields={MOCK_CLIENT_PROFILE.model_dump(exclude_none=True)}")
    print(f"            registry_aliases={MOCK_REGISTRY.aliases}")
    out = await query_augmentation(inp)
    print(f"  Output:   {len(out.queries)} queries")
    for i, q in enumerate(out.queries, 1):
        print(f"            {i}. {q}")
    print()


async def test_combine() -> None:
    """Merge canonical + autocomplete suggestions + ORM queries."""
    print("─── Function 3: combine() ────────────────")
    inp = QueryCombineInput(
        canonical="Rick Caruso",
        disambiguated_query="Rick Caruso",
        orm_queries=MOCK_ORM_QUERIES,
        entity_name="Rick Caruso",
        client_profile=MOCK_CLIENT_PROFILE,
    )
    print(f"  Input:    canonical='{inp.canonical}', orm_count={len(inp.orm_queries)}")
    print(f"            disambiguated_query='{inp.disambiguated_query}'")
    bundle = await combine(inp)
    print(f"  Output:   total={bundle.total_count}")
    for i, q in enumerate(bundle.all_queries, 1):
        print(f"            {i}. {q}")
    print(f"  Autocomplete raw: {bundle.autocomplete_suggestions}")
    print()


async def test_combine_minimal() -> None:
    """combine() with no ORM queries and no profile (fallback behaviour)."""
    print("─── Function 3b: combine() minimal ────────")
    inp = QueryCombineInput(
        canonical="Rick Caruso",
        disambiguated_query="Rick Caruso",
        orm_queries=[],
    )
    print(f"  Input:    canonical='{inp.canonical}', no ORM, no profile")
    bundle = await combine(inp)
    print(f"  Output:   total={bundle.total_count}")
    for i, q in enumerate(bundle.all_queries, 1):
        print(f"            {i}. {q}")
    print()


def test_serialisation() -> None:
    """Verify QueryBundle serialises to JSON for DB storage."""
    print("─── Model serialisation ─────────────")
    bundle = QueryBundle(
        canonical="Rick Caruso",
        all_queries=["Rick Caruso", "Rick Caruso"],
        autocomplete_suggestions=[],
        autocomplete_count=0,
        orm_count=0,
        total_count=2,
    )
    as_dict = bundle.model_dump()
    print(f"  model_dump keys: {list(as_dict.keys())}")
    print(f"  JSON: {json.dumps(as_dict)}")
    print()


# ── Main ───────────────────────────────────────────────

async def _main() -> None:
    print("=" * 70)
    print("QUERY BUILDER — Independent smoke test")
    print("=" * 70)
    print()

    tests = [
        ("query_disambiguation", test_disambiguation),
        ("query_augmentation", test_augmentation),
        ("combine", test_combine),
        ("combine_minimal", test_combine_minimal),
    ]

    for name, fn in tests:
        try:
            await fn()
        except Exception as e:
            print(f"  SKIP ({name}): {e}", file=sys.stderr)
            print()

    test_serialisation()
    print("Done. Live API calls gracefully skipped if unavailable.")


if __name__ == "__main__":
    asyncio.run(_main())
