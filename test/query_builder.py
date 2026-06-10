"""
Query Builder — typed module for building search queries for SERP analysis.
Self-contained, no monitor.* dependencies.
"""

import asyncio
import json
import logging
import os

import aiohttp
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

logger = logging.getLogger(__name__)

_SUGGEST_URL = "https://suggestqueries.google.com/complete/search"
_KG_URL = "https://kgsearch.googleapis.com/v1/entities:search"

# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────


class KgCandidate(BaseModel):
    name: str
    score: float
    description: str | None = None
    detailed_description: str | None = None
    aliases: list[str] = []
    entity_id: str | None = None
    entity_types: list[str] = []


class ClientSearchProfile(BaseModel):
    category: str | None = None
    industry: str | None = None
    legal_name: str | None = None
    additional_context: str = ""
    confidence_note: str | None = None
    sensitivities: list[str] = []


class RegistryEntityContext(BaseModel):
    description: str | None = None
    handles: list[str] = []
    aliases: list[str] = []
    disambiguation: list[str] = []


class _AutocompleteQueryOutput(BaseModel):
    query: str
    reasoning: str | None = None


class _AugmentedQueriesOutput(BaseModel):
    queries: list[str]


class QueryDisambiguationInput(BaseModel):
    entity_name: str
    kg_entity: KgCandidate | None = None


class QueryDisambiguationOutput(BaseModel):
    query: str


class QueryAugmentationInput(BaseModel):
    query: str = ""
    num_queries: int = 10
    client_profile: ClientSearchProfile | None = None
    registry_context: RegistryEntityContext | None = None


class QueryAugmentationOutput(BaseModel):
    queries: list[str]


class QueryCombineInput(BaseModel):
    canonical: str
    disambiguated_query: str
    orm_queries: list[str]
    entity_name: str | None = None
    client_profile: ClientSearchProfile | None = None
    num_autocomplete_suggestions: int = 15


class QueryBundle(BaseModel):
    canonical: str
    all_queries: list[str]
    autocomplete_suggestions: list[str]
    autocomplete_count: int
    orm_count: int
    total_count: int


# ──────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────


def _build_model():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    model_name = os.getenv("ANTHROPIC_MODEL_NAME", "claude-haiku-4-5")
    return AnthropicModel(model_name, provider=AnthropicProvider(api_key=key))


def _build_entity_block(
    canonical_name: str,
    client_profile: ClientSearchProfile | None,
    registry_context: RegistryEntityContext | None,
) -> str:
    lines = [f"Entity: {canonical_name}"]
    if registry_context:
        if registry_context.description:
            lines.append(f"Description: {registry_context.description}")
        if registry_context.aliases:
            lines.append(f"Known aliases: {', '.join(registry_context.aliases)}")
        if registry_context.disambiguation:
            lines.append(f"Disambiguation: {'; '.join(registry_context.disambiguation)}")
        if registry_context.handles:
            lines.append(f"Official handles: {', '.join(registry_context.handles)}")
    if not client_profile:
        return "\n".join(lines)
    if client_profile.category:
        lines.append(f"Category: {client_profile.category}")
    if client_profile.industry:
        lines.append(f"Industry: {client_profile.industry}")
    if client_profile.legal_name and client_profile.legal_name != canonical_name:
        lines.append(f"Legal name: {client_profile.legal_name}")
    if client_profile.additional_context:
        lines.append(f"ORM context: {client_profile.additional_context}")
    confidence_note = client_profile.confidence_note or ""
    if "Google Knowledge Graph confirmed entity" in confidence_note or "KG description:" in confidence_note:
        lines.append("Google Knowledge Graph: confirmed entity match")
    if client_profile.sensitivities:
        lines.append(f"Known sensitivities: {', '.join(client_profile.sensitivities[:5])}")
    return "\n".join(lines)


def _build_reputation_seeds(entity_name: str, client_profile: ClientSearchProfile) -> list[str]:
    sensitivities = client_profile.sensitivities or []
    if not sensitivities:
        return []
    seeds: list[str] = []
    seen: set[str] = set()
    for sensitivity in sensitivities:
        topic = sensitivity.split(" - ")[0].strip()
        if not topic:
            continue
        segments = [s.strip() for s in topic.split(" and ")]
        for segment in segments:
            words = segment.split()
            if len(words) > 3:
                seed_text = " ".join(words[:3])
            else:
                seed_text = segment
            seed = f"{entity_name} {seed_text}".strip()
            seed_key = seed.lower()
            if seed and seed_key not in seen:
                seen.add(seed_key)
                seeds.append(seed)
    return seeds


async def _search_kg(query: str, limit: int = 5) -> list[KgCandidate]:
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return []
    params = {
        "query": query,
        "limit": str(limit),
        "indent": "true",
        "key": api_key,
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        async with session.get(_KG_URL, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)
    candidates: list[KgCandidate] = []
    for item in data.get("itemListElement", []):
        result = item.get("result", {})
        score = float(item.get("resultScore", 0))
        detailed = result.get("detailedDescription", {})
        detailed_text = detailed.get("articleBody") if isinstance(detailed, dict) else None
        raw_types = result.get("@type", [])
        if isinstance(raw_types, str):
            raw_types = [raw_types]
        entity_types = [str(t) for t in raw_types]
        candidates.append(
            KgCandidate(
                name=result.get("name", query),
                score=score,
                description=result.get("description"),
                detailed_description=detailed_text,
                aliases=[
                    a.get("@value", a) if isinstance(a, dict) else str(a) for a in result.get("alternateName", [])
                ],
                entity_id=result.get("@id"),
                entity_types=entity_types,
            )
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


async def _resolve_kg(query: str) -> KgCandidate | None:
    candidates = await _search_kg(query, limit=5)
    return candidates[0] if candidates else None


async def _fetch_single(query: str) -> list[str]:
    params = {
        "client": "firefox",
        "q": query,
        "hl": "en",
        "gl": "us",
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        async with session.get(
            _SUGGEST_URL,
            params=params,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ORMA/1.0)"},
        ) as resp:
            resp.raise_for_status()
            text = await resp.text()
    data = json.loads(text)
    if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
        return [str(s) for s in data[1]]
    return []


async def _fetch_autocomplete_suggestions(
    disambiguated_query: str,
    client_profile: ClientSearchProfile | None,
    entity_name: str,
) -> list[str]:
    name = entity_name or disambiguated_query
    queries = [disambiguated_query]
    if client_profile:
        seeds = _build_reputation_seeds(name, client_profile)
        if seeds:
            queries.extend(seeds)
    results = await asyncio.gather(*[_fetch_single(q) for q in queries])
    seen: set[str] = set()
    merged: list[str] = []
    for suggestions in results:
        for s in suggestions:
            key = s.strip().lower()
            if key not in seen:
                seen.add(key)
                merged.append(s)
    return merged


def _merge_deduplicate(
    canonical: str,
    autocomplete: list[str],
    orm: list[str],
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    def _add(q: str) -> None:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            ordered.append(q.strip())

    _add(canonical)
    for q in autocomplete:
        _add(q)
    for q in orm:
        _add(q)
    return ordered


# ──────────────────────────────────────────────
# Public functions
# ──────────────────────────────────────────────


async def query_disambiguation(input_: QueryDisambiguationInput) -> QueryDisambiguationOutput:
    entity_name = input_.entity_name
    kg_entity = input_.kg_entity
    if kg_entity is not None:
        if kg_entity.score >= 0.3:
            pass
        else:
            return QueryDisambiguationOutput(query=entity_name)
    else:
        if not os.getenv("GOOGLE_API_KEY"):
            return QueryDisambiguationOutput(query=entity_name)
        try:
            kg_entity = await _resolve_kg(entity_name)
        except Exception:
            pass
        if not kg_entity or kg_entity.score < 0.3:
            return QueryDisambiguationOutput(query=entity_name)
    aliases_text = f"\n- Known aliases: {', '.join(kg_entity.aliases)}" if kg_entity.aliases else ""
    description_text = f"\n- Description: {kg_entity.description}" if kg_entity.description else ""
    detailed_text = f"\n- Detailed bio: {kg_entity.detailed_description}" if kg_entity.detailed_description else ""
    context_block = f"""Google Knowledge Graph confirmed entity:
- Canonical name: {kg_entity.name}
- KG confidence score: {kg_entity.score:.3f} (>= 0.3 = confident match){description_text}{detailed_text}{aliases_text}
"""
    model = _build_model()
    agent = Agent(model=model)
    prompt = f"""
You are an expert search strategist. Given the confirmed entity below, determine the SINGLE best 1–3 word Google search query to use for Google Autocomplete.

{context_block}

OBJECTIVE:
Generate a short Google query that maximizes the chance that Google Autocomplete returns suggestions about the intended entity.

GENERAL RULES:
- The query must be short (1–3 words).
- Prefer the most recognizable public-facing name of the entity.
- The query must uniquely identify the intended entity.
- Do NOT use unnecessary extra words.
- Do NOT include generic suffixes like:
  - news
  - controversy
  - scandal
  - reviews
  - company reviews
  - reddit
  - lawsuit
- Autocomplete will naturally expand into those topics.

DISAMBIGUATION RULES:
- If the entity name could reasonably refer to:
  - a common dictionary word,
  - another famous entity,
  - a product category,
  - a place,
  - a profession,
  - or multiple unrelated meanings,
  then add a short disambiguating term.

- The disambiguating term should clarify the entity type, industry, profession, or category.

- Examples of useful disambiguators:
  - company
  - singer
  - actor
  - DJ
  - airline
  - software
  - brand
  - politician
  - university
  - crypto
  - app

- Avoid ambiguity with generic words.
- Do not confuse the entity with a generic word, category, or unrelated concept.
- The goal is to ensure autocomplete suggestions are about the correct specific person, place, organization, product, or thing.

SELECTION PRIORITY:
1. Use the canonical public identity if already distinctive.
2. If ambiguous, add the smallest possible clarifier.
3. Keep the query natural and realistically searchable.

OUTPUT FORMAT:
Return ONLY valid JSON:
"""
    try:
        response = await agent.run(prompt, output_type=_AutocompleteQueryOutput)
        resolved_query = response.output.query.strip()
        return QueryDisambiguationOutput(query=resolved_query)
    except Exception:
        return QueryDisambiguationOutput(query=entity_name)


async def query_augmentation(input_: QueryAugmentationInput) -> QueryAugmentationOutput:
    query = input_.query
    num_queries = input_.num_queries
    client_profile = input_.client_profile
    registry_context = input_.registry_context
    entity_block = _build_entity_block(query, client_profile, registry_context)
    supplemental = num_queries - 1
    prompt = f"""You are a search query optimization expert specialising in Online Reputation Management (ORM).

{entity_block}

Your task: generate exactly {supplemental} additional Google search queries (beyond the canonical entity name itself) that would surface reputationally significant results for this entity.

Query strategy — cover as many of these angles as the entity warrants:
- News coverage and media mentions
- Legal history, lawsuits, regulatory actions
- Controversies and criticism
- Professional profile and career history
- Business dealings, partnerships, competitors
- Social media presence and public discourse
- Third-party references and analyst opinions

CRITICAL RULES:
- Every query must unambiguously refer to THIS specific entity, not a common word or different entity with the same name.
- If the entity name is short or generic (e.g. "Apple", "LionTree", "Banks"), always append a disambiguating term (industry, role, location) to each query — e.g. "LionTree investment bank advisory" not just "LionTree".
- Do NOT generate generic queries like "{query} news" or "{query} controversy" — the supplemental queries must target specific reputation angles.
- Queries should be 2-5 words. Short enough for Google to return broad results, specific enough to avoid the wrong entity.
- Do NOT repeat the bare canonical name as one of the {supplemental} queries — it is already the first query.

Generate exactly {supplemental} distinct queries.
Return ONLY a JSON object with a "queries" key containing an array of {supplemental} strings.
"""
    model = _build_model()
    agent = Agent(model=model)
    try:
        response = await agent.run(prompt, output_type=_AugmentedQueriesOutput)
        supplemental_queries = response.output.queries[:supplemental]
        return QueryAugmentationOutput(queries=[query, *supplemental_queries])
    except Exception:
        return QueryAugmentationOutput(queries=[query])


async def combine(input_: QueryCombineInput) -> QueryBundle:
    canonical = input_.canonical
    disambiguated_query = input_.disambiguated_query
    orm_queries = input_.orm_queries
    entity_name = input_.entity_name or canonical
    client_profile = input_.client_profile
    max_ac = input_.num_autocomplete_suggestions
    autocomplete_raw = await _fetch_autocomplete_suggestions(disambiguated_query, client_profile, entity_name)
    autocomplete = [s for s in (autocomplete_raw or []) if s][:max_ac]
    orm_supplemental = [q for q in (orm_queries or [])[1:] if q]
    merged = _merge_deduplicate(canonical, autocomplete, orm_supplemental)
    return QueryBundle(
        canonical=canonical,
        all_queries=merged,
        autocomplete_suggestions=autocomplete,
        autocomplete_count=len(autocomplete),
        orm_count=len(orm_supplemental),
        total_count=len(merged),
    )
