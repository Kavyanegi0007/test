"""Dynamic probing question generator — LLM-powered questions from entity profile + augmented queries."""

import json
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent


class ClientProfile(BaseModel):
    name: str | None = None
    category: str | None = None
    industry: str | None = None
    sensitivities: list[str] = []
    additional_context: str | None = None


class EntityProfile(BaseModel):
    entity_name: str
    client_profile: ClientProfile
    client_id: str | None = None


class _QuestionsOutput(BaseModel):
    questions: list[str]
    reasoning: str | None = None


class QueryDisambiguator:
    """
    Uses an LLM to generate dynamic probing questions from an entity profile
    and a list of augmented search queries.
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    @staticmethod
    def load_profile(path: str | Path) -> list[EntityProfile]:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [EntityProfile(**item) for item in data]
        return [EntityProfile(**data)]

    @staticmethod
    def _build_context(profile: EntityProfile) -> str:
        p = profile.client_profile
        lines = [f"Entity: {profile.entity_name}"]
        if p.name and p.name != profile.entity_name:
            lines.append(f"Legal name: {p.name}")
        if p.category:
            lines.append(f"Category: {p.category}")
        if p.industry:
            lines.append(f"Industry: {p.industry}")
        if p.sensitivities:
            lines.append("Sensitivities:")
            for s in p.sensitivities[:5]:
                lines.append(f"  - {s[:200]}")
        if p.additional_context:
            lines.append(f"Additional context: {p.additional_context[:500]}")
        return "\n".join(lines)

    async def generate_questions(
        self,
        queries: list[str],
        profile: EntityProfile,
        num_questions: int = 7,
    ) -> list[str]:
        context = self._build_context(profile)
        queries_block = "\n".join(f"- {q}" for q in queries)

        prompt = f"""You are a reputation research strategist. Given the entity profile and search queries below, generate {num_questions} precise probing questions to assess how this entity is perceived online.

Entity profile:
{context}

Search queries about this entity:
{queries_block}

Rules for each question:
- Target a distinct reputation angle (media coverage, legal, professional history, social sentiment, financial, community impact, etc.)
- Answerable via a single web search
- Frame neutrally — do not lead the answer toward positive or negative
- Cover diverse angles — do not cluster around one topic
- Be specific enough to surface relevant information, not generic

Output ONLY valid JSON: {{"questions": ["...", "...", ...], "reasoning": "..."}}"""

        response = await self._agent.run(prompt, output_type=_QuestionsOutput)
        return response.output.questions
