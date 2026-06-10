"""LLM provider model builders and question probing logic — no monitor.* dependencies."""

import asyncio
import logging
import os
import re

from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch as WebSearchCapability
from pydantic_ai.messages import ModelResponse, NativeToolReturnPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from test.llm.models import Citation, ProbeQuestionsOutput, ProviderAnswer

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"(?:\((https?://[^)\s]+)\)|\[[^\]]*\]\((https?://[^)\s]+)\))")


def _extract_citations_from_response(response) -> list[Citation]:
    seen: set[str] = set()
    citations: list[Citation] = []
    for msg in response.all_messages():
        if not isinstance(msg, ModelResponse):
            continue
        for part in msg.parts:
            if not isinstance(part, NativeToolReturnPart):
                continue
            if part.tool_name != "web_search":
                continue
            content = part.content
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    url = item.get("url") or item.get("uri")
                    if url and url not in seen:
                        seen.add(url)
                        citations.append(Citation(url=url, text=item.get("title", "") or ""))
            elif isinstance(content, dict):
                for src in content.get("sources", []):
                    if not isinstance(src, dict):
                        continue
                    url = src.get("url") or src.get("uri")
                    if url and url not in seen:
                        seen.add(url)
                        citations.append(Citation(url=url, text=src.get("title", "") or ""))
    if not citations:
        citations = _extract_citations_from_text(str(response.output))
    return citations


def _extract_citations_from_text(text: str) -> list[Citation]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    citations: list[Citation] = []
    for para in paragraphs:
        matches = list(_URL_RE.finditer(para))
        for match in matches:
            url = match.group(1) or match.group(2)
            citations.append(Citation(url=url, text=para[:200]))
    return citations


def _get_model(name: str):
    if name == "claude":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return None
        model_name = os.getenv("ANTHROPIC_MODEL_NAME", "claude-haiku-4-5")
        return AnthropicModel(model_name, provider=AnthropicProvider(api_key=key))
    if name == "chatgpt":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        return OpenAIResponsesModel("gpt-4o-mini", provider=OpenAIProvider(api_key=key))
    if name == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            return None
        return GoogleModel("gemini-2.0-flash", provider=GoogleProvider(api_key=key))
    return None


def available_providers() -> list[str]:
    providers = ["claude"]
    if os.getenv("OPENAI_API_KEY"):
        providers.append("chatgpt")
    if os.getenv("GEMINI_API_KEY"):
        providers.append("gemini")
    return providers


async def probe_questions(
    questions: list[str],
    client_name: str,
    industry: str = "",
    audit_text: str = "",
) -> ProbeQuestionsOutput:
    async def _answer_question(model, question: str) -> ProviderAnswer:
        agent = Agent(
            model=model,
            capabilities=[WebSearchCapability(search_context_size="medium", max_uses=3)],
        )
        prompt = f"""You are an AI assistant. Answer the following question about {client_name} accurately and in detail, using web search to ground your answer in current information.

Industry context: {industry}
{f"Audit context: {audit_text[:300]}" if audit_text else ""}

Question: {question}

Write a thorough, multi-paragraph response (3-5 paragraphs). Be factual and objective. Include citations from your web search."""
        try:
            response = await agent.run(prompt)
            text = str(response.output)
            citations = _extract_citations_from_response(response)
        except Exception as e:
            logger.warning("Question probe failed: %s — %s", question[:60], e)
            text = f"Unable to answer this question about {client_name}."
            citations = []
        return ProviderAnswer(question=question, answer=text, citations=citations)

    providers: dict[str, list[ProviderAnswer]] = {}

    for provider_name in available_providers():
        model = _get_model(provider_name)
        if model is None:
            providers[provider_name] = []
            continue

        tasks = [asyncio.create_task(_answer_question(model, q)) for q in questions]
        answers = await asyncio.gather(*tasks, return_exceptions=True)

        provider_answers: list[ProviderAnswer] = []
        for answer in answers:
            if isinstance(answer, BaseException):
                logger.error("Provider %s question failed: %s", provider_name, answer)
                provider_answers.append(ProviderAnswer(question="", answer=f"Error: {answer}", citations=[]))
            else:
                provider_answers.append(answer)

        providers[provider_name] = provider_answers
        logger.info("Provider %s complete: %d answers", provider_name, len(provider_answers))

    return ProbeQuestionsOutput(entity_name=client_name, questions=questions, providers=providers)
