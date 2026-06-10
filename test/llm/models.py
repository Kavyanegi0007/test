from pydantic import BaseModel


class Citation(BaseModel):
    url: str
    text: str = ""


class ProviderAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = []


class ProbeQuestionsOutput(BaseModel):
    entity_name: str
    questions: list[str]
    providers: dict[str, list[ProviderAnswer]] = {}
