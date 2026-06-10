from pydantic import BaseModel


class OrganicResult(BaseModel):
    title: str
    snippet: str = ""
    link: str = ""
    source: str = ""
    date: str = ""
    section: str = ""
    query: str | None = None


class SerpResult(BaseModel):
    query: str
    results: list[OrganicResult]
    count: int


class EntitySerp(BaseModel):
    entity_name: str
    results: list[OrganicResult]
    total_fetched: int
    queries: list[str] = []
    fetched_at: str = ""
