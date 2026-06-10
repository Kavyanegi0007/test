from pydantic import BaseModel


class Article(BaseModel):
    title: str
    snippet: str = ""
    link: str = ""
    source: str = ""
    date: str = ""
    iso_date: str = ""
    query: str | None = None


class NewsResult(BaseModel):
    query: str
    articles: list[Article]
    count: int


class EntityNews(BaseModel):
    entity_name: str
    articles: list[Article]
    total_fetched: int
    queries: list[str] = []
    fetched_at: str = ""

