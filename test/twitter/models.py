from pydantic import BaseModel


class Tweet(BaseModel):
    id: str
    text: str
    created_at: str | None = None
    author_id: str | None = None
    author_name: str | None = None
    author_username: str | None = None
    like_count: int = 0
    retweet_count: int = 0
    reply_count: int = 0
    query: str | None = None


class AuthorInfo(BaseModel):
    id: str
    name: str
    username: str
    followers_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    verified: bool = False
    description: str | None = None


class EntityTweets(BaseModel):
    entity_name: str
    tweets: list[Tweet] = []
    authors: dict[str, AuthorInfo] = {}
    total_fetched: int = 0
    unique_after_dedup: int = 0
    queries: list[str] = []
    fetched_at: str = ""


class SearchResult(BaseModel):
    tweets: list[Tweet]
    authors: dict[str, AuthorInfo]
    query: str
    count: int
