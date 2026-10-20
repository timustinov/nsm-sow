from pydantic import BaseModel, Field, computed_field
from enum import StrEnum
from datetime import datetime, tzinfo


class VKCommentsInfo(BaseModel):
    count: int


class VKPostType(StrEnum):
    POST = "post"
    COPY = "copy"
    REPLY = "reply"
    POSTPONE = "postpone"
    SUGGEST = "suggest"


class VKCounter(BaseModel):
    count: int


class VKPost(BaseModel):
    id: int
    from_id: int
    raw_date: int = Field(alias="date")
    text: str
    comments: VKCommentsInfo
    likes: VKCounter
    reposts: VKCounter
    views: VKCounter
    post_type: VKPostType
    attachments: list
    marked_as_ads: int = 0
    edited: int | None = None
    copy_history: list | None = None

    def get_normalized_datetime(self, tzinfo: tzinfo) -> datetime:
        return datetime.fromtimestamp(self.raw_date, tzinfo)

    @computed_field
    @property
    def attachments_length(self) -> int:
        return len(self.attachments)
