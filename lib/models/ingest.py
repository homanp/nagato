from pydantic import BaseModel
from typing import Optional


class IngestRequest(BaseModel):
    type: str
    url: Optional[str]
    content: Optional[str]
    webhook_url: Optional[str]
