from pydantic import BaseModel


class IngestRequest(BaseModel):
    webhook_url: str
