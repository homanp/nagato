import asyncio

from fastapi import APIRouter

from lib.models.ingest import IngestRequest
from lib.service.flows import create_finetune

router = APIRouter()


async def ingest(payload: IngestRequest):
    """Endpoint for ingesting data"""

    async def run_training_flow():
        try:
            await create_finetune(
                payload=payload,
            )
        except Exception as flow_exception:
            raise flow_exception

    asyncio.create_task(run_training_flow())
    return {"success": True}
