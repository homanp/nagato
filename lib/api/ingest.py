import asyncio

from fastapi import APIRouter

from lib.models.ingest import IngestRequest
from lib.service.flows import create_finetune
from prisma.models import Datasource

router = APIRouter()


@router.post(
    "/ingest",
    name="ingest",
    description="Ingest data",
)
async def ingest(body: IngestRequest):
    """Endpoint for ingesting data"""

    async def run_training_flow():
        try:
            await create_finetune(
                payload=body,
            )
        except Exception as flow_exception:
            raise flow_exception

    asyncio.create_task(run_training_flow())
    return {"success": True}
