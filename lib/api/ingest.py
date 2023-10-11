import asyncio

from fastapi import APIRouter

from lib.models.ingest import IngestRequest
from lib.service.flows import create_finetune
from lib.utils.prisma import prisma
from prisma.models import Datasource

router = APIRouter()


@router.post(
    "/ingest",
    name="ingest",
    description="Ingest data",
)
async def ingest(body: IngestRequest):
    """Endpoint for ingesting data"""
    datasource = await prisma.datasource.create(data=body.dict())

    async def run_training_flow(datasource: Datasource):
        try:
            await create_finetune(
                datasource=datasource,
            )
        except Exception as flow_exception:
            raise flow_exception

    asyncio.create_task(run_training_flow(datasource=datasource))
    return {"success": True, "data": datasource}
