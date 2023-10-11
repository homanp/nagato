import asyncio

from prisma.models import Datasource
from fastapi import APIRouter, BackgroundTasks
from lib.service.flows import create_finetune

from lib.models.ingest import IngestRequest
from lib.utils.prisma import prisma

router = APIRouter()


@router.post(
    "/ingest",
    name="ingest",
    description="Ingest data",
)
async def ingest(body: IngestRequest, background_tasks: BackgroundTasks):
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
