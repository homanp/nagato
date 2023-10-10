import asyncio

from prisma.models import Datasource
from fastapi import APIRouter, BackgroundTasks
from lib.service.flows import create_embeddings, create_finetune

from lib.models.ingest import IngestRequest
from lib.utils.prisma import prisma

router = APIRouter()


async def run_embedding_flow(datasource: Datasource):
    await create_embeddings(
        datasource=datasource,
    )


async def run_finetune_flow(datasource: Datasource):
    await create_finetune(
        datasource=datasource,
    )


@router.post(
    "/ingest",
    name="ingest",
    description="Ingest data",
)
async def ingest(body: IngestRequest, background_tasks: BackgroundTasks):
    """Endpoint for ingesting data"""
    datasource = await prisma.datasource.create(data=body.dict())

    background_tasks.add_task(run_embedding_flow, datasource=datasource)
    background_tasks.add_task(run_finetune_flow, datasource=datasource)
    return {"success": True, "data": datasource}
