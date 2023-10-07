from fastapi import APIRouter

from lib.models.ingest import IngestRequest
from service.flows import create_finetune, create_embeddings
from lib.utils.prisma import prisma

router = APIRouter()


@router.post(
    "/ingest",
    name="ingest",
    description="Ingest data",
)
async def ingest(body: IngestRequest):
    """Endpoint for ingesting data"""
    datasource = await prisma.datasource.create(data={**body})
    await create_embeddings(datasource=datasource)
    await create_finetune(datasource=datasource)
    return {"success": True, "data": datasource}
