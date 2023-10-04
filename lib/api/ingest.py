from fastapi import APIRouter

from lib.service.embedding import EmbeddingService
from lib.utils.prisma import prisma

router = APIRouter()


@router.post(
    "/ingest",
    name="ingest",
    description="Ingest data",
)
async def ingest(body: dict):
    """Endpoint for ingesting data"""
    datasource = await prisma.datasource.create(data={**body})
    embedding_service = EmbeddingService(datasource=datasource)
    documents = embedding_service.generate_documents()
    nodes = embedding_service.generate_chunks(documents=documents)
    embeddings = embedding_service.generate_embeddings(nodes=nodes)
    print(embeddings)
    return {"success": True, "data": None}
