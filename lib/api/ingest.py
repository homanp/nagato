from fastapi import APIRouter

from lib.service.embedding import EmbeddingService
from lib.service.finetune import get_finetuning_service
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
    documents = await embedding_service.generate_documents()
    nodes = await embedding_service.generate_chunks(documents=documents)
    # embeddings = await embedding_service.generate_embeddings(nodes=nodes)
    finetunning_service = await get_finetuning_service(nodes=nodes, provider="openai")
    await finetunning_service.generate_dataset()
    # print(embeddings)
    return {"success": True, "data": None}
