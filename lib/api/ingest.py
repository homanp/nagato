import openai

from fastapi import APIRouter

from lib.service.embedding import EmbeddingService
from lib.service.finetune import get_finetuning_service
from lib.utils.prisma import prisma
from lib.models.ingest import IngestRequest

router = APIRouter()


@router.post(
    "/ingest",
    name="ingest",
    description="Ingest data",
)
async def ingest(body: IngestRequest):
    """Endpoint for ingesting data"""
    webhook_url = body.webhook_url
    datasource = await prisma.datasource.create(data={**body})
    embedding_service = EmbeddingService(datasource=datasource)
    documents = await embedding_service.generate_documents()
    nodes = await embedding_service.generate_chunks(documents=documents)
    await embedding_service.generate_embeddings(nodes=nodes)
    finetunning_service = await get_finetuning_service(
        nodes=nodes, provider="openai", batch_size=5
    )
    await finetunning_service.generate_dataset()
    finetune_job = await finetunning_service.finetune()
    finetune = await openai.FineTune.retrieve(id=finetune_job.id)
    await prisma.datasource.update(
        where={"id": datasource.id},
        data={"webhook_url": webhook_url, "finetune": finetune},
    )
    await finetunning_service.cleanup(training_file=finetune_job.get("training_file"))
    return {"success": True, "data": datasource}
