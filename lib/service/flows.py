from typing import List, Union

import openai
from llama_index import Document
from prefect import flow, task

from lib.models.ingest import IngestRequest
from lib.service.embedding import EmbeddingService
from lib.service.finetune import get_finetuning_service


@task
async def create_vector_embeddings(
    payload: IngestRequest,
    finetune_id: str,
) -> List[Union[Document, None]]:
    embedding_service = EmbeddingService(payload=payload)
    documents = await embedding_service.generate_documents()
    nodes = await embedding_service.generate_chunks(documents=documents)
    await embedding_service.generate_embeddings(nodes=nodes, finetune_id=finetune_id)
    return nodes


@task
async def create_finetuned_model(payload: IngestRequest):
    embedding_service = EmbeddingService(payload=payload)
    documents = await embedding_service.generate_documents()
    nodes = await embedding_service.generate_chunks(documents=documents)
    finetunning_service = await get_finetuning_service(
        nodes=nodes,
        provider=payload.provider,
        batch_size=5,
        base_model=payload.base_model,
    )
    training_file = await finetunning_service.generate_dataset()
    formatted_training_file = await finetunning_service.validate_dataset(
        training_file=training_file
    )
    finetune = await finetunning_service.finetune(training_file=formatted_training_file)
    if payload.provider == "OPENAI":
        finetune = await openai.FineTune.retrieve(id=finetune.get("id"))
    await finetunning_service.cleanup(training_file=finetune.get("training_file"))
    return finetune


@flow(name="create_finetune", description="Create a finetune", retries=0)
async def create_finetune(payload: IngestRequest):
    finetune = await create_finetuned_model(payload=payload)
    await create_vector_embeddings(payload=payload, finetune_id=finetune.get("id"))
