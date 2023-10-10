from typing import List, Union

import openai
from llama_index import Document
from prefect import flow, task

from lib.service.embedding import EmbeddingService
from lib.service.finetune import get_finetuning_service
from lib.utils.prisma import prisma
from prisma.models import Datasource


@task
async def create_vector_embeddings(
    datasource: Datasource,
) -> List[Union[Document, None]]:
    embedding_service = EmbeddingService(datasource=datasource)
    documents = await embedding_service.generate_documents()
    nodes = await embedding_service.generate_chunks(documents=documents)
    await embedding_service.generate_embeddings(nodes=nodes)
    return nodes


@task
async def create_finetuned_model(datasource: Datasource):
    embedding_service = EmbeddingService(datasource=datasource)
    documents = await embedding_service.generate_documents()
    nodes = await embedding_service.generate_chunks(documents=documents)
    finetunning_service = await get_finetuning_service(
        nodes=nodes, provider="openai", batch_size=5
    )
    training_file = await finetunning_service.generate_dataset()
    finetune_job = await finetunning_service.finetune(training_file=training_file)
    finetune = await openai.FineTune.retrieve(id=finetune_job.get("id"))
    await finetunning_service.cleanup(training_file=finetune_job.get("training_file"))
    return finetune


@flow(name="create_embeddings", description="Create embeddings", retries=0)
async def create_embeddings(datasource: Datasource):
    await create_vector_embeddings(datasource=datasource)


@flow(name="create_finetune", description="Create a finetune", retries=0)
async def create_finetune(datasource: Datasource):
    finetune = await create_finetuned_model(datasource=datasource)
    await prisma.datasource.update(
        where={"id": datasource.id},
        data={"finetune": finetune},
    )
