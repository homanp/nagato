import json
from typing import List, Union

import openai
from llama_index import Document
from prefect import flow, task

from lib.service.embedding import EmbeddingService
from lib.service.finetune import get_finetuning_service
from lib.utils.prisma import prisma
from prisma.models import Datasource
from prisma import Json


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
        nodes=nodes,
        provider=datasource.provider,
        batch_size=5,
        base_model=datasource.base_model,
    )
    training_file = await finetunning_service.generate_dataset()
    formatted_training_file = await finetunning_service.validate_dataset(
        training_file=training_file
    )
    finetune = await finetunning_service.finetune(training_file=formatted_training_file)
    if datasource.provider == "OPENAI":
        finetune = await openai.FineTune.retrieve(id=finetune.get("id"))
    await finetunning_service.cleanup(training_file=finetune.get("training_file"))
    return finetune


@flow(name="create_finetune", description="Create a finetune", retries=0)
async def create_finetune(datasource: Datasource):
    await create_vector_embeddings(datasource=datasource)
    finetune = await create_finetuned_model(datasource=datasource)
    print(finetune)
    await prisma.datasource.update(
        where={"id": datasource.id},
        data={"finetune": Json(data=finetune)},
    )
