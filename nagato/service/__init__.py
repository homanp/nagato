from typing import List, Union

import openai
from llama_index import Document

from nagato.service.embedding import EmbeddingService
from nagato.service.finetune import get_finetuning_service


def create_vector_embeddings(
    type: str, filter_id: str, url: str = None, content: str = None
) -> List[Union[Document, None]]:
    embedding_service = EmbeddingService(type=type, content=content, url=url)
    documents = embedding_service.generate_documents()
    nodes = embedding_service.generate_chunks(documents=documents)
    embedding_service.generate_embeddings(nodes=nodes, filter_id=filter_id)
    return nodes


def create_finetuned_model(
    provider: str,
    base_model: str,
    type: str,
    url: str = None,
    content: str = None,
    webhook_url: str = None,
):
    embedding_service = EmbeddingService(type=type, url=url, content=content)
    documents = embedding_service.generate_documents()
    nodes = embedding_service.generate_chunks(documents=documents)
    finetunning_service = get_finetuning_service(
        nodes=nodes,
        provider=provider,
        batch_size=5,
        base_model=base_model,
        num_questions_per_chunk=1,
    )
    training_file = finetunning_service.generate_dataset()
    formatted_training_file = finetunning_service.validate_dataset(
        training_file=training_file
    )
    finetune = finetunning_service.finetune(
        training_file=formatted_training_file, webhook_url=webhook_url
    )
    if provider == "OPENAI":
        finetune = openai.FineTune.retrieve(id=finetune.get("id"))
    finetunning_service.cleanup(training_file=finetune.get("training_file"))
    return finetune
