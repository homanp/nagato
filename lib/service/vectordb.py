from abc import ABC, abstractmethod
from typing import List

import pinecone
from decouple import config
from numpy import ndarray


class VectorDBService(ABC):
    def __init__(self, index_name: str, dimension: int, namespace: str = None):
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension

    @abstractmethod
    def upsert():
        pass

    @abstractmethod
    def query():
        pass


class PineconeVectorService(VectorDBService):
    def __init__(self, index_name: str, dimension: int, namespace: str = None):
        super().__init__(
            index_name=index_name, dimension=dimension, namespace=namespace
        )
        pinecone.init(api_key=config("PINECONE_API_KEY"))
        # Create a new vector index if it doesn't
        # exist dimensions should be passed in the arguments
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name, metric="cosine", shards=1, dimension=dimension
            )
        self.index = pinecone.Index(index_name=self.index_name)

    def upsert(self, vectors: ndarray):
        self.index.upsert(vectors=vectors, namespace=self.namespace)

    def query(self, queries: List[ndarray], top_k: int, include_metadata: bool = True):
        return self.index.query(
            queries=queries,
            top_k=top_k,
            include_metadata=include_metadata,
            namespace=self.namespace,
        )


def get_vector_service(
    provider: str, index_name: str, namespace: str = None, dimension: int = 384
):
    services = {
        "pinecone": PineconeVectorService,
        # Add other providers here
        # e.g "weaviate": WeaviateVectorService,
    }
    service = services.get(provider)
    if service is None:
        raise ValueError(f"Unsupported provider: {provider}")
    return service(index_name=index_name, namespace=namespace, dimension=dimension)
