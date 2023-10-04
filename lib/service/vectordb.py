import pinecone

from numpy import ndarray
from typing import List
from abc import ABC, abstractmethod
from decouple import config


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
        # Create a new vector index if it doesn't exist dimensions should be passed in the arguments
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name, metric="cosine", shards=1, dimension=dimension
            )
        self.index = pinecone.Index(index_name=self.index_name)

    def __del__(self):
        pinecone.deinit()

    def upsert(self, vectors: ndarray):
        self.index.upsert(vectors=vectors, namespace=self.namespace)

    def query(self, queries: List[ndarray], top_k: int):
        return self.index.query(queries=queries, top_k=top_k)


def get_vector_service(
    provider: str, index_name: str, namespace: str = None, dimension: int = 384
):
    services = {
        "pinecone": PineconeVectorService,
        # Add other providers here
        # "weaviate": WeaviateVectorService,
    }
    service = services.get(provider)
    if service is None:
        raise ValueError(f"Unsupported provider: {provider}")
    return service(index_name=index_name, namespace=namespace, dimension=dimension)
