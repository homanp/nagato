from abc import ABC, abstractmethod
from typing import List

import pinecone
from decouple import config
from numpy import ndarray


class VectorDBService(ABC):
    def __init__(self, index_name: str, dimension: int, filter_id: str = None):
        self.index_name = index_name
        self.filter_id = filter_id
        self.dimension = dimension

    @abstractmethod
    def upsert():
        pass

    @abstractmethod
    def query():
        pass


class PineconeVectorService(VectorDBService):
    def __init__(self, index_name: str, dimension: int, filter_id: str = None):
        super().__init__(
            index_name=index_name, dimension=dimension, filter_id=filter_id
        )
        pinecone.init(
            api_key=config("PINECONE_API_KEY"),
            environment=config("PINECONE_ENVIRONMENT"),
        )
        # Create a new vector index if it doesn't
        # exist dimensions should be passed in the arguments
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name, metric="cosine", shards=1, dimension=dimension
            )
        self.index = pinecone.Index(index_name=self.index_name)

    def upsert(self, vectors: ndarray):
        self.index.upsert(vectors=vectors, namespace=self.filter_id)

    def query(self, queries: List[ndarray], top_k: int, include_metadata: bool = True):
        return self.index.query(
            queries=queries,
            top_k=top_k,
            include_metadata=include_metadata,
            namespace=self.filter_id,
        )

    def rerank(self, data: list, query: str) -> dict:
        from rank_bm25 import BM25Okapi

        docs = [match["metadata"]["content"] for match in data]
        tokenized_docs = [doc.split(" ") for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        for idx, match in enumerate(data[0]["matches"]):
            match["bm25_score"] = doc_scores[idx]
        data[0]["matches"] = sorted(
            data[0]["matches"], key=lambda x: x["bm25_score"], reverse=True
        )
        return data


def get_vector_service(
    provider: str, index_name: str, filter_id: str = None, dimension: int = 384
):
    services = {
        "PINECONE": PineconeVectorService,
        # Add other providers here
        # e.g "weaviate": WeaviateVectorService,
    }
    service = services.get(provider)
    if service is None:
        raise ValueError(f"Unsupported provider: {provider}")
    return service(index_name=index_name, filter_id=filter_id, dimension=dimension)
