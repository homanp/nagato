from tempfile import NamedTemporaryFile
from typing import List, Union

import requests
from decouple import config
from llama_index import Document, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from numpy import ndarray
from sentence_transformers import SentenceTransformer

from nagato.service.vectordb import get_vector_service


class EmbeddingService:
    def __init__(self, type: str, url: str = None, content: str = None):
        self.type = type
        self.url = url
        self.content = content

    def get_datasource_suffix(self) -> str:
        suffixes = {"TXT": ".txt", "PDF": ".pdf", "MARKDOWN": ".md"}
        try:
            return suffixes[self.type]
        except KeyError:
            raise ValueError("Unsupported datasource type")

    def generate_documents(self) -> List[Document]:
        with NamedTemporaryFile(
            suffix=self.get_datasource_suffix(), delete=True
        ) as temp_file:
            if self.url:
                content = requests.get(self.url).content
            else:
                content = self.content
            temp_file.write(content)
            temp_file.flush()
            reader = SimpleDirectoryReader(input_files=[temp_file.name])
            docs = reader.load_data()
            return docs

    def generate_chunks(self, documents: List[Document]) -> List[Union[Document, None]]:
        parser = SimpleNodeParser.from_defaults(chunk_size=350, chunk_overlap=20)
        nodes = parser.get_nodes_from_documents(documents, show_progress=True)
        return nodes

    def generate_embeddings(
        self,
        nodes: List[Union[Document, None]],
        finetune_id: str,
    ) -> List[ndarray]:
        vectordb = get_vector_service(
            provider="pinecone",
            index_name="all-minilm-l6-v2",
            namespace=finetune_id,
            dimension=384,
        )
        model = SentenceTransformer(
            "all-MiniLM-L6-v2", use_auth_token=config("HF_API_KEY")
        )
        embeddings = []
        for node in nodes:
            if node is not None:
                embedding = (
                    node.id_,
                    model.encode(node.text).tolist(),
                    {**node.metadata, "content": node.text},
                )
                embeddings.append(embedding)
        vectordb.upsert(vectors=embeddings)
        return embeddings

    # def generate_query(self):
    #    model = SentenceTransformer(
    #        "all-MiniLM-L6-v2", use_auth_token=config("HF_API_KEY")
    #    )
    #    vectordb = get_vector_service(
    #        provider="pinecone",
    #        index_name="all-minilm-l6-v2",
    #        namespace=self.datasource.id,
    #        dimension=384,
    #    )
    #    query = "How many cars were sold?"
    #    embedding = model.encode([query]).tolist()
    #    return vectordb.query(queries=embedding, top_k=5, include_metadata=True)
