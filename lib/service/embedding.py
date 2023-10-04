import requests

from numpy import ndarray
from typing import Any, Dict, List, Union
from prisma.models import Datasource
from tempfile import NamedTemporaryFile
from decouple import config
from sentence_transformers import SentenceTransformer

from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index import Document

from lib.service.vectordb import get_vector_service


class EmbeddingService:
    def __init__(self, datasource: Datasource):
        self.datasource = datasource

    def get_datasource_suffix(self) -> str:
        suffixes = {"TXT": ".txt", "PDF": ".pdf", "MARKDOWN": ".md"}
        try:
            return suffixes[self.datasource.type]
        except KeyError:
            raise ValueError("Unsupported datasource type")

    def generate_documents(self) -> List[Document]:
        with NamedTemporaryFile(
            suffix=self.get_datasource_suffix(), delete=True
        ) as temp_file:
            if self.datasource.url:
                content = requests.get(self.datasource.url).content
            else:
                content = self.datasource.content
            temp_file.write(content)
            temp_file.flush()
            reader = SimpleDirectoryReader(input_files=[temp_file.name])
            docs = reader.load_data()
            return docs

    def generate_chunks(self, documents: List[Document]) -> List[Union[Document, None]]:
        parser = SimpleNodeParser.from_defaults(chunk_size=350, chunk_overlap=20)
        nodes = parser.get_nodes_from_documents(documents, show_progress=True)
        return nodes

    # def generate_qa_pairs(self, nodes: List[Union[Document, None]]) -> Dict[str, Any]:
    #    qa_pairs = generate_qa_embedding_pairs(nodes=nodes)
    #    return qa_pairs

    def generate_embeddings(self, nodes: List[Union[Document, None]]) -> List[ndarray]:
        vectordb = get_vector_service(
            provider="pinecone",
            index_name="all-minilm-l6-v2",
            namespace=self.datasource.id,
            dimension=384,
        )
        model = SentenceTransformer(
            "all-MiniLM-L6-v2", use_auth_token=config("HF_API_KEY")
        )
        embeddings = []
        for node in nodes:
            if node is not None:
                embedding = (node.id_, model.encode(node.text).tolist(), node.metadata)
                embeddings.append(embedding)
        vectordb.upsert(vectors=embeddings)
        return embeddings
