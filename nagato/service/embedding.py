from tempfile import NamedTemporaryFile
from typing import List, Union

import requests
from decouple import config
from llama_index import Document, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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
                response = requests.get(self.url, stream=True)
                total_size_in_bytes = int(response.headers.get("content-length", 0))
                block_size = 1024
                progress_bar = tqdm(
                    total=total_size_in_bytes,
                    desc="Downloading file",
                    unit="iB",
                    unit_scale=True,
                )
                content = b""
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    content += data
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
            else:
                content = self.content
            temp_file.write(content)
            temp_file.flush()

            with tqdm(total=3, desc="Processing data") as pbar:
                pbar.update()
                pbar.set_description("Analyzing data")
                reader = SimpleDirectoryReader(input_files=[temp_file.name])
                pbar.update()
                pbar.set_description("Generating documents")
                docs = reader.load_data()
                pbar.update()
                pbar.set_description("Documents generated")

            return docs

    def generate_chunks(self, documents: List[Document]) -> List[Union[Document, None]]:
        parser = SimpleNodeParser.from_defaults(chunk_size=350, chunk_overlap=20)
        with tqdm(total=2, desc="Generating chunks") as pbar:
            pbar.update()
            pbar.set_description("Generating nodes")
            nodes = parser.get_nodes_from_documents(documents, show_progress=True)
            pbar.update()
        return nodes

    def generate_embeddings(
        self,
        nodes: List[Union[Document, None]],
        filter_id: str,
    ) -> List[ndarray]:
        vectordb = get_vector_service(
            provider="pinecone",
            index_name="all-minilm-l6-v2",
            filter_id=filter_id,
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
