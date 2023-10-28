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

MODEL_TO_INDEX = {
    "all-MiniLM-L6-v2": {"index_name": "all-minilm-l6-v2", "dimensions": 384},
    "thenlper/gte-base": {"index_name": "gte-base", "dimensions": 768},
    "thenlper/gte-small": {"index_name": "gte-small", "dimensions": 384},
    "thenlper/gte-large": {"index_name": "gte-large", "dimensions": 1024},
    "infgrad/stella-base-en-v2": {"index_name": "stella-base", "dimensions": 768}
    # Add more mappings here as needed
}


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
                content = b""
                with tqdm(
                    total=total_size_in_bytes,
                    desc="🟠 Downloading file",
                    unit="iB",
                    unit_scale=True,
                ) as progress_bar:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        content += data
                    if (
                        total_size_in_bytes != 0
                        and progress_bar.n != total_size_in_bytes
                    ):
                        print("ERROR, something went wrong")
                    else:
                        progress_bar.set_description("🟢 Downloading file")
            else:
                content = self.content
            temp_file.write(content)
            temp_file.flush()

            with tqdm(total=1, desc="🟠 Processing data") as pbar:
                reader = SimpleDirectoryReader(input_files=[temp_file.name])
                docs = reader.load_data()
                pbar.update()
                pbar.set_description("🟢 Processing data")

            return docs

    def generate_chunks(self, documents: List[Document]) -> List[Union[Document, None]]:
        parser = SimpleNodeParser.from_defaults(chunk_size=350, chunk_overlap=20)
        with tqdm(total=1, desc="🟠 Generating chunks") as pbar:
            nodes = parser.get_nodes_from_documents(documents, show_progress=False)
            pbar.update()
            pbar.set_description("🟢 Generating chunks")
        return nodes

    def generate_embeddings(
        self,
        nodes: List[Union[Document, None]],
        filter_id: str,
        model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "PINECONE",
    ) -> List[ndarray]:
        vectordb = get_vector_service(
            provider=embedding_provider,
            index_name=MODEL_TO_INDEX[model].get("index_name"),
            filter_id=filter_id,
            dimension=MODEL_TO_INDEX[model].get("dimensions"),
        )
        model = SentenceTransformer(model, use_auth_token=config("HF_API_KEY"))
        embeddings = []
        with tqdm(total=len(nodes), desc="🟠 Generating embeddings") as pbar:
            for node in nodes:
                if node is not None:
                    embedding = (
                        node.id_,
                        model.encode(node.text).tolist(),
                        {**node.metadata, "content": node.text},
                    )
                    embeddings.append(embedding)
                pbar.update()
            vectordb.upsert(vectors=embeddings)
            pbar.set_description("🟢 Generating embeddings")

        return embeddings
