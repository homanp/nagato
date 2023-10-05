import json
import openai
import asyncio

from abc import ABC, abstractmethod
from typing import List, Union, Tuple
from numpy import ndarray
from decouple import config
from llama_index import Document
from lib.service.prompts import generate_qa_pair_prompt, GPT_DATA_FORMAT

openai.api_key = config("OPENAI_API_KEY")


class FinetuningService(ABC):
    def __init__(self, nodes: List[Union[Document, None]]):
        self.nodes = nodes

    @abstractmethod
    async def generate_dataset(self) -> List[Tuple[str, ndarray]]:
        pass


class OpenAIFinetuningService(FinetuningService):
    def __init__(
        self, nodes: List[Union[Document, None]], num_questions_per_chunk: int = 10
    ):
        super().__init__(nodes=nodes)
        self.num_questions_per_chunk = num_questions_per_chunk

    async def generate_prompt_and_completion(self, node):
        prompt = generate_qa_pair_prompt(
            context=node.text, num_of_qa_paris=10, format=GPT_DATA_FORMAT
        )
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

    async def generate_dataset(self):
        with open("dataset.jsonl", "w") as f:
            for i in range(0, len(self.nodes), 10):  # Process nodes in chunks of 10
                tasks = [
                    self.generate_prompt_and_completion(node)
                    for node in self.nodes[i : i + 10]
                ]
                results = await asyncio.gather(*tasks)
                for data in results:
                    json.dump(data, f)
                    f.write("\n")


async def get_finetuning_service(
    nodes: List[Union[Document, None]],
    provider: str = "openai",
    num_questions_per_chunk: int = 10,
):
    services = {
        "openai": OpenAIFinetuningService,
        # Add other providers here
    }
    service = services.get(provider)
    if service is None:
        raise ValueError(f"Unsupported provider: {provider}")
    return service(nodes=nodes, num_questions_per_chunk=num_questions_per_chunk)
