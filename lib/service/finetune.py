import asyncio
import os
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import openai
from decouple import config
from llama_index import Document
from numpy import ndarray

from lib.service.prompts import GPT_DATA_FORMAT, generate_qa_pair_prompt

openai.api_key = config("OPENAI_API_KEY")


class FinetuningService(ABC):
    def __init__(self, nodes: List[Union[Document, None]]):
        self.nodes = nodes

    @abstractmethod
    async def generate_dataset(self) -> List[Tuple[str, ndarray]]:
        pass

    @abstractmethod
    async def finetune(self, training_file: str) -> Dict:
        pass

    @abstractmethod
    async def cleanup(self, training_file: str) -> None:
        os.remove(training_file)


class OpenAIFinetuningService(FinetuningService):
    def __init__(
        self,
        nodes: List[Union[Document, None]],
        num_questions_per_chunk: int = 10,
        batch_size: int = 10,
    ):
        super().__init__(nodes=nodes)
        self.num_questions_per_chunk = num_questions_per_chunk
        self.batch_size = batch_size

    async def generate_prompt_and_completion(self, node):
        prompt = generate_qa_pair_prompt(
            context=node.text, num_of_qa_paris=10, format=GPT_DATA_FORMAT
        )
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return completion.choices[0].message.content

    async def generate_dataset(self) -> str:
        training_file = f"{uuid.uuid4()}.jsonl"
        with open(training_file, "w") as f:
            for i in range(
                0, len(self.nodes), self.batch_size
            ):  # Process nodes in chunks of batch_size
                tasks = [
                    self.generate_prompt_and_completion(node)
                    for node in self.nodes[i : i + self.batch_size]
                ]
                qa_pairs = await asyncio.gather(*tasks)
                for qa_pair in qa_pairs:
                    json_objects = qa_pair.split("\n\n")
                    for json_obj in json_objects:
                        f.write(json_obj + "\n")

    async def finetune(self, training_file: str) -> Dict:
        file = openai.File.create(file=open(training_file, "rb"), purpose="fine-tune")
        finetune = await openai.FineTuningJob.acreate(
            training_file=file.get("id"), model="gpt-3.5-turbo"
        )
        return {**finetune, "training_file": training_file}


async def get_finetuning_service(
    nodes: List[Union[Document, None]],
    provider: str = "openai",
    num_questions_per_chunk: int = 10,
    batch_size: int = 10,
):
    services = {
        "openai": OpenAIFinetuningService,
        # Add other providers here
    }
    service = services.get(provider)
    if service is None:
        raise ValueError(f"Unsupported provider: {provider}")
    return service(
        nodes=nodes,
        num_questions_per_chunk=num_questions_per_chunk,
        batch_size=batch_size,
    )
