# flake8: noqa

import asyncio
import json
import os
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import httpx
import openai
import replicate
from decouple import config
from llama_index import Document

from lib.service.prompts import (
    GPT_DATA_FORMAT,
    REPLICATE_FORMAT,
    generate_qa_pair_prompt,
)

openai.api_key = config("OPENAI_API_KEY")

REPLICATE_MODELS = {
    "LLAMA2_7B_CHAT": "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e",
    "LLAMA2_7B": "meta/llama-2-7b:527827021d8756c7ab79fde0abbfaac885c37a3ed5fe23c7465093f0878d55ef",
    "LLAMA2_13B_CHAT": "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
    "LLAMA2_13B": "meta/llama-2-13b:078d7a002387bd96d93b0302a4c03b3f15824b63104034bfa943c63a8f208c38",
    "LLAMA2_70B_CHAT": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "LLAMA2_70B": "meta/llama-2-70b:a52e56fee2269a78c9279800ec88898cecb6c8f1df22a6483132bea266648f00",
    "GPT_J_6B": "replicate/gpt-j-6b:b3546aeec6c9891f0dd9929c2d3bedbf013c12e02e7dd0346af09c37e008c827",
    "DOLLY_V2_12B": "replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5",
}

OPENAI_MODELS = {"GPT_35_TURBO": "gpt-3.5-turbo"}


class FinetuningService(ABC):
    def __init__(self, nodes: List[Union[Document, None]]):
        self.nodes = nodes

    @abstractmethod
    async def generate_dataset(self) -> str:
        pass

    @abstractmethod
    async def validate_dataset(self, training_file: str) -> str:
        pass

    @abstractmethod
    async def finetune(self, training_file: str, base_model: str) -> Dict:
        pass

    async def cleanup(self, training_file: str) -> None:
        os.remove(training_file)


class OpenAIFinetuningService(FinetuningService):
    def __init__(
        self,
        nodes: List[Union[Document, None]],
        num_questions_per_chunk: int = 10,
        batch_size: int = 10,
        base_model: str = "GPT_35_TURBO",
    ):
        super().__init__(nodes=nodes)
        self.num_questions_per_chunk = num_questions_per_chunk
        self.batch_size = batch_size
        self.base_model = base_model

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
        return training_file

    async def validate_dataset(self, training_file: str) -> str:
        pass

    async def finetune(self, training_file: str) -> Dict:
        file = openai.File.create(file=open(training_file, "rb"), purpose="fine-tune")
        finetune = await openai.FineTuningJob.acreate(
            training_file=file.get("id"), model=OPENAI_MODELS[self.base_model]
        )
        return {**finetune, "training_file": training_file}


class ReplicateFinetuningService(FinetuningService):
    def __init__(
        self,
        nodes: List[Union[Document, None]],
        num_questions_per_chunk: int = 1,
        batch_size: int = 10,
        base_model: str = "LLAMA2_7B_CHAT",
    ):
        super().__init__(nodes=nodes)
        self.num_questions_per_chunk = num_questions_per_chunk
        self.batch_size = batch_size
        self.base_model = base_model

    async def generate_prompt_and_completion(self, node):
        prompt = generate_qa_pair_prompt(
            context=node.text, num_of_qa_paris=10, format=REPLICATE_FORMAT
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
        return training_file

    async def validate_dataset(self, training_file: str) -> str:
        valid_data = []
        with open(training_file, "r") as f:
            for line in f:
                data = json.loads(line)
                if "prompt" in data and "completion" in data:
                    valid_data.append(data)
        with open(training_file, "w") as f:
            for data in valid_data:
                f.write(json.dumps(data) + "\n")
        return training_file

    async def finetune(self, training_file: str) -> Dict:
        training_file_url = await upload_replicate_dataset(training_file=training_file)
        training = replicate.trainings.create(
            version=REPLICATE_MODELS[self.base_model],
            input={
                "train_data": training_file_url,
                "num_train_epochs": 6,
            },
            destination="homanp/test",
            webhook="https://api.nagato.sh/api/v1/webhook/finetune",
        )
        return {"id": training.id, "training_file": training_file}


async def get_finetuning_service(
    nodes: List[Union[Document, None]],
    provider: str = "openai",
    base_model: str = "GPT_35_TURBO",
    num_questions_per_chunk: int = 10,
    batch_size: int = 10,
):
    services = {
        "OPENAI": OpenAIFinetuningService,
        "REPLICATE": ReplicateFinetuningService,
        # Add other providers here
    }
    service = services.get(provider)
    if service is None:
        raise ValueError(f"Unsupported provider: {provider}")
    return service(
        nodes=nodes,
        num_questions_per_chunk=num_questions_per_chunk,
        batch_size=batch_size,
        base_model=base_model,
    )


async def upload_replicate_dataset(training_file: str) -> str:
    headers = {"Authorization": f"Token {config('REPLICATE_API_TOKEN')}"}
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://dreambooth-api-experimental.replicate.com/v1/upload/data.jsonl",
            headers=headers,
        )
        response_data = response.json()
        upload_url = response_data["upload_url"]

        with open(training_file, "rb") as f:
            await client.put(
                upload_url,
                headers={"Content-Type": "application/jsonl"},
                content=f.read(),
            )

        serving_url = response_data["serving_url"]
        return serving_url
