# flake8: noqa

import sys
import requests
import json
import os
import uuid
import openai
import replicate
import concurrent.futures

from tqdm import tqdm
from decouple import config
from abc import ABC, abstractmethod
from typing import Dict, List, Union
from concurrent.futures import ThreadPoolExecutor

from nagato.utils.logger import logger
from decouple import config
from llama_index import Document

from nagato.service.prompts import (
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
    def __init__(
        self,
        nodes: List[Union[Document, None]],
        num_questions_per_chunk: int,
        batch_size: int,
    ):
        self.nodes = nodes
        self.num_questions_per_chunk = num_questions_per_chunk
        self.batch_size = batch_size

    @abstractmethod
    def generate_prompt_and_completion(self, node):
        pass

    @abstractmethod
    def validate_dataset(self, training_file: str) -> str:
        pass

    @abstractmethod
    def finetune(self, training_file: str, base_model: str) -> Dict:
        pass

    def generate_dataset(self) -> str:
        training_file = f"{uuid.uuid4()}.jsonl"
        total_pairs = len(self.nodes) * self.num_questions_per_chunk
        with open(training_file, "w") as f:
            with ThreadPoolExecutor() as executor:
                progress_bar = tqdm(
                    total=total_pairs,
                    desc="Generating synthetic Q&A pairs",
                    file=sys.stdout,
                )
                for i in range(
                    0, len(self.nodes), self.batch_size
                ):  # Process nodes in chunks of batch_size
                    batch_nodes = self.nodes[i : i + self.batch_size]
                    tasks = [
                        executor.submit(self.generate_prompt_and_completion, node)
                        for node in batch_nodes
                    ]
                    for future in concurrent.futures.as_completed(tasks):
                        qa_pair = future.result()
                        json_objects = qa_pair.split("\n\n")
                        for json_obj in json_objects:
                            f.write(json_obj + "\n")
                            progress_bar.update(1)
                progress_bar.close()
        return training_file

    def cleanup(self, training_file: str) -> None:
        os.remove(training_file)


class OpenAIFinetuningService(FinetuningService):
    def __init__(
        self,
        nodes: List[Union[Document, None]],
        num_questions_per_chunk: int,
        batch_size: int,
        base_model: str = "GPT_35_TURBO",
    ):
        super().__init__(
            nodes=nodes,
            num_questions_per_chunk=num_questions_per_chunk,
            batch_size=batch_size,
        )
        self.base_model = base_model

    def generate_prompt_and_completion(self, node):
        prompt = generate_qa_pair_prompt(
            context=node.text, num_of_qa_paris=10, format=GPT_DATA_FORMAT
        )
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return completion.choices[0].message.content

    def validate_dataset(self, training_file: str) -> str:
        pass

    def finetune(self, training_file: str, _webhook_url: str = None) -> Dict:
        file = openai.File.create(file=open(training_file, "rb"), purpose="fine-tune")
        finetune = openai.FineTuningJob.create(
            training_file=file.get("id"), model=OPENAI_MODELS[self.base_model]
        )
        return {**finetune, "training_file": training_file}


class ReplicateFinetuningService(FinetuningService):
    def __init__(
        self,
        nodes: List[Union[Document, None]],
        num_questions_per_chunk: int,
        batch_size: int,
        base_model: str = "LLAMA2_7B_CHAT",
    ):
        super().__init__(
            nodes=nodes,
            num_questions_per_chunk=num_questions_per_chunk,
            batch_size=batch_size,
        )
        self.base_model = base_model

    def generate_prompt_and_completion(self, node):
        prompt = generate_qa_pair_prompt(
            context=node.text,
            num_of_qa_pairs=self.num_questions_per_chunk,
            format=REPLICATE_FORMAT,
        )
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return completion.choices[0].message.content

    def validate_dataset(self, training_file: str) -> str:
        valid_data = []
        with open(training_file, "r") as f:
            lines = f.readlines()
            total_lines = len(lines)
            progress_bar = tqdm(
                total=total_lines,
                desc="Validating lines",
                file=sys.stdout,
            )
            for i, line in enumerate(lines, start=1):
                try:
                    data = json.loads(line)
                    if "prompt" in data and "completion" in data:
                        valid_data.append(data)
                except json.JSONDecodeError:
                    pass
                progress_bar.update(1)
            progress_bar.close()

        with open(training_file, "w") as f:
            for data in valid_data:
                f.write(json.dumps(data) + "\n")

        return training_file

    def finetune(self, training_file: str, webhook_url: str = None) -> Dict:
        training_file_url = upload_replicate_dataset(training_file=training_file)
        training = replicate.Client(
            api_token=config("REPLICATE_API_KEY")
        ).trainings.create(
            version=REPLICATE_MODELS[self.base_model],
            input={
                "train_data": training_file_url,
                "num_train_epochs": 6,
            },
            destination="homanp/test",
            webhook=webhook_url,
        )
        return {"id": training.id, "training_file": training_file}


def get_finetuning_service(
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


def upload_replicate_dataset(training_file: str) -> str:
    headers = {"Authorization": f"Token {config('REPLICATE_API_KEY')}"}
    upload_response = requests.post(
        "https://dreambooth-api-experimental.replicate.com/v1/upload/data.jsonl",
        headers=headers,
    )
    upload_response_data = upload_response.json()
    upload_url = upload_response_data["upload_url"]

    with open(training_file, "rb") as f:
        requests.put(
            upload_url,
            headers={"Content-Type": "application/jsonl"},
            data=f.read(),
        )

    serving_url = upload_response_data["serving_url"]
    return serving_url
