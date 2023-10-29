from abc import ABC, abstractmethod
from typing import Callable

import replicate
from decouple import config

import litellm

from nagato.service.prompts import (
    generate_replicate_rag_prompt,
)


class QueryService(ABC):
    def __init__(
        self,
        provider: str,
        model: str,
    ):
        self.provider = provider
        self.model = model
        if self.provider == "REPLICATE":
            self.api_key = config("REPLICATE_API_KEY")
        elif self.provider == "OPENAI":
            self.api_key = config("OPENAI_API_KEY")
        else:
            self.api_key = None

    @abstractmethod
    def predict(
        self, input: str, enable_streaming: bool = False, callback: Callable = None
    ):
        pass

    @abstractmethod
    def predict_with_embedding(self, filter_id: str):
        pass


class ReplicateQueryService(QueryService):
    def __init__(
        self,
        provider: str,
        model: str,
    ):
        super().__init__(
            provider=provider,
            model=model,
        )

    def predict(
        self,
        input: str,
        enable_streaming: bool = False,
        system_prompt: str = None,
        callback: Callable = None,
    ):
        litellm.api_key = self.api_key

        output = litellm.completion(
            model=self.model,
            messages=[
                {"content": system_prompt, "role": "system"},
                {"content": input, "role": "user"},
            ],
            max_tokens=450,
            temperature=0,
            stream=enable_streaming,
        )
        if enable_streaming:
            for chunk in output:
                callback(chunk["choices"][0]["delta"]["content"])
        return output

    def predict_with_embedding(
        self,
        input: str,
        context: str,
        system_prompt: str,
        enable_streaming: bool = False,
        callback: Callable = None,
    ):
        litellm.api_key = self.api_key
        prompt = generate_replicate_rag_prompt(context=context, input=input)
        output = litellm.completion(
            model=self.model,
            messages=[
                {
                    "content": system_prompt,
                    "role": "system",
                },
                {
                    "content": prompt,
                    "role": "user",
                },
            ],
            max_tokens=2000,
            temperature=0,
            stream=enable_streaming,
        )
        if enable_streaming:
            for chunk in output:
                callback(chunk["choices"][0]["delta"]["content"])
        return output


def get_query_service(
    provider: str = "openai",
    model: str = "GPT_35_TURBO",
):
    services = {
        "REPLICATE": ReplicateQueryService,
        # Add other providers here
    }
    service = services.get(provider)
    if service is None:
        raise ValueError(f"Unsupported provider: {provider}")
    return service(
        provider=provider,
        model=model,
    )
