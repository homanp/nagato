from abc import ABC, abstractmethod
from typing import Callable

import replicate
from decouple import config

from nagato.service.prompts import (
    generate_replicaste_system_prompt,
)


class QueryService(ABC):
    def __init__(
        self,
        provider: str,
        model: str,
    ):
        self.provider = provider
        self.model = model

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
        client = replicate.Client(api_token=config("REPLICATE_API_KEY"))
        output = client.run(
            self.model,
            input={
                "prompt": input,
                "system_prompt": system_prompt,
            },
        )
        if enable_streaming:
            for item in output:
                callback(item)
        else:
            return "".join(item for item in output)

    def predict_with_embedding(
        self,
        input: str,
        context: str,
        enable_streaming: bool = False,
        callback: Callable = None,
        system_prompt: str = None,
    ):
        client = replicate.Client(api_token=config("REPLICATE_API_KEY"))
        output = client.run(
            self.model,
            input={
                "prompt": input,
                "system_prompt": generate_replicaste_system_prompt(
                    context=context, system_prompt=system_prompt
                ),
            },
        )
        if enable_streaming:
            for item in output:
                callback(item)
        else:
            return "".join(item for item in output)


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
