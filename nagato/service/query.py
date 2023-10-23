from abc import ABC, abstractmethod
from typing import Callable

import replicate
from decouple import config

from nagato.service.prompts import REPLICATE_SYSTEM_PROMPT


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
        self, input: str, enable_streaming: bool = False, callback: Callable = None
    ):
        client = replicate.Client(api_token=config("REPLICATE_API_KEY"))
        output = client.run(
            self.model,
            input={
                "prompt": input,
                "system_prompt": REPLICATE_SYSTEM_PROMPT,
            },
        )
        if enable_streaming:
            for item in output:
                callback(item)
        else:
            return "".join(item for item in output)

    def predict_with_embedding(self, filter_id: str):
        print(filter_id)


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
