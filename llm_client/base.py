from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMClient(ABC):
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

    @abstractmethod
    def chat_completion(self, messages: list[dict], **kwargs) -> dict:
        pass

    def stream_completion(self, messages: list[dict], **kwargs):
        raise NotImplementedError
