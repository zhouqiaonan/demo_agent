from typing import Any

from llm_client.base import BaseLLMClient


class ModelRouter:
    def __init__(self):
        self._clients: dict[str, BaseLLMClient] = {}

    def register(self, task_type: str, client: BaseLLMClient) -> None:
        self._clients[task_type] = client

    def unregister(self, task_type: str) -> None:
        self._clients.pop(task_type, None)

    def route(self, task_type: str) -> BaseLLMClient:
        if task_type not in self._clients:
            raise ValueError(f"No client registered for task type: {task_type}")
        return self._clients[task_type]

    def execute(
        self, task_type: str, messages: list[dict], **kwargs: Any
    ) -> dict:
        client = self.route(task_type)
        return client.chat_completion(messages, **kwargs)

    @property
    def registered_tasks(self) -> list[str]:
        return list(self._clients.keys())
