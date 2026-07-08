import os

from openai import OpenAI

from llm_client.base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gpt-4o",
    ):
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        super().__init__(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            model_name=model_name,
        )
        self._client = OpenAI(api_key=self.api_key)

    def chat_completion(self, messages: list[dict], **kwargs) -> dict:
        model = kwargs.get("model", self.model_name)
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            **{k: v for k, v in kwargs.items() if k != "model"},
        )
        return self._normalize(response)

    def _normalize(self, response) -> dict:
        choice = response.choices[0]
        return {
            "content": choice.message.content or "",
            "role": choice.message.role,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            "finish_reason": choice.finish_reason,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (choice.message.tool_calls or [])
            ],
        }
