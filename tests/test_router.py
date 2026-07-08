import pytest

from llm_client.openai_client import OpenAIClient
from llm_client.deepseek_client import DeepSeekClient
from llm_client.router import ModelRouter


class TestModelRouter:
    def test_register_and_route(self):
        router = ModelRouter()
        openai_client = OpenAIClient(api_key="sk-test")
        deepseek_client = DeepSeekClient(api_key="sk-test")

        router.register("chat", openai_client)
        router.register("coding", deepseek_client)

        assert router.route("chat") is openai_client
        assert router.route("coding") is deepseek_client

    def test_route_unregistered_task_raises_error(self):
        router = ModelRouter()
        with pytest.raises(ValueError, match="No client registered"):
            router.route("nonexistent")

    def test_unregister_removes_client(self):
        router = ModelRouter()
        client = OpenAIClient(api_key="sk-test")
        router.register("chat", client)
        router.unregister("chat")

        with pytest.raises(ValueError):
            router.route("chat")

    def test_registered_tasks_property(self):
        router = ModelRouter()
        router.register("chat", OpenAIClient(api_key="sk-test"))
        router.register("coding", DeepSeekClient(api_key="sk-test"))

        assert set(router.registered_tasks) == {"chat", "coding"}

    def test_execute_calls_chat_completion(self):
        from unittest.mock import MagicMock

        router = ModelRouter()
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = {"content": "result"}

        router.register("task", mock_client)
        result = router.execute("task", [{"role": "user", "content": "hi"}])

        assert result == {"content": "result"}
        mock_client.chat_completion.assert_called_once()
