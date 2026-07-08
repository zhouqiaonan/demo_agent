from unittest.mock import MagicMock, patch

from llm_client.openai_client import OpenAIClient


class TestOpenAIClient:
    def test_chat_completion_returns_normalized_dict(self):
        mock_response = MagicMock()
        mock_response.model = "gpt-4o"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"

        client = OpenAIClient(api_key="sk-test")
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = mock_response

        result = client.chat_completion(
            [{"role": "user", "content": "Hi"}]
        )

        assert result["content"] == "Hello!"
        assert result["role"] == "assistant"
        assert result["model"] == "gpt-4o"
        assert result["usage"]["total_tokens"] == 30
        assert result["finish_reason"] == "stop"

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-env"}):
            client = OpenAIClient()
            assert client.api_key == "sk-from-env"

    def test_api_key_explicit_overrides_env(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-env"}):
            client = OpenAIClient(api_key="sk-explicit")
            assert client.api_key == "sk-explicit"
