from unittest.mock import MagicMock, patch

from llm_client.deepseek_client import DeepSeekClient


class TestDeepSeekClient:
    def test_chat_completion_returns_normalized_dict(self):
        mock_response = MagicMock()
        mock_response.model = "deepseek-chat"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 20
        mock_response.choices[0].message.content = "Hi there!"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].finish_reason = "stop"

        client = DeepSeekClient(api_key="sk-test")
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = mock_response

        result = client.chat_completion(
            [{"role": "user", "content": "Hello"}]
        )

        assert result["content"] == "Hi there!"
        assert result["model"] == "deepseek-chat"
        assert result["usage"]["total_tokens"] == 20

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-env-key"}):
            client = DeepSeekClient()
            assert client.api_key == "sk-env-key"

    def test_base_url_is_deepseek(self):
        client = DeepSeekClient(api_key="sk-test")
        assert client.base_url == "https://api.deepseek.com"
