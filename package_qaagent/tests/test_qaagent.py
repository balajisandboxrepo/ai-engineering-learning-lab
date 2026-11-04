import os
import unittest
from pathlib import Path
from unittest.mock import patch

from qaagent import qaAgent


class OpenAIConfigurationTests(unittest.TestCase):
    def test_data_directories_are_anchored_to_the_package(self) -> None:
        package_dir = Path(qaAgent.__file__).resolve().parents[2]

        self.assertEqual(qaAgent.PACKAGE_DIR, package_dir)
        self.assertEqual(qaAgent.DOCS_DIR, package_dir / "docs")
        self.assertEqual(qaAgent.DB_DIR, package_dir / "db_openai")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("qaagent.qaAgent.load_dotenv")
    def test_api_key_is_loaded_from_environment(self, _load_dotenv) -> None:
        self.assertEqual(qaAgent._get_openai_api_key(), "test-key")

    @patch.dict(os.environ, {}, clear=True)
    @patch("qaagent.qaAgent.load_dotenv")
    def test_missing_api_key_fails_with_setup_instructions(self, _load_dotenv) -> None:
        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
            qaAgent._get_openai_api_key()

    @patch("qaagent.qaAgent.create_agent")
    @patch("qaagent.qaAgent.ChatOpenAI")
    @patch("qaagent.qaAgent._get_openai_api_key", return_value="test-key")
    def test_agent_uses_openai_chat_model(
        self, _get_api_key, chat_openai, create_agent
    ) -> None:
        qaAgent.build_agent(object())

        chat_openai.assert_called_once_with(
            model=qaAgent.CHAT_MODEL,
            temperature=0,
            api_key="test-key",
        )
        self.assertEqual(qaAgent.CHAT_MODEL, "gpt-4o-mini")
        create_agent.assert_called_once()

    @patch("qaagent.qaAgent.Chroma")
    @patch("qaagent.qaAgent.Path.exists", return_value=True)
    @patch("qaagent.qaAgent.OpenAIEmbeddings")
    @patch("qaagent.qaAgent._get_openai_api_key", return_value="test-key")
    def test_vectorstore_uses_openai_embeddings(
        self, _get_api_key, openai_embeddings, _path_exists, chroma
    ) -> None:
        qaAgent.get_vectorstore()

        openai_embeddings.assert_called_once_with(
            model=qaAgent.EMBED_MODEL,
            api_key="test-key",
        )
        self.assertEqual(qaAgent.EMBED_MODEL, "text-embedding-3-small")
        chroma.assert_called_once_with(
            persist_directory=str(qaAgent.DB_DIR),
            embedding_function=openai_embeddings.return_value,
        )
        self.assertEqual(qaAgent.DB_DIR.name, "db_openai")


if __name__ == "__main__":
    unittest.main()
