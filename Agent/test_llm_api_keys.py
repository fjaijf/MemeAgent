from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from memeagent.config import MemeAgentConfig
from memeagent.llm import _openai_compatible_api_key, _zai_api_key


def _config(**overrides: object) -> MemeAgentConfig:
    values = {
        "provider": "openai-compatible",
        "model": "qwen3.7-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "temperature": 0.2,
        "timeout": 60.0,
        "max_tokens": None,
        "max_retries": 0,
        "controller_provider": None,
        "controller_model": "qwen3.7-plus",
        "controller_temperature": 0.2,
        "controller_timeout": 60.0,
        "controller_max_tokens": None,
        "controller_max_retries": 0,
        "controller_thinking_enabled": True,
        "search_enabled": True,
        "search_provider": "ddgs",
        "search_api_key": None,
        "tavily_api_key": None,
        "zhihu_api_key": None,
        "anspire_api_key": None,
        "glm_search_api_key": None,
        "glm_search_engine": "search_pro",
        "glm_search_recency_filter": "noLimit",
        "glm_search_content_size": "medium",
        "glm_search_domain_filter": None,
        "search_proxy": None,
        "context_proxy": None,
        "search_max_results": 5,
        "news_max_results": 5,
        "search_timeout": 12.0,
        "search_country": "us",
        "search_lang": "en",
        "search_context_sites": "reddit.com,x.com",
        "tavily_search_depth": "basic",
        "cache_enabled": False,
        "cache_dir": ".memeagent_cache",
        "search_cache_ttl_seconds": 7 * 24 * 60 * 60,
        "news_cache_ttl_seconds": 6 * 60 * 60,
        "trajectory_cache_enabled": True,
        "trajectory_cache_dir": ".memeagent_cache",
        "memory_enabled": True,
        "memory_dir": ".memeagent_memory",
        "memory_recall_limit": 3,
        "system_prompt": "test",
    }
    values.update(overrides)
    return MemeAgentConfig(**values)


class LlmApiKeyTests(unittest.TestCase):
    def test_qwen_compatible_config_prefers_dashscope_key(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
                "DASHSCOPE_API_KEY": "dashscope-key",
            },
            clear=True,
        ):
            self.assertEqual("dashscope-key", _openai_compatible_api_key(_config()))

    def test_qwen_compatible_config_accepts_qwen_key(self) -> None:
        with patch.dict(os.environ, {"QWEN_API_KEY": "qwen-key"}, clear=True):
            self.assertEqual("qwen-key", _openai_compatible_api_key(_config()))

    def test_glm_key_accepts_memeagent_alias(self) -> None:
        with patch.dict(os.environ, {"MEMEAGENT_GLM_API_KEY": "glm-key"}, clear=True):
            self.assertEqual("glm-key", _zai_api_key())


if __name__ == "__main__":
    unittest.main()
