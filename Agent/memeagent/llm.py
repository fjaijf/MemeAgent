from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI
import requests

from .config import MemeAgentConfig


_LOCAL_TRANSFORMERS_PROVIDERS = {
    "local",
    "local-transformers",
    "transformers",
    "hf",
    "huggingface",
}


@dataclass(frozen=True)
class ChatResponse:
    content: str


class OpenAICompatibleChatClient:
    """Small OpenAI SDK wrapper with the invoke() shape used by MemeAgent."""

    def __init__(
        self,
        config: MemeAgentConfig,
        api_key: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
        max_retries: int | None = None,
        enable_thinking: bool | None = None,
        strip_thinking: bool = True,
    ) -> None:
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": config.timeout if timeout is None else timeout,
            "max_retries": config.max_retries if max_retries is None else max_retries,
        }
        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model or config.model
        self.temperature = config.temperature if temperature is None else temperature
        self.max_tokens = config.max_tokens if max_tokens is None else max_tokens
        self.base_url = config.base_url
        self.enable_thinking = self._resolve_enable_thinking(enable_thinking)
        self.strip_thinking = strip_thinking

    def invoke(self, messages: list[Any]) -> ChatResponse:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [self._convert_message(message) for message in messages],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens
        self._add_thinking_options(request_kwargs)

        response = self.client.chat.completions.create(**request_kwargs)
        content = self._postprocess_content(response.choices[0].message.content or "")
        return ChatResponse(content=content)

    def stream(self, messages: list[Any]):
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [self._convert_message(message) for message in messages],
            "temperature": self.temperature,
            "stream": True,
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens
        self._add_thinking_options(request_kwargs)

        for chunk in self.client.chat.completions.create(**request_kwargs):
            if not getattr(chunk, "choices", None):
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                yield content

    def _resolve_enable_thinking(self, configured: bool | None) -> bool | None:
        env_value = _env_optional_flag("MEMEAGENT_OPENAI_COMPATIBLE_THINKING")
        if env_value is not None:
            return env_value
        if not _supports_openai_compatible_thinking_options(self.base_url):
            return None
        if _is_loopback_base_url(self.base_url):
            if configured is not None:
                return configured
            return _env_optional_flag("MEMEAGENT_LOCAL_THINKING", False)
        if configured is not None:
            return configured
        return None

    def _add_thinking_options(self, request_kwargs: dict[str, Any]) -> None:
        if self.enable_thinking is None:
            return
        if _is_dashscope_base_url(self.base_url):
            request_kwargs["extra_body"] = {"enable_thinking": self.enable_thinking}
            return
        request_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

    def _postprocess_content(self, content: str) -> str:
        if self.strip_thinking:
            content = re.sub(r"(?s)<think>.*?</think>\s*", "", content)
        return content.strip()

    def _convert_message(self, message: Any) -> dict[str, Any]:
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}

        role = getattr(message, "type", None) or getattr(message, "role", "user")
        content = getattr(message, "content", message)
        if role == "human":
            role = "user"
        if role == "ai":
            role = "assistant"
        return {"role": role, "content": content}


class ZhipuAIChatClient:
    """Small Zhipu GLM HTTP wrapper with the invoke() shape used by MemeAgent."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int | None,
        timeout: float,
        thinking_enabled: bool = True,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.thinking_enabled = thinking_enabled
        self.base_url = (
            base_url
            or os.getenv("ZAI_BASE_URL")
            or os.getenv("ZHIPUAI_BASE_URL")
            or "https://open.bigmodel.cn/api/paas/v4"
        ).rstrip("/")

    def invoke(self, messages: list[Any]) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._convert_message(message) for message in messages],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.thinking_enabled:
            payload["thinking"] = {"type": "enabled"}

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ChatResponse(content="")
        message = choices[0].get("message") or {}
        return ChatResponse(content=str(message.get("content") or "").strip())

    def _convert_message(self, message: Any) -> dict[str, Any]:
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": self._text_only_content(message.content)}

        role = getattr(message, "type", None) or getattr(message, "role", "user")
        content = getattr(message, "content", message)
        if role == "human":
            role = "user"
        if role == "ai":
            role = "assistant"
        return {"role": role, "content": self._text_only_content(content)}

    def _text_only_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(part for part in parts if part)
        return str(content)


class LocalTransformersChatClient:
    """Local Hugging Face Transformers chat client with optional vision inputs."""

    def __init__(
        self,
        *,
        model_path: str,
        temperature: float,
        max_tokens: int | None,
        timeout: float,
        enable_thinking: bool | None = None,
        strip_thinking: bool = True,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser())
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self.strip_thinking = strip_thinking
        self.device_map = os.getenv("MEMEAGENT_LOCAL_DEVICE_MAP", "auto").strip() or "auto"
        self.torch_dtype = os.getenv("MEMEAGENT_LOCAL_TORCH_DTYPE", "auto").strip() or "auto"
        self.trust_remote_code = _env_flag("MEMEAGENT_LOCAL_TRUST_REMOTE_CODE", True)
        self.default_max_new_tokens = int(
            os.getenv("MEMEAGENT_LOCAL_MAX_NEW_TOKENS", "2048").strip() or "2048"
        )
        self.top_p = _env_optional_float("MEMEAGENT_LOCAL_TOP_P")
        self.top_k = _env_optional_int("MEMEAGENT_LOCAL_TOP_K")
        self.min_p = _env_optional_float("MEMEAGENT_LOCAL_MIN_P")

        self._torch: Any | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._processor: Any | None = None
        self._is_multimodal = False

    def invoke(self, messages: list[Any]) -> ChatResponse:
        self._ensure_loaded()
        hf_messages, images = self._convert_messages(
            messages,
            include_images=self._is_multimodal,
        )
        if self._is_multimodal:
            return ChatResponse(content=self._invoke_multimodal(hf_messages, images))
        return ChatResponse(content=self._invoke_text(hf_messages))

    def stream(self, messages: list[Any]):
        yield self.invoke(messages).content

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            import transformers
            from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Local Transformers mode requires torch, torchvision, transformers, "
                "accelerate, and pillow. Install the local extras, for example: "
                "pip install -e .[local]"
            ) from exc

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Local model path does not exist: {self.model_path}")

        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self._is_multimodal = bool(
            getattr(config, "vision_config", None)
            or getattr(config, "image_token_id", None) is not None
            or getattr(config, "language_model_only", True) is False
        )

        model_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.torch_dtype:
            model_kwargs["torch_dtype"] = self.torch_dtype

        if self._is_multimodal:
            model_cls = (
                getattr(transformers, "AutoModelForImageTextToText", None)
                or getattr(transformers, "AutoModelForVision2Seq", None)
                or AutoModelForCausalLM
            )
            self._processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
            )
            self._tokenizer = getattr(self._processor, "tokenizer", None)
        else:
            model_cls = AutoModelForCausalLM
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
            )

        self._model = model_cls.from_pretrained(self.model_path, **model_kwargs)
        self._model.eval()
        self._torch = torch

    def _invoke_text(self, messages: list[dict[str, Any]]) -> str:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise RuntimeError("Local text model is not loaded.")

        prompt = self._apply_chat_template(self._tokenizer, messages)
        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)
        generated_ids = self._generate(inputs)
        output_ids = generated_ids[0][inputs.input_ids.shape[-1] :]
        content = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return self._postprocess_content(content)

    def _invoke_multimodal(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
    ) -> str:
        if self._model is None or self._processor is None or self._torch is None:
            raise RuntimeError("Local multimodal model is not loaded.")

        prompt = self._apply_chat_template(self._processor, messages)
        processor_kwargs: dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = images
        inputs = self._processor(**processor_kwargs).to(self._model.device)
        generated_ids = self._generate(inputs)
        output_ids = generated_ids[0][inputs["input_ids"].shape[-1] :]

        decoder = getattr(self._processor, "decode", None)
        if decoder is None:
            decoder = self._processor.tokenizer.decode
        content = decoder(output_ids, skip_special_tokens=True).strip()
        return self._postprocess_content(content)

    def _generate(self, inputs: Any) -> Any:
        if self._model is None or self._torch is None:
            raise RuntimeError("Local model is not loaded.")

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_tokens or self.default_max_new_tokens,
        }
        if self.temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                generation_kwargs["top_p"] = self.top_p
            if self.top_k is not None:
                generation_kwargs["top_k"] = self.top_k
            if self.min_p is not None:
                generation_kwargs["min_p"] = self.min_p
        else:
            generation_kwargs["do_sample"] = False

        tokenizer = self._tokenizer or getattr(self._processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                generation_kwargs["pad_token_id"] = eos_token_id

        with self._torch.inference_mode():
            return self._model.generate(**inputs, **generation_kwargs)

    def _apply_chat_template(self, template_owner: Any, messages: list[dict[str, Any]]) -> str:
        apply_chat_template = getattr(template_owner, "apply_chat_template", None)
        if apply_chat_template is None:
            return self._fallback_prompt(messages)

        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking

        try:
            prompt = apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            prompt = apply_chat_template(messages, **kwargs)
        return str(prompt)

    def _fallback_prompt(self, messages: list[dict[str, Any]]) -> str:
        rendered: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = self._text_only_content(message.get("content", ""))
            rendered.append(f"{role}: {content}")
        rendered.append("assistant:")
        return "\n".join(rendered)

    def _convert_messages(
        self,
        messages: list[Any],
        *,
        include_images: bool,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        hf_messages: list[dict[str, Any]] = []
        images: list[Any] = []
        for message in messages:
            role = getattr(message, "type", None) or getattr(message, "role", "user")
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"

            content = getattr(message, "content", message)
            converted_content = self._convert_content(
                content,
                include_images=include_images and role != "system",
                images=images,
            )
            hf_messages.append({"role": role, "content": converted_content})
        return hf_messages, images

    def _convert_content(
        self,
        content: Any,
        *,
        include_images: bool,
        images: list[Any],
    ) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)

        blocks: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                blocks.append({"type": "text", "text": item})
                continue
            if not isinstance(item, dict):
                text = str(item)
                text_parts.append(text)
                blocks.append({"type": "text", "text": text})
                continue

            item_type = item.get("type")
            if item_type == "text" or "text" in item:
                text = str(item.get("text", ""))
                text_parts.append(text)
                blocks.append({"type": "text", "text": text})
                continue
            if item_type == "image_url" or "image_url" in item:
                if include_images:
                    image = self._load_image_from_image_url(item.get("image_url"))
                    images.append(image)
                    blocks.append({"type": "image", "image": image})
                continue
            if item_type == "image" or "image" in item:
                if include_images:
                    image = self._load_image(item.get("image"))
                    images.append(image)
                    blocks.append({"type": "image", "image": image})
                continue

        if include_images and blocks:
            return blocks
        return "\n".join(part for part in text_parts if part)

    def _load_image_from_image_url(self, image_url: Any) -> Any:
        if isinstance(image_url, dict):
            return self._load_image(image_url.get("url"))
        return self._load_image(image_url)

    def _load_image(self, image_ref: Any) -> Any:
        if image_ref is None:
            raise ValueError("Image block is missing a URL or path.")

        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Local multimodal mode requires pillow to load image inputs."
            ) from exc

        if isinstance(image_ref, Image.Image):
            return image_ref.convert("RGB")

        image_value = str(image_ref)
        if image_value.startswith("data:"):
            _, encoded = image_value.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        if image_value.startswith(("http://", "https://")):
            response = requests.get(image_value, timeout=self.timeout)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        if image_value.startswith("file://"):
            path = Path(urlparse(image_value).path)
        else:
            path = Path(image_value).expanduser()
        return Image.open(path).convert("RGB")

    def _text_only_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and (item.get("type") == "text" or "text" in item):
                    parts.append(str(item.get("text", "")))
            return "\n".join(part for part in parts if part)
        return str(content)

    def _postprocess_content(self, content: str) -> str:
        if self.strip_thinking:
            content = re.sub(r"(?s)<think>.*?</think>\s*", "", content)
        return content.strip()


def _env_secret(*names: str) -> str | None:
    placeholders = {
        "your-api-key",
        "your_api_key_here",
        "your_dashscope_api_key",
        "your_qwen_api_key",
        "your_zai_api_key",
        "your_real_key",
    }
    for name in names:
        value = os.getenv(name, "").strip().strip('"').strip("'")
        if value and value not in placeholders:
            return value
    return None


def _is_qwen_compatible_config(config: MemeAgentConfig) -> bool:
    model = config.model.lower()
    base_url = (config.base_url or "").lower()
    return model.startswith("qwen") or "dashscope.aliyuncs.com" in base_url


def _is_loopback_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").lower()
    return hostname in {"127.0.0.1", "localhost", "::1"}


def _is_dashscope_base_url(base_url: str | None) -> bool:
    return "dashscope.aliyuncs.com" in (base_url or "").lower()


def _supports_openai_compatible_thinking_options(base_url: str | None) -> bool:
    return _is_loopback_base_url(base_url) or _is_dashscope_base_url(base_url)


def _openai_compatible_api_key(config: MemeAgentConfig) -> str:
    if _is_qwen_compatible_config(config):
        api_key = _env_secret(
            "MEMEAGENT_QWEN_API_KEY",
            "QWEN_API_KEY",
            "DASHSCOPE_API_KEY",
            "OPENAI_API_KEY",
            "MEMEAGENT_API_KEY",
        )
        if api_key:
            return api_key
        raise ValueError(
            "Qwen/DashScope API key is not set. Add DASHSCOPE_API_KEY, "
            "QWEN_API_KEY, MEMEAGENT_QWEN_API_KEY, or OPENAI_API_KEY to .env."
        )

    api_key = _env_secret("OPENAI_API_KEY", "MEMEAGENT_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Create D:\\自研Agent\\MemeAgent\\Agent\\.env "
            "from .env.example and set OPENAI_API_KEY=your_real_key, or set "
            "OPENAI_API_KEY in your shell before running."
        )
    return api_key


def _zai_api_key() -> str:
    api_key = _env_secret(
        "MEMEAGENT_GLM_API_KEY",
        "MEMEAGENT_ZAI_API_KEY",
        "ZAI_API_KEY",
        "GLM_API_KEY",
        "ZHIPUAI_API_KEY",
    )
    if not api_key:
        raise ValueError(
            "GLM API key is not set. Add ZAI_API_KEY, GLM_API_KEY, "
            "ZHIPUAI_API_KEY, or MEMEAGENT_GLM_API_KEY to .env when "
            "MEMEAGENT_CONTROLLER_PROVIDER=glm is enabled."
        )
    return api_key


def create_llm(config: MemeAgentConfig) -> Any:
    """Create the primary model client used for vision and final analysis."""

    if config.provider in _LOCAL_TRANSFORMERS_PROVIDERS:
        return LocalTransformersChatClient(
            model_path=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            enable_thinking=_env_optional_flag("MEMEAGENT_LOCAL_THINKING", False),
            strip_thinking=_env_flag("MEMEAGENT_LOCAL_STRIP_THINKING", True),
        )

    if config.provider in {"openai", "openai-compatible"}:
        return OpenAICompatibleChatClient(
            config=config,
            api_key=_openai_compatible_api_key(config),
            strip_thinking=_env_flag("MEMEAGENT_LOCAL_STRIP_THINKING", True),
        )

    if config.provider in {"glm", "zai", "zhipu"}:
        return ZhipuAIChatClient(
            api_key=_zai_api_key(),
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            thinking_enabled=config.controller_thinking_enabled,
        )

    raise ValueError(f"Unsupported provider: {config.provider}")


def create_controller_llm(config: MemeAgentConfig) -> Any | None:
    """Create a separate controller model when configured."""

    provider = config.controller_provider
    if not provider:
        return None

    if provider in _LOCAL_TRANSFORMERS_PROVIDERS:
        return LocalTransformersChatClient(
            model_path=config.controller_model,
            temperature=config.controller_temperature,
            max_tokens=config.controller_max_tokens,
            timeout=config.controller_timeout,
            enable_thinking=config.controller_thinking_enabled,
            strip_thinking=_env_flag("MEMEAGENT_LOCAL_STRIP_THINKING", True),
        )

    if provider in {"openai", "openai-compatible"}:
        return OpenAICompatibleChatClient(
            config=config,
            api_key=_openai_compatible_api_key(config),
            model=config.controller_model,
            temperature=config.controller_temperature,
            timeout=config.controller_timeout,
            max_tokens=config.controller_max_tokens,
            max_retries=config.controller_max_retries,
            enable_thinking=config.controller_thinking_enabled,
            strip_thinking=_env_flag("MEMEAGENT_LOCAL_STRIP_THINKING", True),
        )

    if provider in {"glm", "zai", "zhipu"}:
        return ZhipuAIChatClient(
            api_key=_zai_api_key(),
            model=config.controller_model,
            temperature=config.controller_temperature,
            max_tokens=config.controller_max_tokens,
            timeout=config.controller_timeout,
            thinking_enabled=config.controller_thinking_enabled,
        )

    raise ValueError(f"Unsupported controller provider: {provider}")


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _env_optional_flag(name: str, default: bool | None = None) -> bool | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _env_optional_float(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    return float(value) if value else None


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    return int(value) if value else None
