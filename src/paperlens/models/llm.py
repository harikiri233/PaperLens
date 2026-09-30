"""LLM 客户端：封装 OpenAI 兼容协议，处理 DeepSeek 的特有行为。

- 思考模式：DeepSeek 默认开启思考。结构化调用（查询分析、JSON 抽取）默认关闭，降低延迟和成本。
- JSON 模式：DeepSeek 要求提示词里出现 "json"，并且偶尔会返回空 content，需要重试。
- 用量：每次调用的 token 和费用记入 UsageTracker。
"""

import json
import time
from dataclasses import dataclass
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from paperlens.config import Settings
from paperlens.models.usage import UsageTracker

T = TypeVar("T", bound=BaseModel)

Message = dict[str, Any]

_JSON_HINT = "请只输出一个合法的 JSON 对象，不要输出 JSON 以外的任何内容。"


class LLMError(RuntimeError):
    """模型返回无法使用的结果（空内容、非法 JSON、不符合 Schema 等）。"""


@dataclass
class ChatResult:
    content: str
    reasoning: str | None
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: float
    finish_reason: str | None


def _cached_tokens(usage: Any) -> int:
    """读取缓存命中的输入 token 数。

    DeepSeek 用 prompt_cache_hit_tokens，OpenAI 用 prompt_tokens_details.cached_tokens。
    """
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    if isinstance(hit, int):
        return hit
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None)
    return cached if isinstance(cached, int) else 0


def _ensure_json_hint(messages: list[Message]) -> list[Message]:
    """JSON 模式要求提示词里出现 "json"。没有出现时在 system 消息末尾补一句说明。"""
    if any("json" in str(m.get("content", "")).lower() for m in messages):
        return messages
    patched = [dict(m) for m in messages]
    if patched and patched[0].get("role") == "system":
        patched[0]["content"] = f"{patched[0]['content']}\n\n{_JSON_HINT}（输出格式：JSON）"
    else:
        patched.insert(0, {"role": "system", "content": f"{_JSON_HINT}（输出格式：JSON）"})
    return patched


def _schema_prompt(schema: type[BaseModel]) -> str:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    return f"{_JSON_HINT}\n输出必须符合以下 JSON Schema：\n{schema_json}"


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        *,
        tracker: UsageTracker | None = None,
        client: Any = None,
    ) -> None:
        """client 参数用于测试时注入替身；正常使用时根据配置创建 OpenAI 客户端。"""
        self.settings = settings
        self.model = settings.llm_model
        self.tracker = tracker or UsageTracker()
        if client is None:
            if settings.llm_api_key is None:
                raise LLMError("未配置 LLM_API_KEY，请在 .env 中填写。")
            # SDK 自带重试：连接错误、408、409、429、5xx 会按指数退避重试
            client = OpenAI(
                api_key=settings.llm_api_key.get_secret_value(),
                base_url=settings.llm_base_url,
                timeout=settings.llm_timeout,
                max_retries=3,
            )
        self._client = client

    def chat(
        self,
        messages: list[Message],
        *,
        thinking: bool = False,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        """发起一次非流式对话。

        思考模式下 DeepSeek 会忽略 temperature，所以只在关闭思考时传递。
        """
        if json_mode:
            messages = _ensure_json_hint(messages)
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if temperature is not None and not thinking:
            kwargs["temperature"] = temperature
        if self.settings.llm_send_thinking_param:
            kwargs["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}

        start = time.perf_counter()
        response = self._client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - start

        choice = response.choices[0]
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        cached = _cached_tokens(usage)
        cost = self.tracker.record(
            self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached,
        )
        return ChatResult(
            content=choice.message.content or "",
            reasoning=getattr(choice.message, "reasoning_content", None),
            model=getattr(response, "model", None) or self.model,
            input_tokens=input_tokens,
            cached_input_tokens=cached,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_s=latency,
            finish_reason=choice.finish_reason,
        )

    def complete_json(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        max_tokens: int = 1024,
        max_attempts: int = 3,
        temperature: float | None = 0.0,
    ) -> T:
        """JSON 模式调用，并用 Pydantic 校验结果。

        空内容、被截断、非法 JSON、不符合 Schema 时重试；非空的错误结果会作为反馈发回模型。
        """
        base = [{"role": "system", "content": _schema_prompt(schema)}, *messages]
        conversation = list(base)
        last_error = ""
        for _ in range(max_attempts):
            result = self.chat(
                conversation,
                json_mode=True,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = result.content.strip()
            if not content:
                # 已知的偶发空返回：直接用原始提示重试
                last_error = "模型返回了空内容"
                conversation = list(base)
                continue
            if result.finish_reason == "length":
                last_error = f"输出在 max_tokens={max_tokens} 处被截断"
                conversation = list(base)
                continue
            try:
                return schema.model_validate_json(content)
            except ValidationError as exc:
                last_error = f"输出不符合 Schema：{exc.errors(include_url=False)}"
                conversation = [
                    *base,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": f"上面的 JSON 有误：{last_error}。请修正后重新输出完整的 JSON。",
                    },
                ]
        raise LLMError(f"{max_attempts} 次尝试后仍未得到合法 JSON：{last_error}")
