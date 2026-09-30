from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from paperlens.models.llm import LLMClient, LLMError, _ensure_json_hint


class Answer(BaseModel):
    answer: str
    confidence: float


def _response(content, *, finish="stop", prompt=100, completion=20, cache_hit=40):
    usage = SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, prompt_cache_hit_tokens=cache_hit
    )
    message = SimpleNamespace(content=content, reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=usage,
        model="deepseek-v4.1-flash",
    )


class FakeOpenAI:
    """按顺序返回预设响应，并记录每次请求的参数。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def test_chat_disables_thinking_and_records_usage(settings):
    fake = FakeOpenAI([_response("你好")])
    llm = LLMClient(settings, client=fake)
    result = llm.chat([{"role": "user", "content": "hi"}], temperature=0.2)
    call = fake.calls[0]
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert call["temperature"] == 0.2
    assert result.content == "你好"
    assert result.cached_input_tokens == 40
    assert llm.tracker.snapshot()["deepseek-flash"].input_tokens == 100


def test_thinking_mode_drops_temperature(settings):
    fake = FakeOpenAI([_response("ok")])
    LLMClient(settings, client=fake).chat(
        [{"role": "user", "content": "hi"}], thinking=True, temperature=0.5
    )
    call = fake.calls[0]
    assert call["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "temperature" not in call


def test_thinking_param_can_be_turned_off(settings):
    settings.llm_send_thinking_param = False
    fake = FakeOpenAI([_response("ok")])
    LLMClient(settings, client=fake).chat([{"role": "user", "content": "hi"}])
    assert "extra_body" not in fake.calls[0]


def test_json_hint_added_when_missing():
    patched = _ensure_json_hint([{"role": "user", "content": "列出三个要点"}])
    assert patched[0]["role"] == "system"
    assert "json" in patched[0]["content"].lower()
    untouched = [{"role": "user", "content": "请用 JSON 输出"}]
    assert _ensure_json_hint(untouched) is untouched


def test_complete_json_success(settings):
    fake = FakeOpenAI([_response('{"answer": "reflection tokens", "confidence": 0.9}')])
    result = LLMClient(settings, client=fake).complete_json(
        [{"role": "user", "content": "q"}], Answer
    )
    assert result == Answer(answer="reflection tokens", confidence=0.9)
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


def test_complete_json_retries_empty_then_invalid(settings):
    fake = FakeOpenAI(
        [
            _response(""),  # DeepSeek 偶发的空返回
            _response('{"answer": "x"}'),  # 缺少字段
            _response('{"answer": "x", "confidence": 0.5}'),
        ]
    )
    result = LLMClient(settings, client=fake).complete_json(
        [{"role": "user", "content": "q"}], Answer
    )
    assert result.confidence == 0.5
    # 第三次请求带上了上一次的错误输出和修正要求
    last_messages = fake.calls[2]["messages"]
    assert last_messages[-2] == {"role": "assistant", "content": '{"answer": "x"}'}
    assert "有误" in last_messages[-1]["content"]


def test_complete_json_gives_up(settings):
    fake = FakeOpenAI([_response(""), _response(""), _response("")])
    with pytest.raises(LLMError, match="空内容"):
        LLMClient(settings, client=fake).complete_json([{"role": "user", "content": "q"}], Answer)


def test_complete_json_retries_truncated_output(settings):
    fake = FakeOpenAI(
        [
            _response('{"answer": "x", "conf', finish="length"),
            _response('{"answer": "x", "confidence": 1}'),
        ]
    )
    result = LLMClient(settings, client=fake).complete_json(
        [{"role": "user", "content": "q"}], Answer
    )
    assert result.confidence == 1


def test_missing_key_raises(settings):
    settings.llm_api_key = None
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        LLMClient(settings)
