"""Guard the AI/ML API (`aiml`) judge route that RULER depends on.

RULER delegates every judge call to LiteLLM (`art.rewards.ruler.ruler` ->
`litellm.acompletion`), so AI/ML API support is not ART code — it is whatever
the pinned LiteLLM resolves for the `aiml/` prefix. `pyproject.toml` pins
`litellm>=1.71.1,<=1.82.0`, and the route silently disappearing on a bump would
turn `judge_model="aiml/..."` into a confusing "LLM Provider NOT provided"
error at training time rather than at import time. These assertions are cheap
and require neither network access nor a key.
"""

import litellm
from litellm.utils import get_llm_provider
import pytest

from art.rewards.ruler import _judge_provider

AIML_PROVIDER = "aiml"
AIML_API_BASE = "https://api.aimlapi.com/v1"


def test_aiml_is_a_known_litellm_provider() -> None:
    assert AIML_PROVIDER in litellm.provider_list
    assert AIML_PROVIDER in litellm.openai_compatible_providers


@pytest.mark.parametrize(
    "judge_model, expected_model",
    [
        ("aiml/openai/gpt-5-5", "openai/gpt-5-5"),
        ("aiml/anthropic/claude-sonnet-4.6", "anthropic/claude-sonnet-4.6"),
        ("aiml/google/gemini-2.5-flash", "google/gemini-2.5-flash"),
    ],
)
def test_aiml_judge_model_resolves_to_the_aimlapi_chat_endpoint(
    judge_model: str, expected_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AI/ML API model ids contain slashes of their own, so the prefix must be
    # stripped exactly once and the remainder forwarded verbatim.
    monkeypatch.setenv("AIML_API_KEY", "sentinel-not-a-real-key")
    model, provider, api_key, api_base = get_llm_provider(model=judge_model)

    assert model == expected_model
    assert provider == AIML_PROVIDER
    assert api_base == AIML_API_BASE
    # LiteLLM reads AIML_API_KEY, not the AIMLAPI_API_KEY used elsewhere in the
    # ecosystem; getting this wrong surfaces only as a 401 on the first call.
    assert api_key == "sentinel-not-a-real-key"


def test_ruler_attributes_aiml_judges_to_the_aiml_provider() -> None:
    # `_judge_provider` splits on the first "/" only, which is what keeps
    # multi-segment AI/ML API ids intact for cost attribution.
    assert _judge_provider("aiml/openai/gpt-5-5") == AIML_PROVIDER
    assert _judge_provider("AIML/openai/gpt-5-5") == AIML_PROVIDER
