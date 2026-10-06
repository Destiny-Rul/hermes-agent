"""An Anthropic OAuth rotation picked up by the main loop must reach the auxiliary routes too.

``_try_refresh_anthropic_client_credentials`` re-resolves the token before each request; when the
login rotated (Claude Code refreshing ``~/.claude/.credentials.json``) the old access token is revoked.
Auxiliary calls on ``auto`` pin the main session's key — ``_current_main_runtime()`` and the context
compressor's ``main_runtime`` — so a snapshot left on the old token 401s every compression while the
main loop keeps working.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent

OLD = "sk-ant-oat01-rotated-out"
NEW = "sk-ant-oat01-current"


def _anthropic_agent() -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key=OLD,
            base_url="https://api.anthropic.com",
            provider="anthropic",
            model="claude-opus-5-5",
            api_mode="anthropic_messages",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    assert agent._anthropic_api_key == OLD and agent.api_key == OLD
    return agent


def _rotate(agent: AIAgent) -> bool:
    with (
        patch("agent.anthropic_credentials.resolve_anthropic_token", return_value=NEW),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
    ):
        return agent._try_refresh_anthropic_client_credentials()


def test_rotation_moves_the_aux_main_runtime_to_the_current_token():
    agent = _anthropic_agent()
    assert agent.context_compressor.api_key == OLD

    assert _rotate(agent) is True

    assert agent._anthropic_api_key == NEW
    assert agent._current_main_runtime()["api_key"] == NEW
    assert agent.context_compressor.api_key == NEW


def test_compression_summary_call_carries_the_current_token():
    agent = _anthropic_agent()
    _rotate(agent)
    sent = {}

    def fake_call_llm(**kwargs):
        sent.update(kwargs)
        raise RuntimeError("stop after capturing the route")

    with patch("agent.context_compressor.call_llm", side_effect=fake_call_llm):
        try:
            agent.context_compressor._call_summary_llm("summarize", 0.0)
        except RuntimeError:
            pass

    assert sent["main_runtime"]["api_key"] == NEW


def test_an_unrelated_compressor_key_is_left_alone():
    """A compressor pointed at another route (its own key) is not rewritten by the main rotation."""
    agent = _anthropic_agent()
    agent.context_compressor.api_key = "sk-other-route"

    assert _rotate(agent) is True

    assert agent.context_compressor.api_key == "sk-other-route"
