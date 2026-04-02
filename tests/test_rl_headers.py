from agent.rl_headers import (
    DEFAULT_HERMES_RL_PROXY_PORT,
    build_rl_headers,
    should_inject_rl_headers,
)


def test_should_inject_for_env_flag():
    assert should_inject_rl_headers(
        "https://example.com/v1",
        env={"HERMES_RL_ENABLED": "1"},
    ) is True


def test_should_inject_for_matching_proxy_base_url():
    assert should_inject_rl_headers(
        "http://127.0.0.1:30050/v1",
        env={"HERMES_RL_PROXY_BASE_URL": "http://127.0.0.1:30050"},
    ) is True


def test_should_inject_for_default_local_proxy_port():
    assert should_inject_rl_headers(
        f"http://localhost:{DEFAULT_HERMES_RL_PROXY_PORT}/v1"
    ) is True


def test_should_not_inject_for_non_proxy_url():
    assert should_inject_rl_headers("http://127.0.0.1:8010/v1") is False


def test_build_rl_headers_shape():
    headers = build_rl_headers(
        session_id="session-1",
        outer_turn_id=3,
        step_index=2,
        turn_type="main",
        request_id="req-1",
    )
    assert headers == {
        "X-Session-Id": "session-1",
        "X-Hermes-Outer-Turn-Id": "3",
        "X-Hermes-Step-Index": "2",
        "X-Turn-Type": "main",
        "X-Hermes-Request-Id": "req-1",
    }
