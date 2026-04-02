"""Helpers for live RL request metadata injection.

The Hermes fork keeps live-learning integration intentionally minimal:
main-loop chat-completions requests may be annotated with request metadata
so an external proxy can reconstruct assistant->next_state transitions.
"""

from __future__ import annotations

import os
import uuid
from urllib.parse import urlparse

DEFAULT_HERMES_RL_PROXY_PORT = 30050
_TRUTHY = {"1", "true", "yes", "on"}


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in _TRUTHY


def _normalized_base_url(value: str | None) -> str:
    if not value:
        return ""
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return value.rstrip("/")
    path = parsed.path.rstrip("/")
    if path == "/v1":
        path = ""
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _looks_like_default_local_proxy(base_url: str | None) -> bool:
    if not base_url:
        return False
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost"}:
        return False
    return (parsed.port or 0) == DEFAULT_HERMES_RL_PROXY_PORT


def should_inject_rl_headers(base_url: str | None, env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    if _is_truthy(env.get("HERMES_RL_ENABLED")):
        return True

    normalized_base = _normalized_base_url(base_url)
    target_base = _normalized_base_url(env.get("HERMES_RL_PROXY_BASE_URL"))
    if normalized_base and target_base and normalized_base == target_base:
        return True

    return _looks_like_default_local_proxy(base_url)


def new_rl_request_id() -> str:
    return f"hermes_rl_{uuid.uuid4().hex}"


def build_rl_headers(
    *,
    session_id: str,
    outer_turn_id: str | int,
    step_index: str | int,
    turn_type: str = "main",
    request_id: str | None = None,
) -> dict[str, str]:
    if turn_type not in {"main", "ignore"}:
        raise ValueError(f"Unsupported turn_type: {turn_type}")

    return {
        "X-Session-Id": str(session_id),
        "X-Hermes-Outer-Turn-Id": str(outer_turn_id),
        "X-Hermes-Step-Index": str(step_index),
        "X-Turn-Type": turn_type,
        "X-Hermes-Request-Id": request_id or new_rl_request_id(),
    }
