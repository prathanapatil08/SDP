from __future__ import annotations

import hmac
import ipaddress
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.types import ASGIApp

HEALTH_PATHS = {"/"}
DOCUMENTATION_PATHS = {"/docs", "/openapi.json", "/redoc"}


def _is_loopback(client_host: str | None) -> bool:
    if not client_host:
        return False
    try:
        return ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        return False


class LocalAdminKeyMiddleware:
    """Require an admin key when configured; otherwise permit loopback only."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        key_provider: Callable[[], str | None] | None = None,
    ) -> None:
        self.app = app
        self.key_provider = key_provider or (lambda: os.getenv("API_AUTH_TOKEN"))

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        if request.url.path in HEALTH_PATHS | DOCUMENTATION_PATHS:
            await self.app(scope, receive, send)
            return

        expected_key = self.key_provider()
        if expected_key:
            supplied_key = request.headers.get("x-api-key", "")
            if not hmac.compare_digest(supplied_key, expected_key):
                response: Response = JSONResponse(
                    {"detail": "A valid API key is required."},
                    status_code=401,
                    headers={"WWW-Authenticate": "ApiKey"},
                )
                await response(scope, receive, send)
                return
        elif not _is_loopback(
            request.client.host if request.client is not None else None
        ):
            response = JSONResponse(
                {
                    "detail": (
                        "The API is configured for loopback-only access. "
                        "Set API_AUTH_TOKEN before allowing network access."
                    )
                },
                status_code=403,
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def request_id() -> str:
    return uuid.uuid4().hex


def now_monotonic() -> float:
    return time.perf_counter()
