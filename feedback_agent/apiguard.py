"""Optional protection for a shared deployment: API token, per-IP rate limit and daily request/token caps.
Everything is off unless configured (see Settings), so local use and tests are unaffected."""

from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from .config import Settings
from .database import Database

token_header = APIKeyHeader(
    name="X-API-Token", auto_error=False
)  # also gives Swagger UI an "Authorize" button


class ApiGuard:
    """FastAPI dependencies that protect the API; each check is off while its setting is unset."""

    def __init__(self, settings: Settings, database: Database):
        self.settings, self.database = settings, database
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def auth(self, key: Optional[str] = Security(token_header)) -> None:
        """401 unless the X-API-Token header matches settings.api_token (no check when unset)."""
        if self.settings.api_token and not (
            key and secrets.compare_digest(key, self.settings.api_token)
        ):
            raise HTTPException(401, "missing or invalid X-API-Token header")

    def _client_ip(self, request: Request) -> str:
        """Client IP for rate limiting; X-Forwarded-For is used only with trust_proxy."""
        forwarded_for = request.headers.get("x-forwarded-for", "")
        if self.settings.trust_proxy and forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def cost_guard(self, request: Request) -> None:
        """For POST /feedback (the only call that spends LLM money): per-IP rate, then daily caps."""
        if self.settings.rate_limit_per_min:
            ip, now = self._client_ip(request), time.monotonic()
            with self._lock:
                recent_hits = self._hits[ip]
                while recent_hits and now - recent_hits[0] > 60:
                    recent_hits.popleft()
                if len(recent_hits) >= self.settings.rate_limit_per_min:
                    retry_after = int(60 - (now - recent_hits[0])) + 1
                    raise HTTPException(
                        429,
                        f"rate limit: {self.settings.rate_limit_per_min} requests/minute per client",
                        headers={"Retry-After": str(retry_after)},
                    )
                recent_hits.append(now)
        if self.settings.daily_request_limit or self.settings.daily_token_limit:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            # persisted in SQLite, so restarts do not reset the cap
            requests_today, tokens = self.database.usage_since(day)
            if (
                self.settings.daily_request_limit
                and requests_today >= self.settings.daily_request_limit
            ):
                raise HTTPException(429, "daily request limit reached; try again after 00:00 UTC")
            if self.settings.daily_token_limit and tokens >= self.settings.daily_token_limit:
                raise HTTPException(429, "daily token budget reached; try again after 00:00 UTC")
