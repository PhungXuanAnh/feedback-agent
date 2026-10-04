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
from .store import Store

token_header = APIKeyHeader(name="X-API-Token", auto_error=False)  # also gives Swagger UI an "Authorize" button


class ApiGuard:
    def __init__(self, settings: Settings, store: Store):
        self.s, self.store = settings, store
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def auth(self, key: Optional[str] = Security(token_header)) -> None:
        if self.s.api_token and not (key and secrets.compare_digest(key, self.s.api_token)):
            raise HTTPException(401, "missing or invalid X-API-Token header")

    def _client_ip(self, request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for", "")
        if self.s.trust_proxy and fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def cost_guard(self, request: Request) -> None:
        """For POST /feedback (the only call that spends LLM money): per-IP rate, then daily caps."""
        if self.s.rate_limit_per_min:
            ip, now = self._client_ip(request), time.monotonic()
            with self._lock:
                q = self._hits[ip]
                while q and now - q[0] > 60:
                    q.popleft()
                if len(q) >= self.s.rate_limit_per_min:
                    retry = int(60 - (now - q[0])) + 1
                    raise HTTPException(429, f"rate limit: {self.s.rate_limit_per_min} requests/minute per client",
                                        headers={"Retry-After": str(retry)})
                q.append(now)
        if self.s.daily_request_limit or self.s.daily_token_limit:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            n, tokens = self.store.usage_since(day)  # persisted in SQLite, so restarts do not reset the cap
            if self.s.daily_request_limit and n >= self.s.daily_request_limit:
                raise HTTPException(429, "daily request limit reached; try again after 00:00 UTC")
            if self.s.daily_token_limit and tokens >= self.s.daily_token_limit:
                raise HTTPException(429, "daily token budget reached; try again after 00:00 UTC")
