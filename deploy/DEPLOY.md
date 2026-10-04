# Deploying the demo (single host, Docker)

```bash
cd feedback-agent
cp deploy/.env.prod.example deploy/.env.prod && chmod 600 deploy/.env.prod    # then fill in the key and a long random API_TOKEN
docker compose -f deploy/docker-compose.prod.yml up -d --build
curl -s http://localhost:8898/health                                           # {"status":"ok"}
```

Open the port 8898 in the host firewall. Clients send the token as header `X-API-Token` (Swagger UI at `/docs` has an *Authorize* button). `/health` and `/docs` need no token.
Limits are per UTC day and persisted in SQLite: `DAILY_REQUEST_LIMIT` reports, `DAILY_TOKEN_LIMIT` prompt+output tokens, `RATE_LIMIT_PER_MIN` per client IP on `POST /feedback`.
Also set a spending/quota limit on the Gemini key itself: these caps bound requests, the key limit bounds money.

Stop and remove the demo: `docker compose -f deploy/docker-compose.prod.yml down -v` (the `-v` also deletes the stored reports).
