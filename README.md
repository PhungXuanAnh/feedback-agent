- [1. Agentic Customer Feedback System](#1-agentic-customer-feedback-system)
  - [1.1. Quick start](#11-quick-start)
    - [1.1.0. Live demo (no setup)](#110-live-demo-no-setup)
    - [1.1.1. Local setup](#111-local-setup)
    - [1.1.2. Try your own feedback (needs the Gemini key)](#112-try-your-own-feedback-needs-the-gemini-key)
  - [1.2. Architecture](#12-architecture)
  - [1.3. Components](#13-components)
  - [1.4. Evidence: samples and evaluation](#14-evidence-samples-and-evaluation)
  - [1.5. Scaling the knowledge base (not built here)](#15-scaling-the-knowledge-base-not-built-here)
  - [1.6. Layout and criteria](#16-layout-and-criteria)
  - [1.7. Notes](#17-notes)


# 1. Agentic Customer Feedback System

Takes a customer's free-text feedback and produces a **grounded, structured report for a customer-support (CS) officer**:
classify it, let an LLM query three data sources through real tool calls, check the report's citations and actions in code,
and park it in a human review queue. Design reasoning and trade-offs are in [`WRITEUP.md`](WRITEUP.md); this file covers how to run it,
how it is built and where the evidence is.

Python 3.10+, no agent framework (the loop is ~250 lines on a function-calling API). The LLM is **Gemini** behind a small provider
interface; a **scripted provider** replays recorded turns, so tests, samples and the demo run **offline without an API key**.

## 1.1. Quick start

Use the live demo (nothing to install) or start this project locally, then try your own feedback with the `curl` commands in 1.1.2.

### 1.1.0. Live demo (no setup)

- A hosted instance is available at **http://feedback-agent.xuananh1.site:8898** (plain HTTP, include the port). 
- Open `/docs` for the Swagger UI, click *Authorize* and paste the demo access token `wB8ytvZzWzbU0zex4Av8Hvcw2te81vSy`
(it goes in the `X-API-Token` header; it only opens this demo, please do not share it); `/health` and `/docs` are open, every other call needs the token. Use the mock customers below. The demo is deliberately small and protected:
at most 6 requests per minute per client on `POST /feedback`, and shared daily caps of 60 reports and 600k tokens. It runs on a small shared host, stores the reports you submit, and will be taken down after the review period,
so please do not send real or sensitive data. If it is down or the daily cap is reached, everything works the same locally (next section).

### 1.1.1. Local setup

Requires only Docker with Compose v2. From the repository folder (`feedback-agent/`):

```bash
cd  feedback-agent
docker compose run --rm app pytest                              # the test suite, offline, no key needed
docker compose run --rm app python -m feedback_agent demo       # full run offline: report, trace, review queue, officer approval
```

For the real model, put your key in `.env` (`cp .env.example .env`, set `GEMINI_API_KEY`; `LLM_API_KEY` is also accepted) or export it, then start the API:

```bash
docker compose up --build        # API on http://localhost:8000 (Swagger UI at /docs), health check included
```

Every `python -m feedback_agent ...` command in this README runs the same way inside the container: `docker compose run --rm app python -m feedback_agent <command>`
(reports and traces are kept in a Docker volume between runs; `docker compose down -v` resets them). Without Docker you need Python 3.10+ and `pip install -e ".[dev]"`, then drop the `docker compose run --rm app` prefix.

### 1.1.2. Try your own feedback (needs the Gemini key)

The scripted provider only replays recorded turns, so free text needs the real model. The mock database has these customers (all in `data/seed/customers.csv`);
any other email is treated as an unknown sender (`not_found`, "unverified customer").

| Email | Customer | Good for |
|---|---|---|
| `ops@northwind-analytics.example` | Enterprise | outage / SLA |
| `admin@bluepeak-labs.example` | Pro, one open webhook ticket | bugs, vague messages |
| `billing@orchid-robotics.example` | Pro, duplicate 1200 USD invoices (2026-09-10) | duplicate charge and refund |
| `finance@fathom-labs.example` | Standard, one 299 USD invoice (2026-06-10) | refund policy v1 vs v2 (`received_at` 2026-06-20 vs 2026-07-05) |

**HTTP API** (`docker compose up --build`, needs the key). The quickest way to try it by hand: open **http://localhost:8000/docs** (interactive Swagger UI), expand
`POST /feedback`, click *Try it out*, paste a body such as the one below and *Execute*; then use `GET /reports` and `POST /reports/{id}/review` the same way. The same calls with `curl`.
What each call does (they follow the order of a real session: submit, find the report in the queue, decide, inspect how it was produced):

| Call | What it does |
|---|---|
| `GET /health` | Liveness check, always open (no token). |
| `POST /feedback` | Submits **one** customer message and runs the whole pipeline synchronously (guard, classify, tool-calling agent, grounded report). Returns the finished report with its `report_id` and `trace_id`, status `pending_review`; takes several seconds because it calls the LLM. Body: `text`, `customer_email` (must be a mock customer, see above), `channel`, optional `received_at`. This is the only call that spends LLM tokens and the only one rate-limited. |
| `GET /reports?status=pending_review` | The human review queue: a short summary of each report (id, category, urgency, confidence, whether it needs human triage). Omit `status` to list all; other values: `approved`, `overridden`, `rejected`, `executed`. |
| `GET /reports/<report_id>` | One full report: classification, evidence cited, proposed actions with their basis, and the review history. |
| `POST /reports/<report_id>/review` | The officer's decision on a pending report: `approve` (run the proposed actions), `override` (change category, urgency or actions via `overrides`, then run those) or `reject` (do nothing). Approved actions go to a **stub** executor that only records what it would call (no real refund is made). The first decision wins: repeating the same one is a harmless replay, a conflicting one gets `409`. |
| `GET /traces/<trace_id>` | The step-by-step audit trail of that run (every LLM turn, tool call, validation and token count), to see why the report says what it says. |

```bash
# ----------- local
export BASE=http://localhost:8000      
export TOKEN=local                     # local: any value works (no auth)

# ----------- live demo
export BASE=http://feedback-agent.xuananh1.site:8898
export TOKEN=wB8ytvZzWzbU0zex4Av8Hvcw2te81vSy


# 1) Submit feedback. Every value is yours to choose: "text" is the customer message, "customer_email" must be one of the mock customers in the table above,
#    "channel" is email | web_form | chat | api, "received_at" is optional (see the note below). The JSON response contains "report_id" and "trace_id": copy them for steps 3 and 4.
curl -s -X POST $BASE/feedback -H "X-API-Token: $TOKEN" -H 'Content-Type: application/json' -d '{
  "text": "I was charged twice on 2026-09-10, 1200 USD each time. Please refund the duplicate.",
  "customer_email": "billing@orchid-robotics.example", "channel": "email"}'   # optional: "received_at": "2026-09-22T09:00:00Z"

# 2) List the reports waiting for an officer. "status" is a filter you pick (pending_review | approved | overridden | rejected | executed, or drop it for all);
#    each row shows its "report_id", so this is the other place to find the id for step 3 (e.g. a report created earlier or by someone else).
curl -s "$BASE/reports?status=pending_review" -H "X-API-Token: $TOKEN"

# 3) Officer decision. Replace <report_id> with the "report_id" from step 1 or from the list in step 2 (looks like rep_ + 10 hex characters; one that was already decided returns the same result or 409).
#    "decision" is approve | override | reject, "actor" is your name for the audit log, "note" is free text.
curl -s -X POST $BASE/reports/<report_id>/review -H "X-API-Token: $TOKEN" -H 'Content-Type: application/json' \
     -d '{"decision": "approve", "actor": "me", "note": "ok"}'

# 4) Audit trail of one run. Replace <trace_id> with the "trace_id" from the step 1 response (looks like tr_ + 12 hex characters; it is also a field of the report).
curl -s $BASE/traces/<trace_id> -H "X-API-Token: $TOKEN"
```

Without `received_at` the feedback is dated now; policies are chosen by that date and the mock invoices are from mid-2026, so use `received_at` to line up with them.
Reports and reviews are stored in SQLite and traces as JSONL files, both in the Docker volume (locally without Docker: `data/feedback_agent.db` and `traces/`).
Without the API server, run these as `docker compose run --rm app python -m feedback_agent <command>`: `run <file.json>` (feedback from a JSON file, e.g. `samples/inputs/01_enterprise_outage.json`), `reports`, `show <id>`, `review <id> approve|override|reject --actor you`, `trace <id>`, `samples`, `eval`, `seed`.

| `.env` variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER`, `LLM_MODEL` | `gemini`, set in `.env.example` | provider and model id (configuration, not code) |
| `LLM_THINKING_LEVEL` | `low` in `.env.example` | Gemini 3 thinking level; `low` cut latency and tokens a lot (see Evidence) |
| `MAX_LLM_TURNS`, `MAX_TOOL_CALLS` | `4`, `6` | the two step caps |
| `MAX_OUTPUT_TOKENS` | `8192` | per-call limit (thinking tokens count against it) |
| `PRICE_PER_MTOK_IN` / `_OUT` | unset | cost is estimated only when both are set, otherwise `unknown` (never 0) |
| `DATABASE_PATH`, `TRACE_DIR` | `data/feedback_agent.db`, `traces/` | SQLite file, JSONL traces |
| `API_TOKEN`, `RATE_LIMIT_PER_MIN`, `DAILY_REQUEST_LIMIT`, `DAILY_TOKEN_LIMIT`, `TRUST_PROXY` | unset (all protection off) | for a shared deployment: clients send header `X-API-Token`; per-IP rate limit on `POST /feedback`; daily request and token caps counted in SQLite; see [`deploy/DEPLOY.md`](deploy/DEPLOY.md) |

## 1.2. Architecture

```
feedback ─► intake ─► injection guard ─► classify ─► agent loop (LLM + 3 tools) ─► report + validator ─► review queue
              code         code          LLM + code     LLM, rules enforced by code     LLM + code      code, human decides
```

A fixed six-step pipeline; the LLM is used only where language understanding is needed (classification, the tool loop, the report text). Everything that must be exactly right is plain code.

| # | Step | LLM? | What it does | On failure |
|---|---|---|---|---|
| 1 | Intake | no | validates text + metadata (email, channel, time), caps length | HTTP 422 |
| 2 | Injection guard | no | regexes **flag** injection patterns (never block); text wrapped in escaped `<customer_feedback>` | n/a |
| 3 | Classify | 1 schema-forced call | category, up to 2 alternatives, sentiment, urgency, confidence; code adds urgency floors and `needs_human_triage` | retry once, then keyword rules |
| 4 | Agent loop | yes | LLM chooses and calls the 3 tools, reads results, may search again | see Failure behaviour |
| 5 | Report + validator | LLM + code | LLM calls `submit_report`; code checks grounding, fills category/urgency/flags/confidence | one repair, then code-built template |
| 6 | Review queue | no | saved as `pending_review`; an officer approves, overrides or rejects; only then a stub executor acts | storage error is returned, never "queued" |

```
 LLM turn ──┬─ retrieval tools ─► executor (validate args, cache, budget, 1 retry) ─► results back to the LLM ─┐
            └─ submit_report ─► evidence gate (3 sources consulted?) ─► grounding validator ─► valid ─► report
                                     │ missing: tell the LLM            │ invalid: tell the LLM, one repair;
                                     ▼                                  ▼ still invalid: rebuild from validated data
                                the last turn is reserved for submit_report (forced via tool_choice)
```

**One request end to end.** The feedback is validated and wrapped, then classified once. The agent loop starts with the classification and the metadata in the conversation; the LLM calls
`get_cs_guidelines`, `lookup_customer` and `search_policies` (usually all three in one turn) and each result is fed back. The executor validates arguments, caches repeats and enforces the budget.
When the LLM calls `submit_report`, the evidence gate checks that all three sources were consulted, then the validator checks every citation and action against this run's tool results.
Valid: code adds category, urgency, flags and confidence and saves the report for review. Invalid: the LLM gets the exact problems and one repair; still invalid: code rebuilds the report from validated data only.
Every step writes a trace event. An officer's decision is validated by the same rules before the stub executor acts.

## 1.3. Components

**Intake and injection guard.** Pydantic validation of text and metadata; regexes flag known injection patterns before the LLM sees the text (flags only; easy to evade, so never the only defence). Layered defences: role separation; escaped delimiters around customer text **and every tool result**;
schema-forced output; read-only tools with allow-listed arguments; customer lookup locked to the sender; validator and review state machine decide what can happen. None is 100%, so damage is limited, not just detection attempted.

**Classification** (hybrid: rules are brittle, classic ML needs labelled data, a bare LLM can be confidently wrong). The schema forces valid values; then code applies urgency floors (security/legal wording or an Enterprise outage means at least `high`; the LLM can only raise urgency)
and flags `needs_human_triage` when confidence < 0.65, the category is `unclear` or a second reading has confidence >= 0.4. Thresholds are reasoned (conservative), not tuned on the small eval set.

**Data sources and tools** (mock data in `data/`). Tools are read-only: the LLM cannot refund, close, send or write.

| Source | Tool | Notes |
|---|---|---|
| CS guidelines (`guidelines.json`) | `get_cs_guidelines(category)` | one per category; each lists `allowed_actions` |
| Customer info (SQLite from CSV) | `lookup_customer(email/id)` | tier, tenure, tickets, invoices, `duplicate_charge_candidates`; **locked to the sender** (else `out_of_scope` + flag) |
| Company policy (`policies.json`) | `search_policies(query, category?, top_k<=3)` | only policies **in force on the feedback date** (`effective_date`, `expires_date`, `supersedes`), keyword-scored; category only boosts |

Each tool returns `found`, `not_found` (asked, nothing there: counts as consulted) or `error` (source failed: does **not** count).

**Agent loop and caps.** The evidence gate refuses `submit_report` until guidelines, customer and policies were all consulted. Two separate caps: `max_llm_turns` (agent-loop turns, default 4, includes the forced-submit and repair turns; classify counted separately)
and `max_tool_calls` (default 6; cache hits and `submit_report` are free). The last turn is reserved for `submit_report`; if sources are still missing the system runs those lookups itself (`forced_by_system`, max 3, traced), so gate and cap cannot deadlock.
Worst case per request: 1 classify + 4 agent turns, each retried once, so up to 10 provider calls (all counted).

**Report and grounding validator** (`validator.py`), checked against this run's tool results: cited ids exist; `customer_context` matches the record (no record means `record_found=false`, never an invented tier);
money figures and ISO dates in the summary come from the feedback, an invoice of this customer or a policy limit; every action has a `basis` whose `allowed_actions` includes that action type.
`issue_refund` is the only money action: it needs a **policy** basis, a paid invoice of the customer within the policy window, amount finite, > 0, <= invoice and <= cap, the required tier, and **at most one refund per report** (also for an officer's override).
A failed check gives the LLM a specific message and one repair; still failing, the report is **rebuilt by code** (`degraded_template`, flag `grounding_failed`, bad draft kept in the trace).
It proves claims have a basis, not that a conclusion is true; that is the officer's job. "Money figure in the feedback" is a small heuristic (`$N`, `N USD`, or a number right after a payment word; never a number followed by a time unit, part of a date or an id).
Confidence is computed by code from the classifier's confidence minus penalties (unverified customer, no policy, injection signs, step cap, degradation); triage, degradation or an unverified sender can never be "high". A heuristic, not a calibrated probability.

**Human review.** `pending_review -> approved | overridden -> executed`, or `rejected` (terminal, never executes). The status change and its audit row share one SQLite transaction with a conditional update, so repeating a decision replays the result and a conflicting one returns 409.
An override is validated by the same rules and the audit keeps the machine's suggestion beside the final one. The executor is a stub that prints the call it would make.

**Failure behaviour** (the three required cases, all with tests and samples):

| Failure | What the system does |
|---|---|
| **Ambiguous classification** | `needs_human_triage`, top alternatives, `request_more_info` with a `draft_clarification_question` for the officer (nothing is sent to the customer), low confidence (sample 2) |
| **Required record missing** | tool returns `not_found`; `record_found=false`, "unverified customer", no tier guessed, no refund (sample 3) |
| **LLM fails** | retry once, then a no-LLM path: keep what already succeeded, call only missing sources, build a template report (`degraded_template`, low confidence, triage) (sample 5) |
| **A tool fails** | one retry; still failing: `error` is not `not_found`, so the gate is unmet and the run ends as `incomplete_context`, naming the source and proposing only `escalate_to_human` (basis: system guideline `GL-TRIAGE`) |
| Report store fails | API returns 503; never "queued" |

**Observability.** `traces/<trace_id>.jsonl` has one line per step with fixed event names (`intake`, `classification_completed`, `llm_call`, `tool_call`, `tool_result`, `grounding_validation`, `report_generated`, `human_review`, ...);
no model reasoning is stored. `python -m feedback_agent trace <id>` prints it. Usage and cost include **failed attempts**: `counters` has `llm_calls`, `provider_calls` and `usage_complete`, which is `false`
when an attempt that could have been billed has unknown usage (undecodable 200 body, read timeout), making totals a lower bound. Connect failures and HTTP error statuses count as free
(Google documents failed 400/500 requests as not billed; other statuses are an assumption). Excerpt of sample 1 (live):

```
#03 classification_completed {"category": "service_outage", "urgency": "critical", "confidence": 0.9, "needs_human_triage": false}
#05 llm_call                 {"purpose": "agent_turn", "turn": 1, "latency_ms": 1397, "tool_calls": ["get_cs_guidelines", "lookup_customer", "search_policies"]}
#07 tool_result              {"tool": "get_cs_guidelines", "status": "found", "returned": "GL-OUT-01"}
#09 tool_result              {"tool": "lookup_customer", "status": "found", "returned": "C-1001"}
#11 tool_result              {"tool": "search_policies", "status": "found", "returned": ["POL-ESC-01", "POL-REF-03", "POL-SLA-01"]}
#13 grounding_validation     {"ok": true, "attempt": 1, "issues": []}
#14 report_generated         {"generated_by": "gemini", "confidence": "high", "actions": ["escalate_to_engineering", "escalate_to_human"], "counters": {"llm_turns": 2, "tool_calls": 3}}
```

## 1.4. Evidence: samples and evaluation

- [`samples/`](samples/README.md): 5 inputs (happy path, ambiguous, unknown customer, abusive + injection, LLM/tool failure), each with report, trace and metadata from a **live Gemini** run and an **offline scripted** replay.
  Sample 5 is **not** written by Gemini: failures were injected on purpose (documented there).
- [`eval/`](eval/RESULTS.md): 14 labelled cases scored by code (category, urgency, allowed/forbidden actions, triage, expected sources retrieved, fabricated references, injection, caps, trace completeness, latency). Only measured numbers, misses listed;
  with so few cases they describe this run set, not production quality. `python -m feedback_agent eval` needs a key and exits non-zero if any run crashes.
- **Cap sweep** (14 cases x 3 runs per setting, live model, `max_tool_calls` = 6):

  | `max_llm_turns` | normal completion | avg LLM turns | latency p50 / p95 |
  |---|---|---|---|
  | 2 | 42/42 | 2.00 | 5.5 s / 11.6 s |
  | 3 | 42/42 | 2.07 | 5.5 s / 11.2 s |
  | 4 | 42/42 | 2.07 | 5.4 s / 11.8 s |
  | 6 | 42/42 | 2.02 | 5.5 s / 10.9 s |

  A normal run is 2 turns (three lookups in parallel, then `submit_report`) and 2-6 behaved the same here. **Default 4** = the typical 2 + one repair + one spare; the cap is only a ceiling. 14 cases cannot show when a higher cap would matter.
- **Live findings:** Gemini 3 thinking tokens count against `maxOutputTokens`: at 2048 the long `submit_report` call was truncated or malformed on 2 of 14 cases. `MAX_OUTPUT_TOKENS=8192`, `LLM_THINKING_LEVEL=low` and one retry for malformed calls fixed it
  (14/14 normal; median latency about 13 s to 5 s). Saved live outputs use that configuration; later hardening changed validation and error handling, not prompts, and the stored reports still pass the current validator.

## 1.5. Scaling the knowledge base (not built here)

Built: effective-date and supersession filtering, keyword scoring, hard `top_k`, `not_found` as a first-class answer, citations checked against what was retrieved. For a large corpus: chunk by clause with title/section path and filterable metadata (validity, authority, scope);
stable ids that exclude dates, every version its own row; keyword + vector search with reciprocal-rank fusion and an optional reranker; validity filter pushed into the query; deterministic conflict rules in code (higher authority > lower, specific > general, newer > older) with unresolved conflicts flagged to a human;
caching with re-index invalidation; and recall@k measured separately from answer quality, including date-dependent cases.

## 1.6. Layout and criteria

```
feedback_agent/  pipeline.py · agent.py (loop, executor, caps) · validator.py · report.py · classify.py · tools.py · guard.py · hitl.py · api.py · cli.py
                 trace.py · evaluation.py · samples.py · store.py · llm/ (Protocol, Gemini, scripted, factory) · prompts/*.v1.md (version goes into traces and reports)
data/ (guidelines, policies, seed CSVs)   samples/   eval/   tests/ (~60 short offline tests)
```

A new LLM provider is one class satisfying `LLMProvider` plus one line in `PROVIDERS`; only Gemini and the scripted provider exist on purpose.

| Criterion | Where to look |
|---|---|
| Agent design | Architecture section; `agent.py`; WRITEUP Q1; traces of samples 1-2 show the LLM choosing calls |
| Grounding | `validator.py`, `report.py`, `tests/test_validator.py`; eval "fabricated references"; `suggested_action.basis` |
| Tool use | `tools.py` (typed args, found/not_found/error); `tool_call`/`tool_result` trace events |
| Robustness | Failure behaviour; `tests/test_robustness.py`; samples 2, 3, 5 |
| Code quality | small modules, typed schemas, no framework, short tests |
| Communication | this file and [`WRITEUP.md`](WRITEUP.md) |
| Judgment | WRITEUP Q3; UI, auth, multi-turn, batch and real integrations left out |

## 1.7. Notes

Ideas reused from my earlier personal project "logilens" (a log-analytics assistant): the provider interface and fallback idea, converting a Pydantic JSON Schema to a Gemini function declaration (rewritten here), the keyword-rule fallback idea, JSONL audit logging.
The tool loop, grounding, review flow, eval and usage tracking were written for this task; no other third-party code was copied.

AI assistance: I used AI tools while building this. The architecture and design decisions are mine, and I reviewed and verified the results.
