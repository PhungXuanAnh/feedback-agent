# Design details and reference

Details behind the summaries in the [README](../README.md) (sections [1.4](../README.md#14-design-decisions-and-safeguards), [1.6](../README.md#16-api-and-configuration-reference) and [1.7](../README.md#17-scaling-the-knowledge-base-not-built-here)). For the reasoning and trade-offs in one page, read [`WRITEUP.md`](../WRITEUP.md).

- [Injection guard and layered defences](#injection-guard-and-layered-defences)
- [Classification](#classification)
- [Agent loop and step caps](#agent-loop-and-step-caps)
- [Grounding validator](#grounding-validator)
- [Human review](#human-review)
- [Observability and cost tracking](#observability-and-cost-tracking)
- [HTTP API reference](#http-api-reference)
- [Scaling the knowledge base (not built here)](#scaling-the-knowledge-base-not-built-here)

## Injection guard and layered defences

Pydantic validation of text and metadata, then regexes flag known injection patterns before the LLM sees the text (flags only; easy to evade, so never the only defence). The layers:

- role separation between instructions and customer text;
- escaped delimiters around customer text (`<customer_feedback>`) **and every tool result** (`<tool_result>`), so neither can fake the other's tags or a system message;
- schema-forced output;
- read-only tools with allow-listed arguments;
- customer lookup locked to the sender (else `out_of_scope` plus a flag);
- the validator and the review state machine decide what can actually happen.

None is 100%, so the design limits damage instead of only trying to detect.

## Classification

Hybrid, because rules are brittle, classic ML needs labelled data, and a bare LLM can be confidently wrong. The schema forces valid values (category, up to 2 alternatives, sentiment, urgency, confidence); then code applies urgency floors
(security/legal wording or an Enterprise outage means at least `high`; the LLM can only raise urgency) and flags `needs_human_triage` when confidence < 0.65, the category is `unclear` or a second reading has confidence >= 0.4.
Thresholds are reasoned (conservative), not tuned on the small eval set. If the call fails twice, keyword rules classify instead.

## Agent loop and step caps

```
 LLM turn ──┬─ retrieval tools ─► executor (validate args, cache, budget, 1 retry) ─► results back to the LLM ─┐
            └─ submit_report ─► evidence gate (3 sources consulted?) ─► grounding validator ─► valid ─► report
                                     │ missing: tell the LLM            │ invalid: tell the LLM, one repair;
                                     ▼                                  ▼ still invalid: rebuild from validated data
                                the last turn is reserved for submit_report (forced via tool_choice)
```

The evidence gate refuses `submit_report` until guidelines, customer and policies were all consulted. Each tool returns `found`, `not_found` (asked, nothing there: counts as consulted) or `error` (source failed: does **not** count).

Two separate caps: `max_llm_turns` (agent-loop turns, default 4, includes the forced-submit and repair turns; classify is counted separately) and `max_tool_calls` (default 6; cache hits and `submit_report` are free).
The last turn is reserved for `submit_report`; if sources are still missing the system runs those lookups itself (`forced_by_system`, max 3, traced), so gate and cap cannot deadlock.
Worst case per request: 1 classify + 4 agent turns, each retried once, so up to 10 provider calls (all counted). The sweep behind the default of 4 is in the README ([1.5](../README.md#15-evaluation-and-limitations)).

## Grounding validator

`validator.py` checks the report against this run's tool results:

- cited ids exist;
- `customer_context` matches the record (no record means `record_found=false`, never an invented tier);
- money figures and ISO dates in the summary come from the feedback, an invoice of this customer or a policy limit;
- every action has a `basis` whose `allowed_actions` includes that action type.

`issue_refund` is the only money action: it needs a **policy** basis, a paid invoice of the customer within the policy window, an amount that is finite, > 0, <= invoice and <= cap, the required tier, and **at most one refund per report** (also for an officer's override).
A failed check gives the LLM a specific message and one repair; still failing, the report is **rebuilt by code** (`degraded_template`, flag `grounding_failed`, the bad draft kept in the trace).

It proves that claims have a basis, not that a conclusion is true; that is the officer's job. "Money figure in the feedback" is a small heuristic (`$N`, `N USD`, or a number right after a payment word; never a number followed by a time unit, part of a date or an id).
Confidence is computed by code from the classifier's confidence minus penalties (unverified customer, no policy, injection signs, step cap, degradation); triage, degradation or an unverified sender can never be "high". It is a heuristic, not a calibrated probability.
`confidence_score` is that number before any cap; `confidence_level` is the label after it. A run that needs human triage, degraded, failed grounding, hit the step cap or lacks a required source is forced to `low`, so a report can show a score such as 0.75 with level `low` (this is intentional).

## Human review

`pending_review -> approved | overridden -> executed`, or `rejected` (terminal, never executes). The status change and its audit row share one SQLite transaction with a conditional update, so repeating a decision replays the result and a conflicting one returns 409.
An override is validated by the same rules and the audit keeps the machine's suggestion beside the final one. The executor is a stub that prints the call it would make.

## Observability and cost tracking

`traces/<trace_id>.jsonl` has one line per step with fixed event names (`intake`, `classification_completed`, `llm_call`, `tool_call`, `tool_result`, `grounding_validation`, `report_generated`, `human_review`, ...); no model reasoning is stored.
Usage and cost include **failed attempts**: `counters` has `llm_calls`, `provider_calls` and `usage_complete`, which is `false` when an attempt that could have been billed has unknown usage (undecodable 200 body, read timeout), making totals a lower bound.
Connect failures and HTTP error statuses count as free (Google documents failed 400/500 requests as not billed; other statuses are an assumption). Cost is estimated only when `PRICE_PER_MTOK_IN/OUT` are set, otherwise it is `unknown`, never 0.

## HTTP API reference

Swagger UI at `/docs`. Calls follow the order of a real session: submit, find the report in the queue, decide, inspect how it was produced.

| Call | What it does |
|---|---|
| `GET /health` | Liveness check, always open (no token). |
| `POST /feedback` | Submits **one** customer message and runs the whole pipeline synchronously. Returns the finished report with its `report_id` and `trace_id`, status `pending_review`; takes several seconds because it calls the LLM. Body: `text`, `customer_email`, `channel`, optional `received_at` (default now; it also fixes which policies are in force). The only call that spends LLM tokens and the only one rate-limited (`429`). `422` for invalid input, `503` if the report could not be stored. |
| `GET /reports?status=pending_review` | The human review queue: a summary per report (id, category, urgency, confidence, whether it needs human triage). Omit `status` to list all. |
| `GET /reports/<report_id>` | One full report: classification, evidence cited, proposed actions with their basis, and the review history. |
| `POST /reports/<report_id>/review` | The officer's decision on a pending report: `approve` (run the proposed actions), `override` (change category, urgency or actions via `overrides`, then run those) or `reject` (do nothing). Approved actions go to a **stub** executor that only records what it would call (no real refund). The first decision wins: repeating it is a harmless replay, a conflicting one gets `409`. |
| `GET /traces/<trace_id>` | The step-by-step audit trail of that run (every LLM turn, tool call, validation and token count). |

## Scaling the knowledge base (not built here)

Built: effective-date and supersession filtering, keyword scoring, hard `top_k`, `not_found` as a first-class answer, citations checked against what was retrieved. For a large corpus:

- chunk by clause with title/section path and filterable metadata (validity, authority, scope); stable ids that exclude dates, every version its own row;
- keyword + vector search with reciprocal-rank fusion and an optional reranker; the validity filter pushed into the query;
- deterministic conflict rules in code (higher authority > lower, specific > general, newer > older) with unresolved conflicts flagged to a human;
- caching with re-index invalidation, and recall@k measured separately from answer quality, including date-dependent cases.
