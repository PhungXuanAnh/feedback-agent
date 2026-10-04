# Design write-up: Agentic Customer Feedback System

## 1. Why did I structure the agent this way?

A **fixed six-step pipeline with one bounded, ReAct-style loop in the middle**, not a free-roaming or multi-agent system.
The job is a linear investigation (classify, look up three independent sources, write a report), so more agents would add
cost, latency and failure modes without adding capability, and a single do-everything prompt can be neither audited nor trusted.
The principle is **LLM where language is hard, code where being wrong is expensive**:

* **LLM:** classification (one schema-forced call), deciding which tools to call and with what arguments, and drafting the report.
  Real tool calling: results are fed back into the conversation, and the model may search again.
* **Code:** the evidence gate (all three sources must be consulted; `not_found` counts, `error` does not), two enforced step caps,
  read-only tools, a grounding validator (cited ids must come from this run's tool results; every action needs a basis that allows it;
  refunds are checked against invoice, policy window, cap and tier), the confidence formula, and the review state machine.
* **Human:** the LLM only proposes; an officer approves, overrides or rejects, and only then does a stub executor act.

Classification is a hybrid (LLM structured output plus urgency floor rules and a triage flag) because rules alone are brittle, classic ML
needs labelled data we do not have, and an unguarded LLM can be confidently wrong (README has the full comparison).
Step caps are measured, not guessed: across 42 live runs per setting, a normal run takes 2 LLM turns and `max_llm_turns` 2, 3, 4 and 6 gave the same outcomes
here, so I set the default to 4 (typical 2 turns + a repair + a spare) as a cost ceiling, knowing 14 cases cannot show when a higher cap would matter.

## 2. What would I change to make it production-ready?

* **Reliability:** real CRM/KB behind the same typed tool interfaces (or MCP), timeouts and circuit breakers, a model fallback chain, and a queue with an
  outbox so that "approved" and "executed" are exactly-once against external systems (today only the review step is transactional).
* **Cost and latency:** tune caps and `thinking` level per step (low thinking already cut median latency from about 13 s to 5 s), a smaller model for
  classification, prompt caching, async/batch intake, and per-request cost alerts (cost is reported only when prices are configured; failed attempts are counted and an incomplete usage is flagged).
* **Evaluation:** a much larger labelled set from real tickets, retrieval recall@k, an LLM judge for summaries calibrated against officers, a CI gate per prompt version, and
  the **override rate** as the standing quality signal. Calibrate confidence; today it is a heuristic.
* **Security and governance:** authenticate the sender (the current scope lock only matches metadata), PII redaction and retention for traces, a prompt-injection
  classifier plus a red-team suite, tiered review (auto-approve log-only; always a human for money, legal, security).
* **Retrieval at scale:** metadata pre-filters (as-of date, authority, scope), hybrid keyword + vector search with a reranker, deterministic conflict rules, caching and recall measurement (README).

## 3. What did I deliberately leave out, and why?

UI (not graded; the API covers it) · real multi-turn clarification (instead the report carries a question *for the officer*; true multi-turn needs conversation
state, a channel and handling of injection through replies) · batch rollups · multi-agent or agent frameworks · a zoo of providers (two prove the seam) ·
a trained classifier (no data) · heavier injection defences such as a judge model, canary tokens or a dual-LLM design (layered cheap defences plus the validator instead) ·
Postgres or a vector DB (SQLite and JSON are enough for 12 policies) · real auth · an `apply_credit` action (every money action must be fully checked, so only `issue_refund` exists) ·
target metrics (14 labelled cases cannot support quality claims; the eval reports measured numbers and lists misses).
