# Design write-up: Agentic Customer Feedback System

## 1. Why did I structure the agent this way?

A **fixed six-step pipeline with one bounded, ReAct-style loop in the middle**, not a free-roaming or multi-agent system. The job is a linear investigation
(classify, look up three independent sources, write a report), so more agents would add cost, latency and failure modes without adding capability, and a single do-everything prompt can be neither audited nor trusted.
The principle: **LLM where language is hard, code where being wrong is expensive.**

* **LLM:** classification (one schema-forced call), choosing the tool calls and their arguments, and drafting the report. Tool results are fed back into the conversation and the model may search again.
* **Code:** the evidence gate (all three sources consulted; `not_found` counts, `error` does not), two enforced step caps, read-only tools, the grounding validator
  (cited ids must come from this run's tool results; every action needs a basis that allows it; refunds are checked against invoice, policy window, cap and tier), the confidence formula and the review state machine.
* **Human:** the LLM only proposes; an officer approves, overrides or rejects, and only then does a stub executor act.

Classification is a hybrid (LLM structured output plus urgency-floor rules and a triage flag): rules alone are brittle, classic ML needs labelled data I do not have, and an unguarded LLM can be confidently wrong.
Measured runs informed the default step cap: over 42 live runs per setting a normal run took 2 LLM turns and `max_llm_turns` 2, 3, 4 and 6 behaved the same, so the default is 4 (2 typical turns, a repair, a spare) as a cost ceiling.
14 cases cannot show when a higher cap would matter.

## 2. What would I change to make it production-ready?

* **Reliability:** real CRM and knowledge base behind the same typed tool interfaces (or MCP), timeouts, circuit breakers and a model fallback chain. A queue with an outbox **plus idempotency keys on the external calls**,
  so "approved" and "executed" cannot run twice (today only the review step is transactional).
* **Cost and latency:** tune caps and thinking level per step (low thinking already cut median latency from about 13 s to 5 s), a smaller model for classification, prompt caching, async or batch intake, per-request cost alerts.
* **Evaluation:** a much larger labelled set from real tickets, retrieval recall@k, an LLM judge calibrated against officers, a CI gate per prompt version, and the **override rate** as the standing quality signal. Confidence is a heuristic today and needs calibration.
* **Security and governance:** authenticate the sender (the current scope lock only matches metadata), PII redaction and retention for traces, an injection classifier plus a red-team suite, tiered review (a human always for money, legal and security).
* **Retrieval at scale:** metadata pre-filters (date, authority, scope), hybrid keyword and vector search with a reranker, deterministic conflict rules, caching and recall measurement (README section 1.7).

## 3. What did I deliberately leave out, and why?

* **User experience:** a UI (not graded; the API covers it) and real multi-turn clarification (the report carries a question *for the officer*; true multi-turn needs conversation state, a channel and injection handling for replies).
* **Infrastructure and frameworks:** multi-agent setups and agent frameworks, a zoo of providers (two prove the seam), a trained classifier (no data), and Postgres or a vector DB (SQLite and JSON are enough for 12 policies).
* **Integration and assurance:** real auth and integrations, heavier injection defences (judge model, canary tokens, dual LLM; layered cheap defences plus the validator instead), an `apply_credit` action (every money action must be fully checked, so only `issue_refund` exists),
  and target metrics (14 labelled cases cannot support quality claims, so the eval reports measured numbers and lists misses).
