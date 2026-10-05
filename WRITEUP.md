# Design write-up: Agentic Customer Feedback System

## 1. Why did I structure the agent this way?

A **fixed six-step pipeline with one bounded, ReAct-style loop in the middle**, not a free-roaming or multi-agent system. The job is a linear investigation (classify, look up three independent sources, write a report), so more agents would add cost, latency and failure modes without adding capability. The principle: **LLM where language is hard, code where being wrong is expensive.**

* **LLM:** classification (one schema-forced call), choosing tool calls and arguments, drafting the report. Tool results go back into the conversation and the model may search again.
* **Code:** the evidence gate (all three sources consulted; `not_found` counts, `error` does not), two enforced step caps, read-only tools, the grounding validator (cited ids must come from this run's tool results; every action needs a guideline or policy that permits it; refunds are checked against invoice, policy window, cap and tier), the confidence formula and the review state machine.
* **Human:** the LLM only proposes; an officer approves, overrides or rejects, and only then does a stub executor act.

Classification is a hybrid (LLM structured output plus urgency-floor rules and a triage flag): rules alone are brittle, a classic classifier needs more labelled data than I have, and an unguarded LLM can be confidently wrong. Over 42 live runs per setting, `max_llm_turns` 2, 3, 4 and 6 behaved the same, so the default is 4 (2 typical turns, one repair, one spare) as a cost ceiling; 14 cases cannot show when a higher cap would matter.

## 2. What would I change to make it production-ready?

* **Reliability:** a real CRM and knowledge base behind the same typed tool interfaces, timeouts, circuit breakers, a model fallback chain, and an outbox with idempotency keys so "approved" and "executed" cannot run twice.
* **Cost and latency:** tune caps and thinking level per step (low thinking already cut median latency from about 13 s to 5 s), a smaller model for classification, prompt caching, async or batch intake.
* **Evaluation:** a much larger labelled set from real tickets, retrieval recall@k, an LLM judge calibrated against officers, a CI gate per prompt version, and the **override rate** as the standing quality signal. Confidence is a heuristic today and needs calibration.
* **Security and scale:** authenticate the sender, redact PII in traces, add an injection classifier and red-team suite, and keep a human in the loop for money, legal and security. For a large knowledge base: metadata pre-filters, hybrid search with a reranker, deterministic conflict rules (README section [1.7](README.md#17-scaling-the-knowledge-base-not-built-here)).

## 3. What did I deliberately leave out, and why?

* **UI and multi-turn:** no UI (the API covers it); the report carries a clarification question *for the officer*, because true multi-turn needs conversation state, a channel and injection handling for replies.
* **Infrastructure:** no multi-agent setup, agent framework, extra providers (a live and a scripted one demonstrate the interface), trained classifier (too little labelled data to train a useful one), or Postgres/vector DB (SQLite and JSON suffice for 12 policies).
* **Assurance:** no heavier injection defences (layered cheap ones plus the validator instead), no `apply_credit` action (every money action must be fully checked, so only `issue_refund` exists), and no target metrics: 14 labelled cases cannot support quality claims, so the eval reports measured numbers and lists misses.
