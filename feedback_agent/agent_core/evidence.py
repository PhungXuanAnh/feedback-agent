"""Evidence of one run: everything the tools returned. The grounding validator and the report
may only rely on this set (plus the system-evidence guideline GL-TRIAGE)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from .tools import ToolResult, ToolStatus

REQUIRED_SOURCES = ["guidelines", "customer", "policies"]
SYSTEM_GUIDELINE = "GL-TRIAGE"


@dataclass
class Evidence:
    as_of: date
    guidelines: dict[str, dict] = field(default_factory=dict)
    policies: dict[str, dict] = field(default_factory=dict)
    customer: Optional[dict] = None
    consulted: set[str] = field(default_factory=set)  # found OR not_found
    source_errors: dict[str, str] = field(default_factory=dict)
    system_evidence: dict[str, dict] = field(
        default_factory=dict
    )  # GL-TRIAGE: not one of the 3 sources

    # ------------------------------------------------------------ recording
    def record(self, tool_result: ToolResult) -> None:
        """Store one tool result: found/not_found mark the source consulted, error is only noted.

        Only the 3 required sources count; a later success clears an earlier error.
        """
        if tool_result.source not in REQUIRED_SOURCES:
            return
        if tool_result.status is ToolStatus.error:
            self.source_errors[tool_result.source] = tool_result.error or "error"
            return
        self.consulted.add(tool_result.source)
        self.source_errors.pop(tool_result.source, None)
        if tool_result.status is ToolStatus.found and tool_result.data:
            if tool_result.source == "guidelines":
                self.guidelines[tool_result.data["guideline_id"]] = tool_result.data
            elif tool_result.source == "customer":
                self.customer = tool_result.data
            else:
                self.policies.update(
                    {policy["policy_id"]: policy for policy in tool_result.data["policies"]}
                )

    # ------------------------------------------------------------- queries
    @property
    def missing_sources(self) -> list[str]:
        """Required sources not consulted yet, e.g. ["policies"]; submit needs it empty."""
        return [source for source in REQUIRED_SOURCES if source not in self.consulted]

    @property
    def customer_found(self) -> bool:
        """True when lookup_customer returned a record (not_found leaves it False)."""
        return self.customer is not None

    def guideline(self, guideline_id: str) -> Optional[dict]:
        """Guideline fetched in this run, or the system one (GL-TRIAGE); None if unknown."""
        return self.guidelines.get(guideline_id) or self.system_evidence.get(guideline_id)

    def policy(self, policy_id: str) -> Optional[dict]:
        """Policy returned by search_policies in this run; None if the LLM never saw it."""
        return self.policies.get(policy_id)

    def record_by_id(self, ident: str) -> Optional[dict]:
        """Guideline or policy with this id; the validator checks every cited basis id with it."""
        return self.guideline(ident) or self.policy(ident)

    def citable_ids(self) -> set[str]:
        """Every guideline and policy id a report may cite."""
        return set(self.guidelines) | set(self.policies) | set(self.system_evidence)

    def money_values(self) -> set[str]:
        """Money figures known from records: invoice amounts, duplicate groups, policy caps.

        Example: 49.0 -> "49", 1249.50 -> "1249.5".
        """
        amounts: list[float] = [
            policy["max_amount"]
            for policy in self.policies.values()
            if policy.get("max_amount") is not None
        ]
        if self.customer:
            amounts += [invoice["amount"] for invoice in self.customer["recent_invoices"]]
            amounts += [group["amount"] for group in self.customer["duplicate_charge_candidates"]]
        normalized = set()
        for amount in amounts:
            text = f"{amount:.2f}".rstrip("0").rstrip(".")
            normalized.add(text)
        return normalized

    def as_text(self) -> str:
        """All evidence as one string, so the validator can look a date up in it."""
        return str(self.to_dict())

    # --------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        """JSON-ready copy, saved with the report so an override is checked against it later."""
        return {
            "as_of": self.as_of.isoformat(),
            "guidelines": self.guidelines,
            "policies": self.policies,
            "customer": self.customer,
            "consulted": sorted(self.consulted),
            "source_errors": self.source_errors,
            "system_evidence": self.system_evidence,
        }

    @classmethod
    def from_dict(cls, stored: dict[str, Any]) -> "Evidence":
        """Rebuild the Evidence saved by to_dict (used to validate an officer's override)."""
        return cls(
            as_of=date.fromisoformat(stored["as_of"]),
            guidelines=stored["guidelines"],
            policies=stored["policies"],
            customer=stored["customer"],
            consulted=set(stored["consulted"]),
            source_errors=stored["source_errors"],
            system_evidence=stored["system_evidence"],
        )
