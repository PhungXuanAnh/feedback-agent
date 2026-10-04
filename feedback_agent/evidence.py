"""Evidence of one run: everything the tools returned. The grounding validator and the report
may only rely on this set (plus the system-evidence guideline GL-TRIAGE)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from .tools import SOURCES, ToolResult, ToolStatus

REQUIRED_SOURCES = ["guidelines", "customer", "policies"]
SYSTEM_GUIDELINE = "GL-TRIAGE"


@dataclass
class Evidence:
    as_of: date
    guidelines: dict[str, dict] = field(default_factory=dict)
    policies: dict[str, dict] = field(default_factory=dict)
    customer: Optional[dict] = None
    consulted: set[str] = field(default_factory=set)       # found OR not_found
    source_errors: dict[str, str] = field(default_factory=dict)
    system_evidence: dict[str, dict] = field(default_factory=dict)  # GL-TRIAGE: not one of the 3 sources

    # ------------------------------------------------------------ recording
    def record(self, r: ToolResult) -> None:
        if r.source not in REQUIRED_SOURCES:
            return
        if r.status is ToolStatus.error:
            self.source_errors[r.source] = r.error or "error"
            return
        self.consulted.add(r.source)
        self.source_errors.pop(r.source, None)
        if r.status is ToolStatus.found and r.data:
            if r.source == "guidelines":
                self.guidelines[r.data["guideline_id"]] = r.data
            elif r.source == "customer":
                self.customer = r.data
            else:
                self.policies.update({p["policy_id"]: p for p in r.data["policies"]})

    # ------------------------------------------------------------- queries
    @property
    def missing_sources(self) -> list[str]:
        return [s for s in REQUIRED_SOURCES if s not in self.consulted]

    @property
    def customer_found(self) -> bool:
        return self.customer is not None

    def guideline(self, gid: str) -> Optional[dict]:
        return self.guidelines.get(gid) or self.system_evidence.get(gid)

    def policy(self, pid: str) -> Optional[dict]:
        return self.policies.get(pid)

    def record_for(self, ident: str) -> Optional[dict]:
        return self.guideline(ident) or self.policy(ident)

    def all_ids(self) -> set[str]:
        return set(self.guidelines) | set(self.policies) | set(self.system_evidence)

    def money_values(self) -> set[str]:
        """Normalised money figures known from records: invoice amounts, duplicate groups, policy caps."""
        vals: list[float] = [p["max_amount"] for p in self.policies.values() if p.get("max_amount") is not None]
        if self.customer:
            vals += [i["amount"] for i in self.customer["recent_invoices"]]
            vals += [g["amount"] for g in self.customer["duplicate_charge_candidates"]]
        out = set()
        for v in vals:
            t = f"{v:.2f}".rstrip("0").rstrip(".")
            out.add(t)
        return out

    def numbers_text(self) -> str:
        return str(self.to_dict())

    # --------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        return {"as_of": self.as_of.isoformat(), "guidelines": self.guidelines, "policies": self.policies,
                "customer": self.customer, "consulted": sorted(self.consulted),
                "source_errors": self.source_errors, "system_evidence": self.system_evidence}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Evidence":
        return cls(as_of=date.fromisoformat(d["as_of"]), guidelines=d["guidelines"], policies=d["policies"],
                   customer=d["customer"], consulted=set(d["consulted"]), source_errors=d["source_errors"],
                   system_evidence=d["system_evidence"])
