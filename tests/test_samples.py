from types import SimpleNamespace

from feedback_agent.config import Settings
from feedback_agent.ops.samples import _provenance_note, load_manifest, run_sample


def test_scripted_samples_behave_as_designed():
    """The 5 stored samples can be replayed offline (no key) and still meet their stated expectations."""
    for sample in load_manifest():
        _, meta = run_sample(sample, "scripted", Settings())
        assert meta["expectations_met"], (sample["name"], meta["unmet_expectations"])


def test_provenance_note_says_when_a_live_run_fell_back_to_the_template():
    report = SimpleNamespace(report_generated_by="degraded_template")
    assert _provenance_note({"name": "x"}, "live", report).startswith("NOT written by the LLM")
