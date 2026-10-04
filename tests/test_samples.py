from feedback_agent.config import Settings
from feedback_agent.samples import load_manifest, run_one


def test_scripted_samples_behave_as_designed():
    """The 5 stored samples can be replayed offline (no key) and still meet their stated expectations."""
    for sample in load_manifest():
        _, meta = run_one(sample, "scripted", Settings())
        assert meta["expectations_met"], (sample["name"], meta["unmet_expectations"])
