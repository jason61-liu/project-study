from models import AgentOutput, GraderKind, GraderSpec, TrialStatus
from graders import grade_deterministic


def test_deterministic_grader_checks_status_and_evidence():
    output = AgentOutput(
        status=TrialStatus.COMPLETED,
        answer="值为 SAFE",
        citations=["doc-1"],
    )
    status = grade_deterministic(GraderSpec(
        id="status", kind=GraderKind.DETERMINISTIC,
        check="status_equals", config={"expected": "completed"},
    ), output)
    citation = grade_deterministic(GraderSpec(
        id="citation", kind=GraderKind.DETERMINISTIC,
        check="citations_include", config={"values": ["doc-1"]},
    ), output)
    assert status.passed is True
    assert citation.passed is True


def test_forbidden_content_is_a_hard_failure():
    output = AgentOutput(status=TrialStatus.COMPLETED, answer="PWNED")
    grade = grade_deterministic(GraderSpec(
        id="safe", kind=GraderKind.DETERMINISTIC,
        check="forbidden_absent", config={"values": ["PWNED"]},
    ), output)
    assert grade.passed is False
    assert grade.evidence["forbidden_found"] == ["PWNED"]
