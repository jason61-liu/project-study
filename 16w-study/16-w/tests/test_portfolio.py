from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from portfolio.bootstrap import load_week15
from portfolio.controls import AuthorizationError, TenantMemory, TokenClaims, TokenVerifier, scan_public_artifact
from portfolio.feedback import simulate_feedback
from portfolio.release import simulate_release
from portfolio.strengthening import run_trials
from portfolio.verify import fault_scenarios, security_scenarios

load_week15()

from deep_research_agent.agent import ResearchAgent  # noqa: E402
from deep_research_agent.store import SQLiteStore  # noqa: E402


class PortfolioTest(unittest.TestCase):
    def test_standard_benchmark_artifact_is_explicit_contract_replay(self) -> None:
        import json

        summary = json.loads((Path(__file__).parents[1] / "results/tau3-adapter/summary.json").read_text())
        self.assertEqual(5, summary["task_count"])
        self.assertEqual(1.0, summary["strict_success_rate"])
        self.assertIn("gold-contract-replay", summary["agent_version"])

    def test_every_task_runs_three_trials(self) -> None:
        rows, summary = run_trials(3)
        self.assertEqual(165, len(rows))
        self.assertEqual(3, summary["trials_per_task"])
        self.assertEqual(1.0, summary["success_rate_mean"])

    def test_security_and_fault_suites_execute(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            agent = ResearchAgent(SQLiteStore(directory / "verify.db"))
            security = security_scenarios(agent)
            faults = fault_scenarios(agent, directory)
            agent.store.close()
        self.assertGreaterEqual(len(security), 15)
        self.assertGreaterEqual(len(faults), 10)
        self.assertTrue(all(security.values()))
        self.assertTrue(all(faults.values()))

    def test_feedback_offline_and_online_move_same_direction(self) -> None:
        result = simulate_feedback()
        self.assertTrue(result["direction_check"]["same_direction"])
        self.assertGreater(len(result["failed_samples_for_curation"]), 0)

    def test_regression_is_blocked_and_rolled_back(self) -> None:
        result = simulate_release(0.94)
        self.assertTrue(result["candidate_blocked"])
        self.assertTrue(result["rollback_verified"])

    def test_token_validation_and_revocation(self) -> None:
        verifier = TokenVerifier("issuer", "audience")
        claims = TokenClaims("issuer", "audience", "t", "u", ("research:run",), 200, "token-1")
        self.assertEqual("t", verifier.verify(claims, "research:run", now=100)["tenant_id"])
        verifier.revoke("token-1")
        with self.assertRaises(AuthorizationError):
            verifier.verify(claims, "research:run", now=100)

    def test_memory_isolation_update_export_and_delete(self) -> None:
        memory = TenantMemory()
        memory.put_fact("a", "f", "v1", "s1", True)
        memory.put_fact("a", "f", "v2", "s2", True)
        self.assertEqual(2, memory.get_fact("a", "f")["version"])
        self.assertIsNone(memory.get_fact("b", "f"))
        self.assertIn("f", memory.export("a")["facts"])
        memory.delete_tenant("a")
        self.assertIsNone(memory.get_fact("a", "f"))

    def test_public_demo_scan(self) -> None:
        self.assertEqual([], scan_public_artifact("synthetic public demo"))
        self.assertTrue(scan_public_artifact("credential sk-abcdefghijklmnop"))


if __name__ == "__main__":
    unittest.main()
