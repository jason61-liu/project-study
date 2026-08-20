from __future__ import annotations

import unittest

from deep_research_agent.eval_harness import FAULT_SCENARIOS, FUNCTIONAL_TASKS, SECURITY_SCENARIOS
from deep_research_agent.retrieval import decompose
from deep_research_agent.sandbox import Sandbox


class RetrievalAndSandboxTest(unittest.TestCase):
    def test_decomposition_splits_compound_question(self) -> None:
        self.assertEqual(["如何检索", "如何核验冲突"], decompose("如何检索；如何核验冲突？"))

    def test_sandbox_denies_non_allowlisted_command(self) -> None:
        with self.assertRaises(PermissionError):
            Sandbox().execute(["sh", "-c", "echo unsafe"])

    def test_sandbox_executes_argv_without_shell(self) -> None:
        result = Sandbox().execute(["python3", "-c", "print(6 * 7)"])
        self.assertEqual(0, result.returncode)
        self.assertEqual("42", result.stdout.strip())

    def test_eval_inventory_meets_acceptance_counts(self) -> None:
        self.assertGreaterEqual(len(FUNCTIONAL_TASKS), 50)
        self.assertGreaterEqual(len(SECURITY_SCENARIOS), 15)
        self.assertGreaterEqual(len(FAULT_SCENARIOS), 10)


if __name__ == "__main__":
    unittest.main()

