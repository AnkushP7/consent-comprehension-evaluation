"""Automated test runner for Consent Comprehension Evaluation prototype.

Runs with either unittest or pytest, with zero external test-runner dependency required.
Usage:
  python run_tests.py
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "tests"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS_DIR))


def run_all() -> int:
    print(f"Discovering and running tests in {TESTS_DIR} ...")
    loader = unittest.TestLoader()

    # Discover unittest test cases
    suite = loader.discover(start_dir=str(TESTS_DIR), pattern="test_*.py")

    # Also adapt functions in test_consent_analyzer if not wrapped in TestCase
    import test_consent_analyzer
    func_suite = unittest.TestSuite()
    for name in dir(test_consent_analyzer):
        if name.startswith("test_") and callable(getattr(test_consent_analyzer, name)):
            fn = getattr(test_consent_analyzer, name)
            class FuncTestCase(unittest.TestCase):
                def __init__(self, func, func_name):
                    super().__init__(methodName="runTest")
                    self.func = func
                    self.func_name = func_name
                def runTest(self):
                    self.func()
                def __str__(self):
                    return f"{self.func_name} (test_consent_analyzer)"
            func_suite.addTest(FuncTestCase(fn, name))

    combined = unittest.TestSuite([suite, func_suite])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined)

    print("\n" + "=" * 50)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Failures:       {len(result.failures)}")
    print(f"Errors:         {len(result.errors)}")
    print("=" * 50)

    if result.wasSuccessful():
        print("ALL TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print("SOME TESTS FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all())
