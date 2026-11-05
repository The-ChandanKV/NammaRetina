"""Compatibility entry point for current offline checks; never writes live data."""
import unittest


def run_checks():
    suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_progression_lime_eval.SimulationTests'])
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


if __name__ == "__main__":
    raise SystemExit(0 if run_checks() else 1)
