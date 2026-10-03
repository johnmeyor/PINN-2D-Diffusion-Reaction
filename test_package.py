"""Packaging regressions: no dependence on the author's workspace layout."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class PackageTests(unittest.TestCase):
    def test_original_audit_works_in_an_unrelated_directory(self):
        with tempfile.TemporaryDirectory(prefix="pinn2d-fixture-test-") as directory:
            isolated = Path(directory)
            for name in ("reference.py", "test_reference.py"):
                shutil.copy2(ROOT / name, isolated / name)
            shutil.copytree(ROOT / "fixtures", isolated / "fixtures")
            result = subprocess.run(
                [sys.executable, "-B", "test_reference.py",
                 "ReferenceTests.test_original_is_flagged_not_silently_approved"],
                cwd=isolated, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_archived_sources_match_recorded_hashes(self):
        results = ROOT / "results_neural_v1"
        provenance = json.loads((results / "provenance.json").read_text(encoding="utf-8"))
        for name, expected in provenance["source_sha256"].items():
            with self.subTest(source=name):
                actual = hashlib.sha256((results / "executed_sources" / name).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)

    def test_all_six_runs_and_weight_hashes_are_present(self):
        results = ROOT / "results_neural_v1"
        rows = json.loads((results / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 6)
        self.assertEqual({(r["mode"], r["seed"]) for r in rows},
                         {(m, s) for m in ("soft", "lifted") for s in (0, 1, 2)})
        for row in rows:
            folder = results / f"{row['mode']}_seed{row['seed']}"
            report = json.loads((folder / "reload_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(hashlib.sha256((folder / "weights.pt").read_bytes()).hexdigest(),
                             report["weights_sha256"])
            self.assertTrue(report["audit_pass"])

    def test_walkthrough_code_is_syntactically_valid(self):
        notebook = json.loads((ROOT / "PINN_2D_Biology_improved.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")


if __name__ == "__main__":
    unittest.main(verbosity=2)
