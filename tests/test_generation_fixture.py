#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_generation_fixture as checker
import emit_generation_fixtures as reference


class GenerationFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = ROOT / "tests/fixtures/generation.json"
        cls.document = checker.audit(cls.fixture)

    def reject(self, mutation, message, *, rehash=False):
        document = copy.deepcopy(self.document)
        mutation(document)
        if rehash:
            document["payload_sha256"] = checker.payload_digest(document)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation.json"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, message):
                checker.audit(path)

    def test_offline_without_torch_or_sources(self):
        code = '''
import importlib.abc, sys, urllib.request
from pathlib import Path
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'numpy', 'transformers', 'safetensors'):
            raise AssertionError('forbidden optional dependency ' + fullname)
sys.meta_path.insert(0, NoTorch())
def reject(*args, **kwargs):
    raise AssertionError('network access during audit')
urllib.request.urlopen = reject
sys.path.insert(0, sys.argv[1])
import check_generation_fixture as checker
checker.audit(Path(sys.argv[2]))
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, "-S", "-c", code,
                                 str(ROOT / "tools"), str(self.fixture)],
                                cwd=tempfile.gettempdir(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_literal_random_vectors(self):
        self.assertEqual(reference.splitmix64(0)[1], 0xE220A8397B1DCDAF)
        for vector in reference.RNG_BOUNDARIES:
            _, word, uniform = reference.splitmix64(vector["seed"])
            self.assertEqual(word, int(vector["output_hex"], 16))
            self.assertEqual(uniform, vector["uniform"])

    def test_cdf_and_filter_boundaries(self):
        cases = {case["name"]: case["expected"] for case in self.document["sampler"]}
        self.assertEqual(cases["cdf_zero"]["token_id"], 0)
        self.assertEqual(cases["cdf_half"]["token_id"], 1)
        self.assertEqual(cases["cdf_maximum"]["token_id"], 1)
        self.assertEqual(cases["top_p_exact_prefix"]["support_ids"], [0, 1])
        self.assertEqual(cases["top_p_crossing_token"]["support_ids"], [0, 1])
        self.assertEqual(cases["top_p_singleton"]["support_ids"], [0])
        self.assertEqual(cases["top_k_stable_ties"]["support_ids"], [0, 1])
        self.assertEqual(cases["subnormal_mass_retained"]["masses"][1], float.fromhex("0x0.0000000000001p-1022"))
        self.assertEqual(cases["greedy_tie"]["state_hex"], "000000000000002a")
        self.assertNotEqual(cases["top_k_one_consumes_draw"]["state_hex"], "000000000000002a")

    def test_native_sequences_and_stop_precedence(self):
        cases = {case["name"]: case for case in self.document["generation"]}
        self.assertEqual(cases["eos_immediate"]["generated_ids"], [6])
        self.assertEqual(cases["eos_multistep"]["generated_ids"], [14, 2, 6])
        self.assertEqual(cases["stored_only_prompt"]["generated_ids"], [14, 2, 6])
        self.assertEqual(cases["eos_in_prompt"]["generated_ids"], [11, 19, 0, 24, 13, 1, 17, 35, 27, 26, 6])
        self.assertEqual(cases["zero_output"]["steps"], [])
        self.assertEqual(cases["zero_output"]["stop_reason"], "length")
        self.assertEqual(cases["full_context"]["steps"], [])
        self.assertEqual(cases["full_context"]["stop_reason"], "context")
        self.assertEqual(cases["context_cutoff"]["stop_reason"], "context")
        self.assertEqual(cases["length_context_tie"]["stop_reason"], "length")
        self.assertEqual(cases["eos_length_context_tie"]["stop_reason"], "eos")

    def test_wrong_sampler_draw(self):
        self.reject(lambda d: d["sampler"][0]["expected"].__setitem__("state_hex", "wrong"),
                    "independent sampler", rehash=True)

    def test_wrong_generation_stop(self):
        self.reject(lambda d: d["generation"][0].__setitem__("stop_reason", "length"),
                    "generation steps or outcome", rehash=True)

    def test_wrong_native_prefix(self):
        self.reject(lambda d: d["generation"][0]["steps"][0].__setitem__("prefix_ids", [9]),
                    "evaluated prefix", rehash=True)

    def test_payload_corruption(self):
        self.reject(lambda d: d["generation"][0]["steps"][0]["logits"].__setitem__(0, 99),
                    "payload digest")

    def test_missing_native_case(self):
        self.reject(lambda d: d["generation"].pop(), "generation coverage", rehash=True)

    def test_wrong_source_revision(self):
        self.reject(lambda d: d["references"]["transformers"].__setitem__("revision", "wrong"),
                    "source provenance", rehash=True)

    def test_dependency_changed(self):
        with patch.object(checker, "dependency_hashes", return_value={}):
            with self.assertRaisesRegex(ValueError, "dependency changed"):
                checker.audit(self.fixture)

    def test_wrong_original_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "original miniature model manifest"):
                reference.authenticated_model(path)

    def test_missing_model_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_bytes((ROOT / "tests/fixtures/model/manifest.json").read_bytes())
            with self.assertRaises(FileNotFoundError):
                reference.authenticated_model(path)

    def test_generation_comparison_rejects_each_observation(self):
        expected = next(case for case in self.document["generation"] if case["steps"])
        checker.compare_generation(expected, expected)
        for key, value in (("generated_ids", []), ("stop_reason", "wrong"), ("steps", [])):
            actual = copy.deepcopy(expected)
            actual[key] = value
            with self.assertRaises(ValueError):
                checker.compare_generation(actual, expected)
        actual = copy.deepcopy(expected)
        actual["steps"][0]["token_id"] = -1
        with self.assertRaisesRegex(ValueError, "selected token"):
            checker.compare_generation(actual, expected)
        for value in (99.0, float("nan")):
            actual = copy.deepcopy(expected)
            actual["steps"][0]["logits"][0] = value
            with self.assertRaises(ValueError):
                checker.compare_generation(actual, expected)

    def test_sampler_comparison_rejects_each_observation(self):
        expected = self.document["sampler"][0]["expected"]
        checker.compare_sampler(expected, expected)
        for key, value in (("token_id", -1), ("state_hex", "wrong"), ("support_ids", []),
                           ("probabilities", [0.0] * len(expected["probabilities"]))):
            actual = copy.deepcopy(expected)
            actual[key] = value
            with self.assertRaises(ValueError):
                checker.compare_sampler(actual, expected)

    def test_generation_comparison_requires_actual_count(self):
        expected = self.document["generation"][0]
        for value in (expected["generated_count"] + 1, None, True, 1.0):
            with self.subTest(value=value):
                actual = copy.deepcopy(expected)
                actual["generated_count"] = value
                with self.assertRaisesRegex(ValueError, "generated_count"):
                    checker.compare_generation(actual, expected)
        actual = copy.deepcopy(expected)
        del actual["generated_count"]
        with self.assertRaisesRegex(ValueError, "generated_count"):
            checker.compare_generation(actual, expected)

    def test_generation_comparison_requires_each_effective_option(self):
        expected = self.document["generation"][0]
        for key, target in expected["effective"].items():
            for missing in (False, True):
                with self.subTest(key=key, missing=missing):
                    actual = copy.deepcopy(expected)
                    if missing:
                        del actual["effective"][key]
                    else:
                        actual["effective"][key] = target + 1
                    with self.assertRaisesRegex(ValueError, "effective " + key):
                        checker.compare_generation(actual, expected)
        for value in (None, []):
            actual = copy.deepcopy(expected)
            actual["effective"] = value
            with self.assertRaisesRegex(ValueError, "effective options"):
                checker.compare_generation(actual, expected)
        actual = copy.deepcopy(expected)
        del actual["effective"]
        with self.assertRaisesRegex(ValueError, "effective options"):
            checker.compare_generation(actual, expected)

    def test_generation_comparison_preserves_exact_uint64_seed(self):
        expected = copy.deepcopy(self.document["generation"][0])
        for seed in (0, (1 << 53) + 1, (1 << 64) - 1):
            expected["effective"]["seed"] = seed
            for value in (seed, str(seed)):
                actual = copy.deepcopy(expected)
                actual["effective"]["seed"] = value
                checker.compare_generation(actual, expected)
        for value in (True, False, float((1 << 64) - 1), -1, 1 << 64,
                      "18446744073709551616", "01", "+1", "-1", "1.0", " 1", "١", ""):
            with self.subTest(value=value):
                actual = copy.deepcopy(expected)
                actual["effective"]["seed"] = value
                with self.assertRaisesRegex(ValueError, "effective seed"):
                    checker.compare_generation(actual, expected)

    def test_sampler_comparison_requires_masses(self):
        expected = self.document["sampler"][0]["expected"]
        actual = copy.deepcopy(expected)
        del actual["masses"]
        with self.assertRaisesRegex(ValueError, "sampler masses"):
            checker.compare_sampler(actual, expected)

    def test_sampler_comparison_rejects_invalid_mass_values(self):
        expected = self.document["sampler"][0]["expected"]
        for key in ("masses", "probabilities"):
            for values in ([], [float("nan")], [float("inf")], [-1.0], [0.5], [True]):
                with self.subTest(key=key, values=values):
                    actual = copy.deepcopy(expected)
                    actual[key] = values
                    with self.assertRaisesRegex(ValueError, "sampler " + key):
                        checker.compare_sampler(actual, expected)

    def test_sampler_comparison_preserves_zero_and_positive_values(self):
        subnormal = next(case["expected"] for case in self.document["sampler"]
                         if case["name"] == "subnormal_mass_retained")
        checker.compare_sampler(subnormal, subnormal)
        for key in ("masses", "probabilities"):
            actual = copy.deepcopy(subnormal)
            actual[key][1] = 0.0
            with self.assertRaisesRegex(ValueError, "sampler " + key):
                checker.compare_sampler(actual, subnormal)
            expected = copy.deepcopy(subnormal)
            expected[key][1] = 0.0
            actual = copy.deepcopy(expected)
            actual[key][1] = float.fromhex("0x0.0000000000001p-1022")
            with self.assertRaisesRegex(ValueError, "sampler " + key):
                checker.compare_sampler(actual, expected)
        actual = copy.deepcopy(subnormal)
        actual["masses"][0] -= 1e-15
        actual["probabilities"][0] -= 1e-15
        checker.compare_sampler(actual, subnormal)
        del actual["probabilities"]
        checker.compare_sampler(actual, subnormal)


if __name__ == "__main__":
    unittest.main()
