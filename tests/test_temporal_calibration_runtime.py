import unittest

from dual2pose.eval.calibrate_temporal_alignment import confidence_threshold_candidates


class TemporalCalibrationRuntimeTest(unittest.TestCase):
    """Break caught: threshold sweep cannot choose gate-all or gate-none."""

    def test_candidates_include_zero_and_value_above_maximum(self) -> None:
        actual = confidence_threshold_candidates([0.1, 0.4, 0.9], quantiles=3)
        self.assertEqual(actual[0], 0.0)
        self.assertGreater(actual[-1], 0.9)
        self.assertEqual(actual, sorted(set(actual)))


if __name__ == "__main__":
    unittest.main()
