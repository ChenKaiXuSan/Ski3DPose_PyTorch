import unittest

from dual2pose.eval.summarize_frontend_adaptation import payload_sample_count


class FrontEndAdaptationSampleCountTest(unittest.TestCase):
    def test_top_level_evaluated_sample_count_is_used(self) -> None:
        self.assertEqual(
            payload_sample_count(
                {"sample_count": 64_440, "manifest_metadata": {"sample_count": 1}}
            ),
            64_440,
        )

    def test_missing_evaluated_sample_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_count"):
            payload_sample_count({"manifest_metadata": {"stream_count": 720}})


if __name__ == "__main__":
    unittest.main()
