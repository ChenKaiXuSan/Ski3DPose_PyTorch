import unittest

from paper.ivc_draft_20260821.scripts.generate_p1_nview_artifacts import (
    validate_summary_rows,
)
from tests import test_p1_nview_artifacts as nview_fixtures


class P1NViewCIValidationTest(unittest.TestCase):
    def test_confidence_interval_must_contain_mean(self) -> None:
        rows = nview_fixtures.P1NViewArtifactsTest()._rows()
        rows[0]["mpjpe_ci95_low"] = "0.4"
        with self.assertRaisesRegex(ValueError, "confidence interval"):
            validate_summary_rows(rows)


if __name__ == "__main__":
    unittest.main()
