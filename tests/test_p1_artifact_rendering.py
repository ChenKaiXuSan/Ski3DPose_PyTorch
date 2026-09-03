from pathlib import Path
import tempfile
import unittest

from paper.ivc_draft_20260821.scripts.generate_p1_frontend_adaptation_artifacts import (
    render_figure as render_frontend_figure,
    validate_matrix_rows,
)
from paper.ivc_draft_20260821.scripts.generate_p1_nview_artifacts import (
    render_figure as render_nview_figure,
    validate_summary_rows as validate_nview_rows,
)
from paper.ivc_draft_20260821.scripts.generate_p1_temporal_alignment_artifacts import (
    render_figure as render_temporal_figure,
    validate_summary_rows as validate_temporal_rows,
)
from tests import test_p1_frontend_adaptation_artifacts as frontend_fixtures
from tests import test_p1_nview_artifacts as nview_fixtures
from tests import test_p1_temporal_alignment_artifacts as temporal_fixtures


class P1ArtifactRenderingTest(unittest.TestCase):
    def test_all_three_pdf_figures_render_from_validated_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = (
                root / "nview.pdf",
                root / "temporal.pdf",
                root / "frontend.pdf",
            )
            nview_rows = nview_fixtures.P1NViewArtifactsTest()._rows()
            for row in nview_rows:
                mean = float(row["mpjpe_mean"])
                row["mpjpe_ci95_low"] = str(mean - 0.01)
                row["mpjpe_ci95_high"] = str(mean + 0.01)
            render_nview_figure(validate_nview_rows(nview_rows), outputs[0])
            render_temporal_figure(
                validate_temporal_rows(temporal_fixtures.P1TemporalAlignmentArtifactsTest()._rows()),
                outputs[1],
            )
            render_frontend_figure(
                validate_matrix_rows(frontend_fixtures.P1FrontendAdaptationArtifactsTest()._rows()),
                outputs[2],
            )
            for output in outputs:
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 1_000)
                self.assertEqual(output.read_bytes()[:4], b"%PDF")


if __name__ == "__main__":
    unittest.main()
