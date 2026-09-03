import unittest

from dual2pose.experiments.run_frontend_adaptation_matrix import (
    AdaptationRun,
    EvaluationCell,
    build_evaluation_cells,
    build_training_runs,
)


class FrontEndAdaptationMatrixTest(unittest.TestCase):
    """Breaks caught: an adaptation condition or cross-front-end cell is omitted."""

    def test_seven_training_runs_are_declared(self) -> None:
        self.assertEqual(
            set(build_training_runs()),
            {
                AdaptationRun("videopose3d", "heads_only"),
                AdaptationRun("poseformer", "heads_only"),
                AdaptationRun("motionbert", "heads_only"),
                AdaptationRun("videopose3d", "full"),
                AdaptationRun("poseformer", "full"),
                AdaptationRun("motionbert", "full"),
                AdaptationRun("mixed", "full"),
            },
        )

    def test_evaluation_matrix_is_eight_models_by_four_frontends(self) -> None:
        cells = build_evaluation_cells()
        self.assertEqual(len(cells), 32)
        self.assertEqual(len(set(cells)), 32)
        self.assertEqual(
            {cell.test_frontend for cell in cells},
            {"sam3d", "videopose3d", "poseformer", "motionbert"},
        )
        self.assertIn(EvaluationCell("mmsports", "sam3d"), cells)
        self.assertIn(EvaluationCell("mixed_full", "motionbert"), cells)


if __name__ == "__main__":
    unittest.main()
