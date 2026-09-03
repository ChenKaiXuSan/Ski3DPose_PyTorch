import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "ivc_draft_20260821"
SCRIPTS = PAPER / "scripts"


def read_expanded_manuscript() -> str:
    manuscript = (PAPER / "main.tex").read_text(encoding="utf-8")
    p1_results = (PAPER / "sections" / "p1_results.tex").read_text(
        encoding="utf-8"
    )
    return manuscript.replace(r"\input{sections/p1_results.tex}", p1_results)


class IVCJournalVisualsTest(unittest.TestCase):
    def test_manuscript_uses_journal_title_and_no_conference_figure_files(self) -> None:
        manuscript = (PAPER / "main.tex").read_text(encoding="utf-8")
        self.assertIn(
            "CanonFuse3D: Robustness and Generalization of Calibration-Free "
            "Dual-View 3D Pose Refinement for Skiing Videos",
            manuscript,
        )
        for conference_asset in (
            "figures/original/intro1.png",
            "figures/original/fig2.pdf",
            "figures/original/experiment2/masking_both_view_trends.png",
            "figures/original/experiment3/pro_1_frame0000_real_compare.png",
            "figures/original/experiment3/run_3_frame0000_real_compare.png",
        ):
            self.assertNotIn(conference_asset, manuscript)
        for journal_asset in (
            "figures/journal/study_overview.pdf",
            "figures/journal/method_pipeline.pdf",
            "figures/journal/dataset_examples.pdf",
            "figures/journal/realworld_qualitative.pdf",
        ):
            self.assertIn(journal_asset, manuscript)

    def test_masking_table_uses_held_out_split_wording(self) -> None:
        table = (PAPER / "tables" / "masking_summary.tex").read_text(encoding="utf-8")
        self.assertIn("Complete held-out-split masking robustness", table)
        self.assertNotIn("Complete fold-0 masking robustness", table)

    def test_journal_figures_render_as_vector_pdfs(self) -> None:
        script = SCRIPTS / "generate_journal_figures.py"
        self.assertTrue(script.is_file(), "journal figure generator is missing")
        module = importlib.import_module(
            "paper.ivc_draft_20260821.scripts.generate_journal_figures"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            module.render_study_overview(output / "study_overview.pdf")
            module.render_method_pipeline(output / "method_pipeline.pdf")
            module.render_realworld_qualitative(
                PAPER
                / "figures"
                / "original"
                / "experiment3"
                / "pro_1_frame0000_real_compare.png",
                PAPER
                / "figures"
                / "original"
                / "experiment3"
                / "run_3_frame0000_real_compare.png",
                output / "realworld_qualitative.pdf",
            )
            for name in (
                "study_overview.pdf",
                "method_pipeline.pdf",
                "realworld_qualitative.pdf",
            ):
                artifact = output / name
                self.assertTrue(artifact.is_file())
                self.assertGreater(artifact.stat().st_size, 5_000)
                self.assertEqual(artifact.read_bytes()[:4], b"%PDF")

    def test_dataset_examples_render_as_two_by_three_pdf_and_png(self) -> None:
        module = importlib.import_module(
            "paper.ivc_draft_20260821.scripts.generate_journal_figures"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            image_paths = []
            colors = (
                (30, 80, 150),
                (50, 110, 170),
                (80, 130, 180),
                (100, 150, 190),
            )
            for index, color in enumerate(colors):
                path = output / f"source_{index}.png"
                Image.new("RGB", (160, 90), color).save(path)
                image_paths.append(path)
            author_paths = []
            for index, value in enumerate((120, 180)):
                path = output / f"author_{index}.npz"
                np.savez(
                    path,
                    frame=np.full((90, 160, 3), value, dtype=np.uint8),
                )
                author_paths.append(path)

            pdf = output / "dataset_examples.pdf"
            png = output / "dataset_examples.png"
            module.render_dataset_examples(
                tuple(image_paths[:2]),
                tuple(image_paths[2:]),
                tuple(author_paths),
                pdf,
                png,
            )

            self.assertEqual(pdf.read_bytes()[:4], b"%PDF")
            self.assertGreater(pdf.stat().st_size, 5_000)
            with Image.open(png) as rendered:
                self.assertGreater(rendered.width, rendered.height)
                self.assertGreaterEqual(rendered.width, 2_000)
                pixels = np.asarray(rendered.convert("RGB")).reshape(-1, 3)
                for color in colors:
                    distances = np.abs(
                        pixels.astype(np.int16) - np.asarray(color, dtype=np.int16)
                    ).sum(axis=1)
                    self.assertLess(
                        distances.min(), 10, f"source color {color} was not rendered"
                    )

    def test_unity_foreground_crop_preserves_wide_panel_aspect(self) -> None:
        module = importlib.import_module(
            "paper.ivc_draft_20260821.scripts.generate_journal_figures"
        )
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        frame[65:115, 140:180] = 255
        cropped = module._crop_to_content(frame, target_aspect=16 / 9)
        self.assertLess(cropped.shape[0], frame.shape[0])
        self.assertLess(cropped.shape[1], frame.shape[1])
        self.assertAlmostEqual(
            cropped.shape[1] / cropped.shape[0],
            16 / 9,
            delta=0.05,
        )

    def test_journal_diagram_text_stays_inside_its_boxes(self) -> None:
        module = importlib.import_module(
            "paper.ivc_draft_20260821.scripts.generate_journal_figures"
        )
        self.assertTrue(
            hasattr(module, "build_study_overview_figure"),
            "study-overview layout builder is missing",
        )
        self.assertTrue(
            hasattr(module, "build_method_pipeline_figure"),
            "method-pipeline layout builder is missing",
        )
        for builder in (
            module.build_study_overview_figure,
            module.build_method_pipeline_figure,
        ):
            figure, records = builder()
            try:
                self.assertEqual(module.count_text_overflows(figure, records), 0)
            finally:
                module.plt.close(figure)

    def test_all_extension_generators_use_shared_journal_style(self) -> None:
        style = SCRIPTS / "journal_figure_style.py"
        self.assertTrue(style.is_file(), "shared journal figure style is missing")
        for filename in (
            "generate_extension_artifacts.py",
            "generate_p1_nview_artifacts.py",
            "generate_p1_temporal_alignment_artifacts.py",
            "generate_p1_frontend_adaptation_artifacts.py",
            "generate_journal_figures.py",
        ):
            source = (SCRIPTS / filename).read_text(encoding="utf-8")
            self.assertIn("apply_journal_style", source, filename)
            self.assertIn("JOURNAL_COLORS", source, filename)

    def test_results_follow_scientific_question_order(self) -> None:
        manuscript = read_expanded_manuscript()
        headings = (
            r"\section{Results}",
            r"\subsection{Primary performance and cross-dataset evidence}",
            r"\subsection{Mechanism and component analysis}",
            r"\subsection{Robustness to capture perturbations}",
            r"\subsection{Front-end generalization and adaptation}",
            r"\subsection{Scaling beyond two views}",
            r"\subsection{Automatic synchronization diagnostic}",
            r"\subsection{Qualitative evaluation}",
            r"\section{Discussion}",
        )
        positions = [manuscript.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))
        for obsolete_heading in (
            r"\section{Core Evaluation}",
            r"\section{Generalization and Robustness Analysis}",
            r"\subsection{Baseline masking robustness}",
        ):
            self.assertNotIn(obsolete_heading, manuscript)

    def test_manuscript_is_self_contained_outside_disclosure(self) -> None:
        manuscript = (PAPER / "main.tex").read_text(encoding="utf-8")
        for process_framing in (
            "journal evaluation",
            "Journal-specific recomposition",
            "extend the MMSports masking analysis",
            "conference-stage plot",
            "correct provisional implementation values in the preliminary manuscript",
        ):
            self.assertNotIn(process_framing, manuscript)
        self.assertLessEqual(manuscript.count("MMSports"), 2)
        self.assertLessEqual(manuscript.count("preliminary"), 4)

    def test_native_reference_and_frontend_transfer_have_distinct_tables(self) -> None:
        native = (PAPER / "tables" / "native_reference.tex").read_text(
            encoding="utf-8"
        )
        transfer = (PAPER / "tables" / "frontend_generalization.tex").read_text(
            encoding="utf-8"
        )
        self.assertIn(r"\label{tab:native_reference}", native)
        self.assertIn("native SAM3D", native)
        self.assertIn("0.2792", native)
        self.assertIn("0.1551", native)
        self.assertNotIn("SAM3D (native)", transfer)
        for front_end in ("MotionBERT", "PoseFormer", "VideoPose3D"):
            self.assertIn(front_end, transfer)

    def test_result_floats_cannot_cross_into_discussion(self) -> None:
        manuscript = (PAPER / "main.tex").read_text(encoding="utf-8")
        self.assertIn(r"\usepackage[section]{placeins}", manuscript)
        discussion = manuscript.index(r"\section{Discussion}")
        prefix = manuscript[:discussion]
        self.assertTrue(prefix.rstrip().endswith(r"\FloatBarrier"))

    def test_qualitative_figure_is_anchored_after_its_heading(self) -> None:
        manuscript = (PAPER / "main.tex").read_text(encoding="utf-8")
        self.assertIn(r"\usepackage{float}", manuscript)
        start = manuscript.index(r"\subsection{Qualitative evaluation}")
        end = manuscript.index(r"\section{Discussion}")
        qualitative = manuscript[start:end]
        self.assertIn(r"\begin{figure}[H]", qualitative)


if __name__ == "__main__":
    unittest.main()
