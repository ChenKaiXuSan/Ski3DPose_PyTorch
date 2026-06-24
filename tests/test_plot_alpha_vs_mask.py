import unittest

import numpy as np

from dual2pose.eval import plot_alpha_vs_mask as plotter


class PlotAlphaVsMaskTests(unittest.TestCase):
    def test_parse_dirname_reads_decimal_ratio(self):
        self.assertEqual(plotter._parse_dirname("both_random_r0p10")["ratio"], 0.10)
        self.assertEqual(plotter._parse_dirname("both_random_r0p05")["ratio"], 0.05)
        self.assertEqual(plotter._parse_dirname("both_random_r1p00")["ratio"], 1.00)

    def test_paper_trends_prepend_shared_no_mask_baseline(self):
        agg_data = {
            ("none", "random"): (
                np.array([0.10]),
                np.array([0.50]),
                np.array([0.18]),
            ),
            ("none", "distal"): (
                np.array([0.10]),
                np.array([0.50]),
                np.array([0.18]),
            ),
            ("both", "random"): (
                np.array([0.10, 0.20]),
                np.array([0.51, 0.52]),
                np.array([0.26, 0.32]),
            ),
            ("both", "distal"): (
                np.array([0.10, 0.20]),
                np.array([0.49, 0.48]),
                np.array([0.23, 0.29]),
            ),
        }

        trends = plotter._build_both_view_paper_trends(agg_data)

        self.assertEqual(trends["random"]["ratio"], [0.0, 0.1, 0.2])
        self.assertEqual(trends["distal"]["ratio"], [0.0, 0.1, 0.2])
        self.assertEqual(trends["random"]["mpjpe"][0], 0.18)
        self.assertEqual(trends["distal"]["mpjpe"][0], 0.18)
        self.assertNotEqual(trends["random"]["mpjpe"][0], 0.26)


if __name__ == "__main__":
    unittest.main()
