import unittest

from dual2pose.eval.eval_unity_nview import collect_accepted_samples
from dual2pose.eval.nview_protocol import InsufficientCommonFrames


class NViewAcceptanceLimitTest(unittest.TestCase):
    """Break caught: smoke limit counts a rejected proposal as its only group."""

    def test_limit_counts_accepted_groups_not_proposals(self) -> None:
        def loader(group, lookup, target_t):
            del lookup, target_t
            if group == "reject":
                raise InsufficientCommonFrames("reject", 20, 30)
            return f"sample-{group}"

        accepted, rejected, evaluated = collect_accepted_samples(
            ["reject", "accept", "unused"],
            row_lookup={},
            target_t=30,
            limit_accepted=1,
            sample_loader=loader,
        )
        self.assertEqual(accepted, [("accept", "sample-accept")])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(evaluated, 2)


if __name__ == "__main__":
    unittest.main()
