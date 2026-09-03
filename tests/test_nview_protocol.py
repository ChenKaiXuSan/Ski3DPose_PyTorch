import unittest

from dual2pose.eval.nview_protocol import (
    CameraGroup,
    build_nested_camera_groups,
    nested_cameras,
)


def synthetic_fold_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    actions = [
        ("female", "action_f0"),
        ("female", "action_f1"),
        ("male", "action_m0"),
        ("male", "action_m1"),
    ]
    for person_id, action_id in actions:
        for layer in range(5):
            for azimuth in range(0, 360, 10):
                rows.append(
                    {
                        "person_id": person_id,
                        "action_id": action_id,
                        "cam1_id": f"capture_L{layer}_A{azimuth:03d}",
                        "cam2_id": f"capture_L{layer}_A{(azimuth + 10) % 360:03d}",
                    }
                )
    return rows


class NViewGroupTest(unittest.TestCase):
    """Breaks caught: cyclic camera groups duplicate rotations or change nesting."""

    def test_build_nested_camera_groups_has_180_unique_groups(self) -> None:
        groups = build_nested_camera_groups(synthetic_fold_rows())
        self.assertEqual(len(groups), 180)
        self.assertEqual(len({group.group_id for group in groups}), 180)

    def test_nested_camera_order_is_declared(self) -> None:
        group = CameraGroup(
            "g",
            "male",
            "action",
            0,
            (
                "capture_L0_A000",
                "capture_L0_A090",
                "capture_L0_A180",
                "capture_L0_A270",
            ),
        )
        self.assertEqual(nested_cameras(group, 1), ("capture_L0_A000",))
        self.assertEqual(
            nested_cameras(group, 2),
            ("capture_L0_A000", "capture_L0_A180"),
        )
        self.assertEqual(
            nested_cameras(group, 3),
            ("capture_L0_A000", "capture_L0_A090", "capture_L0_A180"),
        )
        self.assertEqual(nested_cameras(group, 4), group.cameras)

    def test_invalid_view_count_is_rejected(self) -> None:
        group = build_nested_camera_groups(synthetic_fold_rows())[0]
        with self.assertRaisesRegex(ValueError, "1, 2, 3, 4"):
            nested_cameras(group, 0)

    def test_missing_quadrant_camera_is_rejected(self) -> None:
        rows = synthetic_fold_rows()
        rows = [
            row
            for row in rows
            if not (
                row["action_id"] == "action_f0"
                and (row["cam1_id"] == "capture_L0_A090" or row["cam2_id"] == "capture_L0_A090")
            )
        ]
        with self.assertRaisesRegex(ValueError, "incomplete 90-degree camera ring"):
            build_nested_camera_groups(rows)


if __name__ == "__main__":
    unittest.main()
