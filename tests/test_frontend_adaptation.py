from pathlib import Path
import tempfile
import unittest

import torch

from dual2pose.frontend_adaptation import (
    common13_mpjpe,
    configure_trainable_scope,
    load_model_weights_only,
)


class TinyFusion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.models = torch.nn.ModuleDict(
            {
                "encoder": torch.nn.Linear(2, 2),
                "gate_head": torch.nn.Linear(2, 1),
                "residual_head": torch.nn.Linear(2, 3),
            }
        )


class FrontEndAdaptationTest(unittest.TestCase):
    """Breaks caught: optimizer state is resumed or heads-only updates the backbone."""

    def test_heads_only_trainable_names_are_gate_and_residual(self) -> None:
        model = TinyFusion()
        report = configure_trainable_scope(model, "heads_only")
        names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(names)
        self.assertTrue(
            all("gate_head" in name or "residual_head" in name for name in names)
        )
        self.assertLess(report["trainable_parameters"], report["total_parameters"])

    def test_full_scope_enables_every_parameter(self) -> None:
        model = TinyFusion()
        report = configure_trainable_scope(model, "full")
        self.assertEqual(report["trainable_parameters"], report["total_parameters"])
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_weights_only_load_ignores_checkpoint_optimizer_state(self) -> None:
        source = TinyFusion()
        for parameter in source.parameters():
            torch.nn.init.constant_(parameter, 3.0)
        target = TinyFusion()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            torch.save(
                {
                    "state_dict": source.state_dict(),
                    "optimizer_states": [{"forbidden": "resume"}],
                    "epoch": 99,
                    "global_step": 1234,
                },
                path,
            )
            report = load_model_weights_only(target, path)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(source.parameters(), target.parameters())))
        self.assertEqual(report.loaded_parameter_tensors, len(source.state_dict()))
        self.assertFalse(hasattr(target, "global_step"))

    def test_common13_metric_excludes_two_eye_joints(self) -> None:
        prediction = torch.zeros((1, 2, 15, 3))
        target = prediction.clone()
        prediction[:, :, :2, 0] = 100.0
        self.assertEqual(float(common13_mpjpe(prediction, target).item()), 0.0)


if __name__ == "__main__":
    unittest.main()
