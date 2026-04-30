from pathlib import Path
from typing import Any, Dict, List

import torch
from pytorch_lightning import (
    LightningModule,
)

from project.models.pose2equip import Pose2EquipNet


# =========================
# Loss
# =========================
def mpjpe(pred, gt):
    return torch.norm(pred - gt, dim=-1).mean()


def length_variance_loss(pred_obj):
    # pred_obj: [B, 8, 3]
    left_ski_len = torch.norm(pred_obj[:, 0] - pred_obj[:, 1], dim=-1)
    right_ski_len = torch.norm(pred_obj[:, 2] - pred_obj[:, 3], dim=-1)
    left_pole_len = torch.norm(pred_obj[:, 4] - pred_obj[:, 5], dim=-1)
    right_pole_len = torch.norm(pred_obj[:, 6] - pred_obj[:, 7], dim=-1)

    loss = 0.0
    loss += left_ski_len.var(unbiased=False)
    loss += right_ski_len.var(unbiased=False)
    loss += left_pole_len.var(unbiased=False)
    loss += right_pole_len.var(unbiased=False)
    return loss


def symmetry_loss(pred_obj):
    # pred_obj: [B, 8, 3]
    left_ski_len = torch.norm(pred_obj[:, 0] - pred_obj[:, 1], dim=-1)
    right_ski_len = torch.norm(pred_obj[:, 2] - pred_obj[:, 3], dim=-1)
    left_pole_len = torch.norm(pred_obj[:, 4] - pred_obj[:, 5], dim=-1)
    right_pole_len = torch.norm(pred_obj[:, 6] - pred_obj[:, 7], dim=-1)

    loss = 0.0
    loss += torch.abs(left_ski_len.mean() - right_ski_len.mean())
    loss += torch.abs(left_pole_len.mean() - right_pole_len.mean())
    return loss


def attachment_loss(pred_obj, human_3d, idx):
    # human_3d: [B, J, 3], pred_obj: [B, 8, 3]
    left_ankle = human_3d[:, idx["left_ankle"]]
    right_ankle = human_3d[:, idx["right_ankle"]]
    left_wrist = human_3d[:, idx["left_wrist"]]
    right_wrist = human_3d[:, idx["right_wrist"]]

    left_ski_center = 0.5 * (pred_obj[:, 0] + pred_obj[:, 1])
    right_ski_center = 0.5 * (pred_obj[:, 2] + pred_obj[:, 3])
    left_pole_grip = pred_obj[:, 4]
    right_pole_grip = pred_obj[:, 6]

    loss = 0.0
    loss += torch.norm(left_ski_center - left_ankle, dim=-1).mean()
    loss += torch.norm(right_ski_center - right_ankle, dim=-1).mean()
    loss += torch.norm(left_pole_grip - left_wrist, dim=-1).mean()
    loss += torch.norm(right_pole_grip - right_wrist, dim=-1).mean()
    return loss


class Pose2EquipTrainer(LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = Pose2EquipNet(
            num_joints=15,
            left_ankle_idx=args.pose2equip.left_ankle_idx,
            right_ankle_idx=args.pose2equip.right_ankle_idx,
            left_wrist_idx=args.pose2equip.left_wrist_idx,
            right_wrist_idx=args.pose2equip.right_wrist_idx,
        )
        self.lr = float(args.loss.lr)
        self.weight_decay = float(getattr(args.pose2equip, "weight_decay", 1e-4))
        self.loss_w_attach = float(getattr(args.pose2equip, "loss_w_attach", 0.1))
        self.loss_w_len = float(getattr(args.pose2equip, "loss_w_len", 0.05))
        self.loss_w_sym = float(
            getattr(args.pose2equip, "loss_w_sym", self.loss_w_len)
        )
        self.loss_w_temp = float(getattr(args.pose2equip, "loss_w_temp", 0.01))

        # GT point reorder for object_3d target (8 points):
        # [left_ski_tip, left_ski_tail, right_ski_tip, right_ski_tail,
        #  left_pole_grip, left_pole_tip, right_pole_grip, right_pole_tip]
        # 这个里面应该和unity记录的GT点顺序一致，或者至少保证能正确选取对应的点进行训练
        self.ski_gt_idx = list(getattr(args.pose2equip, "ski_gt_idx", [1, 2, 4, 5]))
        self.pole_gt_idx = list(getattr(args.pose2equip, "pole_gt_idx", [0, 1, 2, 3]))

        self.idx = {
            "left_ankle": args.pose2equip.left_ankle_idx,
            "right_ankle": args.pose2equip.right_ankle_idx,
            "left_wrist": args.pose2equip.left_wrist_idx,
            "right_wrist": args.pose2equip.right_wrist_idx,
        }
        self.test_outputs: List[Dict[str, Any]] = []
        self.test_save_dir = Path(str(args.log_path)) / "pose_analysis"

    @staticmethod
    def _select_points(x: torch.Tensor, idx: List[int], name: str) -> torch.Tensor:
        # x: B, J, 3
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected {name} shape [B, J, 3], got {tuple(x.shape)}")
        max_idx = x.shape[1] - 1
        if any(i < 0 or i > max_idx for i in idx):
            raise ValueError(
                f"Invalid {name} index in {idx}, valid range is [0, {max_idx}]"
            )
        return x[:, idx, :]

    def _build_object_gt(
        self, pole_gt: torch.Tensor, ski_gt: torch.Tensor
    ) -> torch.Tensor:
        ski_obj = self._select_points(ski_gt, self.ski_gt_idx, "ski_gt")  # B, 4, 3
        pole_obj = self._select_points(pole_gt, self.pole_gt_idx, "pole_gt")  # B, 4, 3

        return torch.cat([ski_obj, pole_obj], dim=1)

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        # 只用cam 1 的结果进行训练
        human_3d = batch["kpt3d_sam"]["character_cam1"].float()  # [B, J, 3]
        human_frame = None
        pole_mask = None
        ski_mask = None
        if isinstance(batch.get("frames"), dict) and "cam1" in batch["frames"]:
            human_frame = batch["frames"]["cam1"].float()
        if isinstance(batch.get("masks"), dict):
            if "ski_pole" in batch["masks"]:
                pole_mask = batch["masks"]["ski_pole"].float()
            if "ski" in batch["masks"]:
                ski_mask = batch["masks"]["ski"].float()

        _gt = batch["kpt3d_gt"]
        pole_gt = _gt["pole"].float()  # [B, 4, 3]
        ski_gt = _gt["ski"].float()  # [B, 6, 3]

        if human_3d.ndim == 4:
            bsz, t_steps = human_3d.shape[:2]
            human_3d = human_3d.reshape(
                bsz * t_steps, human_3d.shape[2], human_3d.shape[3]
            )
            pole_gt = pole_gt.reshape(bsz * t_steps, pole_gt.shape[2], pole_gt.shape[3])
            ski_gt = ski_gt.reshape(bsz * t_steps, ski_gt.shape[2], ski_gt.shape[3])
            if human_frame is not None:
                if human_frame.ndim != 5:
                    raise ValueError(
                        f"Expected human_frame shape [B,C,T,H,W], got {tuple(human_frame.shape)}"
                    )
                human_frame = human_frame.permute(0, 2, 1, 3, 4).reshape(
                    bsz * t_steps,
                    human_frame.shape[1],
                    human_frame.shape[3],
                    human_frame.shape[4],
                )
            if pole_mask is not None:
                if pole_mask.ndim != 5:
                    raise ValueError(
                        f"Expected pole_mask shape [B,1,T,H,W], got {tuple(pole_mask.shape)}"
                    )
                pole_mask = pole_mask.permute(0, 2, 1, 3, 4).reshape(
                    bsz * t_steps,
                    pole_mask.shape[1],
                    pole_mask.shape[3],
                    pole_mask.shape[4],
                )
            if ski_mask is not None:
                if ski_mask.ndim != 5:
                    raise ValueError(
                        f"Expected ski_mask shape [B,1,T,H,W], got {tuple(ski_mask.shape)}"
                    )
                ski_mask = ski_mask.permute(0, 2, 1, 3, 4).reshape(
                    bsz * t_steps,
                    ski_mask.shape[1],
                    ski_mask.shape[3],
                    ski_mask.shape[4],
                )

        object_gt = self._build_object_gt(pole_gt=pole_gt, ski_gt=ski_gt)  # B, 8, 3

        out = self.model(
            human_3d,
            human_frame=human_frame,
            pole_mask=pole_mask,
            ski_mask=ski_mask,
        )
        pred_obj = out["object_3d"]

        l3d = mpjpe(pred_obj, object_gt)
        lcontact = attachment_loss(pred_obj, human_3d, self.idx)
        llength = length_variance_loss(pred_obj)
        lsymmetry = symmetry_loss(pred_obj)
        loss = (
            l3d
            + self.loss_w_attach * lcontact
            + self.loss_w_len * llength
            + self.loss_w_sym * lsymmetry
        )

        batch_size = human_3d.shape[0]
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/mpjpe",
            l3d,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/L3D",
            l3d,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/Lcontact",
            lcontact,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/Llength",
            llength,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/Lsymmetry",
            lsymmetry,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        if stage == "test":
            self.test_outputs.append(
                {
                    "human_3d": human_3d.detach().cpu(),
                    "pred_obj": pred_obj.detach().cpu(),
                    "gt_obj": object_gt.detach().cpu(),
                    "directions": out["directions"].detach().cpu(),
                    "lengths": out["lengths"].detach().cpu(),
                }
            )

        return loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    @torch.no_grad()
    def validation_step(
        self, batch: Dict[str, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def on_test_start(self) -> None:
        self.test_outputs = []
        self.test_save_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def test_step(
        self, batch: Dict[str, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def on_test_epoch_end(self) -> None:
        if len(self.test_outputs) == 0:
            return

        payload = {
            "human_3d": torch.cat([x["human_3d"] for x in self.test_outputs], dim=0),
            "pred_obj": torch.cat([x["pred_obj"] for x in self.test_outputs], dim=0),
            "gt_obj": torch.cat([x["gt_obj"] for x in self.test_outputs], dim=0),
            "directions": torch.cat(
                [x["directions"] for x in self.test_outputs], dim=0
            ),
            "lengths": torch.cat([x["lengths"] for x in self.test_outputs], dim=0),
        }
        save_file = self.test_save_dir / "pose2equip_outputs.pt"
        torch.save(payload, save_file)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        tmax = getattr(self.trainer, "estimated_stepping_batches", None)
        if not isinstance(tmax, int) or tmax <= 0:
            tmax = 1000
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
            },
        }
