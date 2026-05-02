from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging

import numpy as np
import torch
from pytorch_lightning import (
    LightningModule,
)
from scipy.spatial.transform import Rotation

from project.models.pose2equip_net import Pose2EquipNet
from project.map_config import TARGET_SKELETON_CONNECTIONS_INDEX

logger = logging.getLogger(__name__)


# =========================
# Loss
# =========================
def mpjpe(pred, gt):
    """Mean per-point Euclidean distance on object keypoints."""
    return torch.norm(pred - gt, dim=-1).mean()


def length_variance_loss(pred_obj):
    """Penalize sample-wise length jitter in a batch.

    We compute 4 equipment lengths (left/right ski, left/right pole) and
    minimize their per-batch variance for geometric stability.
    """
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
    """Encourage left/right symmetry by matching mean lengths."""
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
    """Keep equipment anchors close to body anchors.

    Constraints:
    - ski center -> ankle
    - pole grip  -> wrist
    """
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


def _bbox_diag_from_mask(mask_2d: torch.Tensor) -> torch.Tensor:
    """Return bbox diagonal length for one binary mask tensor [H,W]."""
    ys, xs = torch.where(mask_2d > 0.5)
    if ys.numel() == 0:
        return mask_2d.new_tensor(0.0)
    h = (ys.max() - ys.min() + 1).float()
    w = (xs.max() - xs.min() + 1).float()
    return torch.sqrt(h * h + w * w)


def mask_length_ratio_loss(
    pred_obj: torch.Tensor,
    ski_mask: torch.Tensor | None,
    ski_pole_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Constrain 3D ski/pole length ratio with 2D mask extent ratio.

    We only use ratio (not absolute scale) to avoid camera-scale ambiguity.
    """
    if ski_mask is None or ski_pole_mask is None:
        return pred_obj.new_tensor(0.0)
    if ski_mask.ndim != 4 or ski_pole_mask.ndim != 4:
        return pred_obj.new_tensor(0.0)

    # 3D predicted lengths [B]
    pred_ski_len = 0.5 * (
        torch.norm(pred_obj[:, 0] - pred_obj[:, 1], dim=-1)
        + torch.norm(pred_obj[:, 2] - pred_obj[:, 3], dim=-1)
    )
    pred_pole_len = 0.5 * (
        torch.norm(pred_obj[:, 4] - pred_obj[:, 5], dim=-1)
        + torch.norm(pred_obj[:, 6] - pred_obj[:, 7], dim=-1)
    )

    bsz = pred_obj.shape[0]
    losses = []
    eps = 1e-6
    for b in range(bsz):
        ski_m = ski_mask[b, 0]
        ski_pole_m = ski_pole_mask[b, 0]
        pole_only_m = torch.clamp(ski_pole_m - ski_m, min=0.0)

        ski_extent = _bbox_diag_from_mask(ski_m)
        pole_extent = _bbox_diag_from_mask(pole_only_m)
        if ski_extent <= 0 or pole_extent <= 0:
            continue

        pred_ratio = pred_ski_len[b] / (pred_pole_len[b] + eps)
        mask_ratio = ski_extent / (pole_extent + eps)
        losses.append(
            torch.abs(torch.log(pred_ratio + eps) - torch.log(mask_ratio + eps))
        )

    if len(losses) == 0:
        return pred_obj.new_tensor(0.0)
    return torch.stack(losses).mean()


def equipment_segment_lengths(obj: torch.Tensor) -> torch.Tensor:
    """Return 4 segment lengths from equipment keypoints.

    Output order: [left_ski, right_ski, left_pole, right_pole], shape [B, 4].
    """
    if obj.ndim != 3 or obj.shape[1] != 8 or obj.shape[2] != 3:
        raise ValueError(f"Expected object shape [B,8,3], got {tuple(obj.shape)}")

    return torch.stack(
        [
            torch.norm(obj[:, 0] - obj[:, 1], dim=-1),
            torch.norm(obj[:, 2] - obj[:, 3], dim=-1),
            torch.norm(obj[:, 4] - obj[:, 5], dim=-1),
            torch.norm(obj[:, 6] - obj[:, 7], dim=-1),
        ],
        dim=-1,
    )


def absolute_length_loss(pred_obj: torch.Tensor, gt_obj: torch.Tensor) -> torch.Tensor:
    """Supervise absolute equipment lengths with SmoothL1.

    Segment order:
      0: left_ski  (0,1)
      1: right_ski (2,3)
      2: left_pole (4,5)
      3: right_pole(6,7)
    """
    pred_len = equipment_segment_lengths(pred_obj)
    gt_len = equipment_segment_lengths(gt_obj)
    return torch.nn.functional.smooth_l1_loss(pred_len, gt_len)


def compute_procrustes_alignment(
    pred: np.ndarray, gt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Procrustes alignment: find rotation/scale/translation to align pred to gt.

    Args:
        pred: (N, 3) predicted points
        gt: (N, 3) ground truth points

    Returns:
        pred_aligned: (N, 3) aligned prediction
        scale: scalar alignment scale
        alignment_error: alignment RMSE
    """
    # Center both
    pred_mean = pred.mean(axis=0, keepdims=True)
    gt_mean = gt.mean(axis=0, keepdims=True)
    pred_c = pred - pred_mean
    gt_c = gt - gt_mean

    # SVD for rotation
    H = pred_c.T @ gt_c
    U, _, VT = np.linalg.svd(H)
    R = (U @ VT).astype(np.float32)

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        VT[-1] *= -1
        R = (U @ VT).astype(np.float32)

    # Scale
    scale = np.linalg.norm(gt_c) / (np.linalg.norm(pred_c) + 1e-8)

    # Align
    pred_aligned = (scale * pred_c @ R.T) + gt_mean
    alignment_error = np.linalg.norm(pred_aligned - gt, axis=1).mean()

    return pred_aligned, scale, alignment_error


def evaluate_pose_metrics(pred_obj: np.ndarray, gt_obj: np.ndarray) -> Dict[str, float]:
    """Evaluate 3D equipment keypoint prediction metrics.

    Args:
        pred_obj: (N, 8, 3) predicted object keypoints
        gt_obj: (N, 8, 3) ground truth object keypoints

    Returns:
        Dict with keys: mpjpe, pa_mpjpe, mpjpe_left_ski, mpjpe_right_ski, mpjpe_left_pole, mpjpe_right_pole
    """
    # MPJPE: direct Euclidean distance
    mpjpe = np.linalg.norm(pred_obj - gt_obj, axis=2).mean()  # (N, 8) -> scalar

    # PA-MPJPE: Procrustes-aligned
    pred_flat = pred_obj.reshape(-1, 3)  # (N*8, 3)
    gt_flat = gt_obj.reshape(-1, 3)
    pred_aligned, _, _ = compute_procrustes_alignment(pred_flat, gt_flat)
    pa_mpjpe = np.linalg.norm(pred_aligned - gt_flat, axis=1).mean()

    # Per-object metrics
    # Points mapping: 0,1=left_ski, 2,3=right_ski, 4,5=left_pole, 6,7=right_pole
    left_ski_err = np.linalg.norm(pred_obj[:, :2] - gt_obj[:, :2], axis=2).mean()
    right_ski_err = np.linalg.norm(pred_obj[:, 2:4] - gt_obj[:, 2:4], axis=2).mean()
    left_pole_err = np.linalg.norm(pred_obj[:, 4:6] - gt_obj[:, 4:6], axis=2).mean()
    right_pole_err = np.linalg.norm(pred_obj[:, 6:8] - gt_obj[:, 6:8], axis=2).mean()

    return {
        "mpjpe": float(mpjpe),
        "pa_mpjpe": float(pa_mpjpe),
        "mpjpe_left_ski": float(left_ski_err),
        "mpjpe_right_ski": float(right_ski_err),
        "mpjpe_left_pole": float(left_pole_err),
        "mpjpe_right_pole": float(right_pole_err),
    }


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
            target_skeleton_connections_idx=TARGET_SKELETON_CONNECTIONS_INDEX,
        )
        self.lr = float(getattr(args.loss, "lr", 0.1))
        self.weight_decay = float(getattr(args.loss, "weight_decay", 0.01))

        self.loss_w_attach = float(getattr(args.pose2equip, "loss_w_attach", 0.1))
        self.loss_w_len = float(getattr(args.pose2equip, "loss_w_len", 0.05))
        self.loss_w_sym = float(getattr(args.pose2equip, "loss_w_sym", self.loss_w_len))
        self.loss_w_mask_len = float(getattr(args.pose2equip, "loss_w_mask_len", 0.05))
        self.loss_w_len_abs = float(getattr(args.pose2equip, "loss_w_len_abs", 0.2))
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
        human_3d = batch["kpt3d_sam"]["character_cam1"].float()  # [B, t, J, 3]
        human_frame = None
        pole_mask = None
        ski_mask = None
        if isinstance(batch.get("frames"), dict) and "cam1" in batch["frames"]:
            human_frame = batch["frames"]["cam1"].float()  # b, c, t, h, w
        if isinstance(batch.get("masks"), dict):
            if "ski_pole" in batch["masks"]:
                pole_mask = batch["masks"]["ski_pole"].float()  # b, 1, t, h, w
            if "ski" in batch["masks"]:
                ski_mask = batch["masks"]["ski"].float()  # b, 1, t, h, w

        # GT fron Unity
        _gt = batch["kpt3d_gt"]
        pole_gt = _gt["pole"].float()  # [B, t, 4, 3]
        ski_gt = _gt["ski"].float()  # [B, t, 6, 3]

        if human_3d.ndim == 4:  # B, T, J, 3 -> merge B,T for frame-wise processing
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
        lmask_len = mask_length_ratio_loss(
            pred_obj=pred_obj,
            ski_mask=ski_mask,
            ski_pole_mask=pole_mask,
        )
        l_len_abs = absolute_length_loss(pred_obj=pred_obj, gt_obj=object_gt)
        pred_len = equipment_segment_lengths(pred_obj)
        gt_len = equipment_segment_lengths(object_gt)

        pred_ski_len_mean = 0.5 * (pred_len[:, 0].mean() + pred_len[:, 1].mean())
        pred_pole_len_mean = 0.5 * (pred_len[:, 2].mean() + pred_len[:, 3].mean())
        gt_ski_len_mean = 0.5 * (gt_len[:, 0].mean() + gt_len[:, 1].mean())
        gt_pole_len_mean = 0.5 * (gt_len[:, 2].mean() + gt_len[:, 3].mean())

        # Final objective:
        #   L = L3D + w_attach * Lcontact + w_len * Llength + w_sym * Lsymmetry
        #       + w_mask_len * LmaskLen + w_len_abs * LlenAbs
        # L3D is the main supervision term; others are geometric regularizers.
        loss = (
            l3d
            + self.loss_w_attach * lcontact
            + self.loss_w_len * llength
            + self.loss_w_sym * lsymmetry
            + self.loss_w_mask_len * lmask_len
            + self.loss_w_len_abs * l_len_abs
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
        self.log(
            f"{stage}/Lmask_len",
            lmask_len,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/Llen_abs",
            l_len_abs,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/len_mean",
            out["lengths"].mean(),
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/len_std",
            out["lengths"].std(unbiased=False),
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/pred_ski_len_mean",
            pred_ski_len_mean,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/pred_pole_len_mean",
            pred_pole_len_mean,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/gt_ski_len_mean",
            gt_ski_len_mean,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/gt_pole_len_mean",
            gt_pole_len_mean,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        if stage == "train" and float(lcontact.detach().item()) < 1e-6:
            logger.debug(
                "Lcontact is near zero; current geometry anchors ski-center/pole-grip to ankle/wrist by design."
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

        # Compute performance metrics
        pred_obj_np = payload["pred_obj"].numpy()
        gt_obj_np = payload["gt_obj"].numpy()
        metrics = evaluate_pose_metrics(pred_obj_np, gt_obj_np)

        # Save metrics to txt
        metrics_file = self.test_save_dir / "evaluation_metrics.txt"
        with open(metrics_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("Equipment 3D Keypoint Prediction - Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total samples evaluated: {pred_obj_np.shape[0]}\n\n")

            f.write("Global Metrics:\n")
            f.write("-" * 60 + "\n")
            f.write(
                f"  MPJPE (Mean Per Joint Position Error):  {metrics['mpjpe']:.4f} mm\n"
            )
            f.write(
                f"  PA-MPJPE (Procrustes Aligned):         {metrics['pa_mpjpe']:.4f} mm\n\n"
            )

            f.write("Per-Object Metrics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Left Ski MPJPE:   {metrics['mpjpe_left_ski']:.4f} mm\n")
            f.write(f"  Right Ski MPJPE:  {metrics['mpjpe_right_ski']:.4f} mm\n")
            f.write(f"  Left Pole MPJPE:  {metrics['mpjpe_left_pole']:.4f} mm\n")
            f.write(f"  Right Pole MPJPE: {metrics['mpjpe_right_pole']:.4f} mm\n\n")

            ski_avg = (metrics["mpjpe_left_ski"] + metrics["mpjpe_right_ski"]) / 2.0
            pole_avg = (metrics["mpjpe_left_pole"] + metrics["mpjpe_right_pole"]) / 2.0
            f.write(f"  Avg Ski Error:    {ski_avg:.4f} mm\n")
            f.write(f"  Avg Pole Error:   {pole_avg:.4f} mm\n\n")

            f.write("=" * 60 + "\n")

        logger.info(f"Evaluation metrics saved to {metrics_file}")
        logger.info(
            f"MPJPE: {metrics['mpjpe']:.4f} mm, PA-MPJPE: {metrics['pa_mpjpe']:.4f} mm"
        )

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
