import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule

from ..models.res3dcnn import Res3DCNN

logger = logging.getLogger(__name__)


class Res3DCNNTrainer(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.img_size = hparams.data.img_size
        self.lr = getattr(hparams.loss, "lr", 1e-3)  # default lr

        self.num_classes = hparams.model.model_class_num

        # define model
        self.model = Res3DCNN(hparams)

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        self.save_root = getattr(hparams.train, "log_path", "./logs")

        # Ablation switches for keypoint-as-image experiments.
        self.use_kpt_as_image = bool(getattr(hparams.model, "use_kpt_as_image", False))
        self.kpt_alpha = float(getattr(hparams.model, "kpt_alpha", 1.0))
        self.kpt_sigma = float(getattr(hparams.model, "kpt_sigma", 2.0))
        self.kpt_fusion_mode = str(
            getattr(hparams.model, "kpt_fusion_mode", "max")
        ).lower()
        self.kpt_source = str(getattr(hparams.model, "kpt_source", "kpt2d_sam")).lower()

        if self.kpt_fusion_mode not in {"max", "add", "replace"}:
            raise ValueError(
                f"Unsupported kpt_fusion_mode={self.kpt_fusion_mode}, choose from [max, add, replace]."
            )

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _pick_first_tensor_from_dict(d: Any) -> torch.Tensor | None:
        if not isinstance(d, dict) or len(d) == 0:
            return None
        for v in d.values():
            if isinstance(v, torch.Tensor):
                return v
        return None

    def _extract_kpt2d(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Extract 2D keypoints from batch in a robust way.

        Supported shapes:
          - [B, T, J, 2/3]
          - [T, J, 2/3] (single sample)
          - [B, J, 2/3]
          - [J, 2/3]
        """
        direct_keys = [
            "kpt2d",
            "kpt2d_sam",
            "kpt2d_gt",
            "keypoints_2d",
            "pose2d",
        ]

        x = None
        for k in direct_keys:
            if k in batch and isinstance(batch[k], torch.Tensor):
                x = batch[k]
                break

        # Nested dict format from current single-view dataset.
        if x is None:
            if self.kpt_source == "kpt2d_gt":
                x = self._pick_first_tensor_from_dict(batch.get("kpt2d_gt"))
            else:
                x = self._pick_first_tensor_from_dict(batch.get("kpt2d_sam"))
                if x is None:
                    x = self._pick_first_tensor_from_dict(batch.get("kpt2d_gt"))

        if x is None:
            return None

        if x.ndim == 4:
            # [B,T,J,C]
            return x
        if x.ndim == 3:
            # [T,J,C] or [B,J,C]
            if x.shape[-1] < 2:
                return None
            if x.shape[0] == batch["video"].shape[0]:
                return x.unsqueeze(1)  # [B,1,J,C]
            return x.unsqueeze(0)  # [1,T,J,C]
        if x.ndim == 2:
            # [J,C]
            if x.shape[-1] < 2:
                return None
            return x.unsqueeze(0).unsqueeze(0)  # [1,1,J,C]
        return None

    def _build_kpt_heatmap(
        self,
        kpt2d: torch.Tensor,
        *,
        batch_size: int,
        time_steps: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert keypoints [B,T,J,2/3] to heatmap [B,1,T,H,W]."""
        if kpt2d.shape[0] != batch_size:
            if kpt2d.shape[0] == 1:
                kpt2d = kpt2d.expand(batch_size, *kpt2d.shape[1:])
            else:
                raise ValueError(
                    f"kpt2d batch size mismatch: {kpt2d.shape[0]} vs video batch {batch_size}"
                )

        if kpt2d.shape[1] != time_steps:
            # nearest temporal index selection for robustness
            src_t = kpt2d.shape[1]
            idx = (
                torch.linspace(0, src_t - 1, steps=time_steps, device=kpt2d.device)
                .round()
                .long()
            )
            kpt2d = kpt2d.index_select(1, idx)

        kpt = kpt2d[..., :2].to(device=device, dtype=dtype)
        # normalize [0,1] -> pixel if needed
        if torch.isfinite(kpt).any():
            max_abs = torch.nan_to_num(
                kpt.abs(), nan=0.0, posinf=0.0, neginf=0.0
            ).amax()
            if max_abs <= 2.0:
                kpt_x = kpt[..., 0] * (width - 1)
                kpt_y = kpt[..., 1] * (height - 1)
            else:
                kpt_x = kpt[..., 0]
                kpt_y = kpt[..., 1]
        else:
            kpt_x = kpt[..., 0]
            kpt_y = kpt[..., 1]

        xs = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, 1, width)
        ys = torch.arange(height, device=device, dtype=dtype).view(1, 1, 1, height, 1)

        x0 = kpt_x.unsqueeze(-1).unsqueeze(-1)
        y0 = kpt_y.unsqueeze(-1).unsqueeze(-1)
        sigma2 = max(self.kpt_sigma * self.kpt_sigma, 1e-6)

        gauss = torch.exp(
            -((xs - x0) ** 2 + (ys - y0) ** 2) / (2.0 * sigma2)
        )  # [B,T,J,H,W]
        heat = gauss.max(dim=2).values  # [B,T,H,W]
        heat = heat.unsqueeze(1)  # [B,1,T,H,W]
        return heat

    def _prepare_attn_input(
        self,
        *,
        batch: dict[str, torch.Tensor],
        video: torch.Tensor,
        attn_map: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_kpt_as_image:
            return attn_map

        kpt2d = self._extract_kpt2d(batch)
        if kpt2d is None:
            return attn_map

        b, _, t, h, w = video.shape
        kpt_heat = self._build_kpt_heatmap(
            kpt2d,
            batch_size=b,
            time_steps=t,
            height=h,
            width=w,
            device=video.device,
            dtype=video.dtype,
        )

        if attn_map.shape[1] != kpt_heat.shape[1]:
            if attn_map.shape[1] == 1:
                pass
            else:
                kpt_heat = kpt_heat.expand(-1, attn_map.shape[1], -1, -1, -1)

        if self.kpt_fusion_mode == "replace":
            return kpt_heat
        if self.kpt_fusion_mode == "add":
            return (attn_map + self.kpt_alpha * kpt_heat).clamp(0.0, 1.0)
        # default: max
        return torch.maximum(attn_map, self.kpt_alpha * kpt_heat)

    def _forward_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video = batch["video"].detach()  # b, c, t, h, w
        attn_map = batch["attn_map"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float()  # b

        attn_input = self._prepare_attn_input(
            batch=batch, video=video, attn_map=attn_map
        )
        video_preds = self.model(video, attn_input)
        video_preds_softmax = torch.softmax(video_preds, dim=1)
        return video_preds, video_preds_softmax, label

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        video = batch["video"].detach()  # b, c, t, h, w
        b, _, _, _, _ = video.shape

        video_preds, video_preds_softmax, label = self._forward_batch(batch)

        assert label.shape[0] == video_preds.shape[0]

        loss = F.cross_entropy(video_preds, label.long())

        self.log("train/loss", loss, on_epoch=True, on_step=True, batch_size=b)

        # log metrics
        video_acc = self._accuracy(video_preds_softmax, label)
        video_precision = self._precision(video_preds_softmax, label)
        video_recall = self._recall(video_preds_softmax, label)
        video_f1_score = self._f1_score(video_preds_softmax, label)
        video_confusion_matrix = self._confusion_matrix(video_preds_softmax, label)

        self.log_dict(
            {
                "train/video_acc": video_acc,
                "train/video_precision": video_precision,
                "train/video_recall": video_recall,
                "train/video_f1_score": video_f1_score,
            },
            on_epoch=True,
            on_step=True,
            batch_size=b,
        )
        logger.info(f"train loss: {loss.item()}")

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        video = batch["video"].detach()  # b, c, t, h, w
        b, _, _, _, _ = video.shape

        video_preds, video_preds_softmax, label = self._forward_batch(batch)

        loss = F.cross_entropy(video_preds, label.long())

        self.log("val/loss", loss, on_epoch=True, on_step=True, batch_size=b)

        # log metrics
        video_acc = self._accuracy(video_preds_softmax, label)
        video_precision = self._precision(video_preds_softmax, label)
        video_recall = self._recall(video_preds_softmax, label)
        video_f1_score = self._f1_score(video_preds_softmax, label)
        video_confusion_matrix = self._confusion_matrix(video_preds_softmax, label)

        self.log_dict(
            {
                "val/video_acc": video_acc,
                "val/video_precision": video_precision,
                "val/video_recall": video_recall,
                "val/video_f1_score": video_f1_score,
            },
            on_epoch=True,
            on_step=True,
            batch_size=b,
        )

        logger.info(f"val loss: {loss.item()}")

    ##############
    # test step
    ##############
    # the order of the hook function is:
    # on_test_start -> test_step -> on_test_batch_end -> on_test_epoch_end -> on_test_end

    def on_test_start(self) -> None:
        """hook function for test start"""

        self.test_pred_list: list[torch.Tensor] = []
        self.test_label_list: list[torch.Tensor] = []

        logger.info("test start")

    def on_test_end(self) -> None:
        """hook function for test end"""
        logger.info("test end")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        video = batch["video"].detach()  # b, c, t, h, w
        attn_map = batch["attn_map"].detach()  # b, c, t, h, w

        b, _, _, _, _ = video.shape

        video_preds, video_preds_softmax, label = self._forward_batch(batch)

        loss = F.cross_entropy(video_preds, label.long())

        self.log("test/loss", loss, on_epoch=True, on_step=True, batch_size=b)

        # log metrics
        video_acc = self._accuracy(video_preds_softmax, label)
        video_precision = self._precision(video_preds_softmax, label)
        video_recall = self._recall(video_preds_softmax, label)
        video_f1_score = self._f1_score(video_preds_softmax, label)
        video_confusion_matrix = self._confusion_matrix(video_preds_softmax, label)

        metric_dict = {
            "test/video_acc": video_acc,
            "test/video_precision": video_precision,
            "test/video_recall": video_recall,
            "test/video_f1_score": video_f1_score,
        }
        self.log_dict(metric_dict, on_epoch=True, on_step=True, batch_size=b)

        self.test_pred_list.append(video_preds_softmax.detach().cpu())
        self.test_label_list.append(label.detach().cpu())

        fold = (
            getattr(self.logger, "root_dir", "fold").split("/")[-1]
            if self.logger
            else "fold"
        )
        if batch_idx < 10:
            dump_all_feature_maps(
                model=self.model,
                video=video,
                video_info=batch.get("info", None),
                attn_map=attn_map,
                save_root=f"{self.save_root}/test_all_feature_maps/{fold}/batch_{batch_idx}",
                include_types=(torch.nn.Conv3d, torch.nn.Linear),
                include_name_contains=["conv_c"],
                resize_to=(256, 256),  # 指定输出大小
                resize_mode="bilinear",  # 放大更平滑
            )

        return video_preds_softmax, video_preds

    def on_test_epoch_end(self) -> None:
        """hook function for test epoch end"""

        # save the metrics to file
        fold = (
            getattr(self.logger, "root_dir", "fold").split("/")[-1]
            if self.logger
            else "fold"
        )
        save_helper(
            all_pred=self.test_pred_list,
            all_label=self.test_label_list,
            fold=fold,
            save_path=self.save_root,
            num_class=self.num_classes,
        )

        logger.info("test epoch end")

    def configure_optimizers(self):
        """
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.trainer.estimated_stepping_batches,
                    # verbose=True,
                ),
                "monitor": "train/loss",
            },
        }
