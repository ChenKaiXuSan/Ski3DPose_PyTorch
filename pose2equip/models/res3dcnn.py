from torchvision.models.video import r3d_18
from torch import nn
import torch


class Res3DCNN(nn.Module):
    """Fallback 3D CNN classifier when project-specific Res3DCNN is unavailable."""

    def __init__(self, hparams):
        super().__init__()
        num_classes = int(getattr(hparams.model, "model_class_num", 2))
        in_channels = int(getattr(hparams.model, "in_channels", 3))
        self.backbone = r3d_18(weights=None)

        if in_channels != 3:
            old_conv = self.backbone.stem[0]
            self.backbone.stem[0] = nn.Conv3d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

        in_feat = int(self.backbone.fc.in_features)
        self.backbone.fc = nn.Linear(in_feat, num_classes)

    def forward(self, video: torch.Tensor, attn_map: torch.Tensor | None = None):
        x = video
        if attn_map is not None:
            if attn_map.shape[1] == 1 and x.shape[1] > 1:
                attn_map = attn_map.expand(-1, x.shape[1], -1, -1, -1)
            x = x * (1.0 + attn_map.clamp(0.0, 1.0))
        return self.backbone(x)
