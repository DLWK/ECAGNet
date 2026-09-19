"""PyTorch Lightning module for ECAGNet training and evaluation."""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as functional
from monai.losses import DiceCELoss
from torch import nn
from torchmetrics.classification import BinaryJaccardIndex

from .model import ECAFUNet_V2


class GaussianBlur(nn.Module):
    """Fixed Gaussian filter used to create the soft GPS supervision target."""

    def __init__(self, kernel_size: int = 15, sigma: float = 3.0) -> None:
        super().__init__()
        coordinates = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        self.register_buffer("kernel", kernel_2d.view(1, 1, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return functional.conv2d(value, self.kernel, padding=self.padding)


class ECAGNetModule(pl.LightningModule):
    """Segmentation objective plus Gaussian prior supervision (GPS)."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.model = ECAFUNet_V2(cfg.bert_type, cfg.vision_type, cfg.project_dim)
        self.model.mars_routing = getattr(cfg, "mars_routing", "default")
        self.model.fusion_mode = getattr(cfg, "fusion_mode", "ecagnet")
        self.learning_rate = cfg.lr
        self.attention_weight = getattr(cfg, "attention_loss_weight", 0.1)
        self.gps_mode = getattr(cfg, "gps_mode", "blurred")
        self.segmentation_loss = DiceCELoss()
        self.attention_loss = nn.BCELoss()
        self.blur = GaussianBlur(
            kernel_size=getattr(cfg, "gps_kernel_size", 15),
            sigma=getattr(cfg, "gps_sigma", 3.0),
        )
        self.val_iou = BinaryJaccardIndex()

        pretrained = getattr(cfg, "vision_pretrained_path", "")
        if pretrained:
            self._load_vision_pretraining(pretrained)

    def _load_vision_pretraining(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"ConvNeXt pretraining checkpoint not found: {path}. "
                "Set MODEL.vision_pretrained_path to a local checkpoint or leave it empty to train from scratch."
            )
        checkpoint = torch.load(path, map_location="cpu")
        source_state = checkpoint.get("model", checkpoint)
        target_state = self.model.state_dict()
        matched = {name: value for name, value in source_state.items() if name in target_state and value.shape == target_state[name].shape}
        target_state.update(matched)
        self.model.load_state_dict(target_state)
        self.print(f"Loaded {len(matched)} compatible ConvNeXt parameters from {path}.")

    def forward(self, batch):
        return self.model(batch)

    def _step(self, batch, stage: str) -> torch.Tensor:
        inputs, target = batch
        prediction, attention = self(inputs)
        segmentation = self.segmentation_loss(prediction, target)
        if self.gps_mode == "none":
            guidance = torch.zeros_like(segmentation)
        else:
            if attention.shape[-2:] != target.shape[-2:]:
                target_for_attention = functional.interpolate(target.float(), size=attention.shape[-2:], mode="nearest")
            else:
                target_for_attention = target.float()
            with torch.no_grad():
                guidance_target = target_for_attention if self.gps_mode == "hard" else self.blur(target_for_attention)
                guidance_target = guidance_target.clamp(0, 1)
            guidance = self.attention_loss(attention, guidance_target)
        loss = segmentation + self.attention_weight * guidance
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=target.shape[0])
        self.log(f"{stage}_seg_loss", segmentation, on_step=False, on_epoch=True, batch_size=target.shape[0])
        if self.gps_mode != "none":
            self.log(f"{stage}_gps_loss", guidance, on_step=False, on_epoch=True, batch_size=target.shape[0])
        if stage == "val":
            self.val_iou((prediction >= 0.5).int(), target.int())
            self.log("val_iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True, batch_size=target.shape[0])
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
