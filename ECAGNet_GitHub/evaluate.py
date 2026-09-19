#!/usr/bin/env python3
"""Evaluate an ECAGNet checkpoint and report mean Dice and IoU."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from ecagnet.config import load_config
from ecagnet.data import FlexibleReportSegDataset
from ecagnet.module import ECAGNetModule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qata_cov19.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    module = ECAGNetModule.load_from_checkpoint(args.checkpoint, cfg=cfg, map_location=args.device).to(args.device).eval()
    dataset = FlexibleReportSegDataset(cfg.test_csv_path, cfg.test_root_path, cfg.bert_type, "test", cfg.image_size)
    loader = DataLoader(dataset, batch_size=cfg.valid_batch_size, shuffle=False, num_workers=cfg.num_workers)
    dice_values, iou_values = [], []
    epsilon = 1e-7
    with torch.no_grad():
        for inputs, target in loader:
            image, text = inputs
            image = image.to(args.device)
            text = {key: value.to(args.device) for key, value in text.items()}
            target = target.to(args.device).float()
            prediction, _ = module((image, text))
            prediction = (prediction >= 0.5).float()
            intersection = (prediction * target).sum(dim=(1, 2, 3))
            dice_values.extend(((2 * intersection + epsilon) / (prediction.sum((1, 2, 3)) + target.sum((1, 2, 3)) + epsilon)).cpu().tolist())
            union = ((prediction + target) > 0).float().sum(dim=(1, 2, 3))
            iou_values.extend(((intersection + epsilon) / (union + epsilon)).cpu().tolist())
    print(f"mDice: {100 * sum(dice_values) / len(dice_values):.2f}")
    print(f"mIoU: {100 * sum(iou_values) / len(iou_values):.2f}")


if __name__ == "__main__":
    main()
