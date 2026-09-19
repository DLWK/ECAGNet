#!/usr/bin/env python3
"""Train ECAGNet with a YAML configuration file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from ecagnet.config import load_config
from ecagnet.data import FlexibleReportSegDataset
from ecagnet.module import ECAGNetModule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qata_cov19.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = load_config(args.config)
    pl.seed_everything(args.seed, workers=True)

    train_data = FlexibleReportSegDataset(cfg.train_csv_path, cfg.train_root_path, cfg.bert_type, "train", cfg.image_size)
    valid_data = FlexibleReportSegDataset(cfg.valid_csv_path, cfg.valid_root_path, cfg.bert_type, "valid", cfg.image_size)
    loader_options = {"num_workers": cfg.num_workers, "pin_memory": True}
    train_loader = DataLoader(train_data, batch_size=cfg.train_batch_size, shuffle=True, **loader_options)
    valid_loader = DataLoader(valid_data, batch_size=cfg.valid_batch_size, shuffle=False, **loader_options)

    output_dir = Path(cfg.output_dir)
    checkpoint = ModelCheckpoint(output_dir / "checkpoints", "ecagnet-{epoch:03d}-{val_loss:.4f}", monitor="val_loss", mode="min", save_top_k=1)
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=cfg.patience)
    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        min_epochs=cfg.min_epochs,
        max_epochs=cfg.max_epochs,
        callbacks=[checkpoint, early_stop],
        default_root_dir=output_dir,
    )
    trainer.fit(ECAGNetModule(cfg), train_loader, valid_loader)
    print(f"Best checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
