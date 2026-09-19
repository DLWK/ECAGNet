from __future__ import annotations

import os

import pandas as pd
import torch
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, NormalizeIntensityd, RandZoomd, Resized, ToTensord
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class FlexibleReportSegDataset(Dataset):
    """Dataset wrapper for QaTa-COV19 and MosMedData+ CSV/path conventions."""

    def __init__(self, csv_path, root_path, tokenizer, mode="train", image_size=(224, 224)):
        self.mode = mode
        self.root_path = root_path
        self.image_size = image_size
        self.data = pd.read_csv(csv_path)
        self.image_list = list(self.data["Image"])
        text_col = "Description" if "Description" in self.data.columns else "text"
        self.caption_list = list(self.data[text_col])
        if mode == "train" and "QaTa-COV19" in root_path:
            cut = int(0.8 * len(self.image_list))
            self.image_list = self.image_list[:cut]
            self.caption_list = self.caption_list[:cut]
        elif mode == "valid" and "QaTa-COV19" in root_path:
            cut = int(0.8 * len(self.image_list))
            self.image_list = self.image_list[cut:]
            self.caption_list = self.caption_list[cut:]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image = os.path.join(self.root_path, "img", image_name)
        qata_mask = os.path.join(self.root_path, "labelcol", "mask_" + image_name)
        mos_mask = os.path.join(self.root_path, "labelcol", image_name)
        gt = qata_mask if os.path.exists(qata_mask) else mos_mask
        token_output = self.tokenizer.encode_plus(
            str(self.caption_list[idx]),
            padding="max_length",
            max_length=24,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        data = {
            "image": image,
            "gt": gt,
            "token": token_output["input_ids"],
            "mask": token_output["attention_mask"],
        }
        data = self.transform()(data)
        image_tensor, gt_tensor = data["image"], data["gt"]
        gt_tensor = torch.where(gt_tensor > 0, 1, 0)
        gt_tensor = (torch.sum(gt_tensor, axis=0) > 0).int()
        gt_tensor = torch.unsqueeze(gt_tensor, 0)
        text = {"input_ids": data["token"].squeeze(0), "attention_mask": data["mask"].squeeze(0)}
        return [image_tensor, text], gt_tensor

    def transform(self):
        if self.mode == "train":
            return Compose(
                [
                    LoadImaged(["image", "gt"], reader="PILReader"),
                    EnsureChannelFirstd(["image", "gt"]),
                    RandZoomd(["image", "gt"], min_zoom=0.95, max_zoom=1.2, mode=["bicubic", "nearest"], prob=0.1),
                    Resized(["image"], spatial_size=self.image_size, mode="bicubic"),
                    Resized(["gt"], spatial_size=self.image_size, mode="nearest"),
                    NormalizeIntensityd(["image"], channel_wise=True),
                    ToTensord(["image", "gt", "token", "mask"]),
                ]
            )
        return Compose(
            [
                LoadImaged(["image", "gt"], reader="PILReader"),
                EnsureChannelFirstd(["image", "gt"]),
                Resized(["image"], spatial_size=self.image_size, mode="bicubic"),
                Resized(["gt"], spatial_size=self.image_size, mode="nearest"),
                NormalizeIntensityd(["image"], channel_wise=True),
                ToTensord(["image", "gt", "token", "mask"]),
            ]
        )
