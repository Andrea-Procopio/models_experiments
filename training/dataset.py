import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from config import DATA_DIR, MODEL_ID, IMAGE_H, IMAGE_W

class PolygonDataset(torch.utils.data.Dataset):
    """
    Loads images + GT masks from data/{train|val}/{images,masks} + meta_shape.json
    Keeps native 400x1000; no geometric aug (to preserve concavity geometry).
    """
    def __init__(self, split: str, processor: AutoImageProcessor):
        meta = json.load(open(Path(DATA_DIR) / "meta_shape.json"))
        self.items = [m for m in meta if m["split"] == split]
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img = Image.open(rec["image"]).convert("RGB")
        mask = Image.open(rec["mask"]).convert("L")

        # assert resolution matches expected
        if img.size != (IMAGE_W, IMAGE_H):
            img = img.resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
            mask = mask.resize((IMAGE_W, IMAGE_H), Image.NEAREST)

        # to tensors
        img_arr = np.array(img)
        m_arr = (np.array(mask) > 0).astype(np.int64)  # 0/1

        processed = self.processor(images=img_arr, return_tensors="pt")
        pixel_values = processed["pixel_values"][0].contiguous()  # (3,H,W), normalized
        labels = torch.from_numpy(m_arr).contiguous()               # (H,W) int64

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "meta": rec,
        }

@dataclass
class Collator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        px = torch.stack([b["pixel_values"] for b in batch], dim=0).contiguous()
        y  = torch.stack([b["labels"] for b in batch], dim=0).long().contiguous()
        meta = [b["meta"] for b in batch]
        return {"pixel_values": px, "labels": y, "meta": meta}

def load_datasets():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    return (
        PolygonDataset("train", processor),
        PolygonDataset("val", processor),
        Collator(),
        processor,
    )
