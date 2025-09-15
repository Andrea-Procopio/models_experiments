#!/usr/bin/env python3
"""
Adapter that exposes a v1-compatible interface on top of segformer_interface_v2.SegFormerRunner.

Provides:
- class SegFormerInterfaceAdapter with methods: load_model(), infer_image(PIL.Image) -> {'pred_masks': torch.Tensor}
- optional visualize_predictions(image, predictions) for overlays used by experiments
"""

from typing import Any, Dict, Optional, List

import numpy as np
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from .segformer_interface_v2 import SegFormerRunner


class SegFormerInterfaceAdapter:
    def __init__(self, model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
                 device: Optional[str] = None,
                 target_long_side: int = 512,
                 tau: float = 0.6,
                 max_components: int = 10) -> None:
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_long_side = int(target_long_side) if target_long_side else None
        self.tau = float(tau)
        self.max_components = int(max_components)

        self.model: Optional[SegformerForSemanticSegmentation] = None
        self.processor: Optional[SegformerImageProcessor] = None
        self.runner: Optional[SegFormerRunner] = None

    def load_model(self) -> None:
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name).to(self.device).eval()
        # We handle resize/letterbox ourselves via the runner; keep do_resize=False here
        self.processor = SegformerImageProcessor(do_resize=False, do_normalize=True)
        self.runner = SegFormerRunner(self.model, self.processor, device=str(self.device),
                                      target_long_side=self.target_long_side, tau=self.tau)

    def infer_image(self, image: Image.Image) -> Dict[str, Any]:
        if self.runner is None:
            raise RuntimeError("Call load_model() before infer_image().")
        img_np = np.array(image.convert("RGB"), dtype=np.uint8)
        out = self.runner.segment(img_np, topk=self.max_components, return_probs=False)
        masks: List[np.ndarray] = out.get("masks", [])
        h, w = img_np.shape[:2]

        if not masks:
            pred_masks = torch.zeros(1, 0, h, w, dtype=torch.float32, device=self.device)
        else:
            stack = np.stack([m.astype(np.float32) for m in masks], axis=0)  # [N, H, W], values 0/1
            pred_masks = torch.from_numpy(stack).unsqueeze(0).to(self.device)  # [1, N, H, W]

        # Dummy logits/boxes for compatibility when needed downstream
        return {"pred_masks": pred_masks}

    def visualize_predictions(self, image: Image.Image, predictions: Dict[str, Any], threshold: float = 0.5) -> Image.Image:
        """Simple overlay: union of masks in red over the RGB image."""
        img_np = np.array(image.convert("RGB"), dtype=np.uint8)
        h, w = img_np.shape[:2]
        pred_masks: torch.Tensor = predictions.get("pred_masks")
        if pred_masks is None or pred_masks.shape[1] == 0:
            return image
        masks = (pred_masks[0].detach().cpu().numpy() > threshold).astype(np.uint8)  # [N, H, W]
        union = (masks.sum(axis=0) > 0)
        overlay = img_np.copy()
        overlay[union] = np.array([255, 0, 0], dtype=np.uint8)
        blend = (0.6 * img_np + 0.4 * overlay).astype(np.uint8)
        return Image.fromarray(blend, mode="RGB")


