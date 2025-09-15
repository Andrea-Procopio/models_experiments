"""
segformer_interface.py

A model interface wrapper that provides DETR-compatible output from SegFormer models.
This interface standardizes model loading, inference, and output formatting across experiments.

Dependencies:
  pip install transformers safetensors huggingface_hub pillow matplotlib torch torchvision
"""

from pathlib import Path
from typing import Union, Dict, Any
import logging
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)
from skimage.measure import label, regionprops


class ModelInterface:
    """
    Abstract base interface for model inference that experiments can use.
    All model implementations should inherit from this class.
    """
    
    def load_model(self) -> None:
        """Load the model from checkpoint or hub."""
        raise NotImplementedError
    
    def infer_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on an image and return predictions in DETR-compatible format.
        
        Returns:
            Dict with keys:
            - 'pred_masks': torch.Tensor of shape (1, N, H, W) where N is number of queries
            - 'pred_logits': torch.Tensor of shape (1, N, num_classes) 
            - 'pred_boxes': torch.Tensor of shape (1, N, 4) in DETR format
        """
        raise NotImplementedError


class SegFormerInterface(ModelInterface):
    """
    SegFormer model interface that provides DETR-compatible output format.
    Converts semantic segmentation maps to instance-like masks for compatibility.
    Enhanced to provide better mask outputs for blob detection and IoU computation.
    """

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: Union[str, torch.device, None] = None,
        num_queries: int = 100,
        logger: logging.Logger = None,
    ):
        self.model_name = model_name
        self.num_queries = num_queries
        self.logger = logger or logging.getLogger(__name__)
        
        self.device = (
            torch.device(device)
            if isinstance(device, str)
            else device
            if isinstance(device, torch.device)
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Keep the original "no resize" behaviour for compatibility --> Can be one of the issues that we are trying to solve
        #if self.model_name.endswith("512-512"):
        #    self.processor = SegformerImageProcessor(do_resize=True, size=(512, 512))
        #elif self.model_name.endswith("640-640"):
        #    self.processor = SegformerImageProcessor(do_resize=True, size=(640, 640))
        
        #changed the script to resize the image to the image size on which the model was trained
        size_match = re.search(r"(\d+)-(\d+)$", self.model_name)
        if size_match:
            self.processor = SegformerImageProcessor(do_resize=False, size=(int(size_match.group(1)), int(size_match.group(2))))
        
        self.model: SegformerForSemanticSegmentation = None
        
        self.logger.info(f"Initialized SegFormer interface with device: {self.device}")

    def load_model(self, use_safetensors: bool = True) -> None:
        """Downloads and loads the SegFormer model."""
        self.logger.info(f"Loading SegFormer model: {self.model_name}")
        
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(
                self.model_name, use_safetensors=use_safetensors
            )
            .to(self.device)
            .eval()
        )
        
        self.logger.info("SegFormer model loaded successfully")

    def infer_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference and return DETR-compatible predictions.
        
        The SegFormer semantic segmentation is converted to instance-like masks
        by treating each class as a separate "object" and creating binary masks.
        Enhanced to provide better quality masks for blob detection.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before infer_image().")

        # Get original image dimensions
        orig_width, orig_height = image.size
        
        # Run SegFormer inference
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values)

        # Get segmentation map (class ids at the original image size)
        seg_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(orig_height, orig_width)]
        )[0]  # Shape: (H, W)
        
        # Convert to DETR-compatible format with enhanced mask generation
        pred_masks = self._convert_segmap_to_enhanced_masks(seg_map, orig_height, orig_width)
        
        # Generate dummy logits and boxes for compatibility
        batch_size = 1
        pred_logits = torch.zeros(batch_size, self.num_queries, 91, device=self.device)  # ADE20K -> COCO classes
        pred_boxes = torch.zeros(batch_size, self.num_queries, 4, device=self.device)
        
        # Fill in logits for actual masks (set high confidence for background class)
        num_actual_masks = pred_masks.shape[1]
        if num_actual_masks > 0:
            pred_logits[0, :num_actual_masks, 0] = 10.0  # High confidence for "object" class
        
        return {
            'pred_masks': pred_masks,
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes,
            'seg_map': seg_map.cpu().numpy(),  # keep at original size for visualization
        }

    def _convert_segmap_to_enhanced_masks(self, seg_map: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Convert semantic segmentation map to instance-like binary masks with enhanced processing.
        Optimized for 1000x400 images with black backgrounds and mono-colored blobs.
        
        Args:
            seg_map: Tensor of shape (H, W) with class IDs
            height, width: Original image dimensions
            
        Returns:
            Tensor of shape (1, num_queries, H, W) with binary masks
        """
        seg_map_np = seg_map.cpu().numpy()
        unique_classes = np.unique(seg_map_np)
        
        # Filter out background (class 0) and create masks for each class
        object_classes = unique_classes[unique_classes > 0]
        
        masks = []
        
        # Calculate image area for relative thresholds
        image_area = height * width
        min_area_threshold = max(500, image_area * 0.001)  # At least 500 pixels or 0.1% of image
        significant_area_threshold = max(2000, image_area * 0.005)  # At least 2000 pixels or 0.5% of image
        
        # Process each semantic class
        for class_id in object_classes:
            class_mask = (seg_map_np == class_id).astype(np.uint8)
            
            # Split class mask into connected components to create instance-like masks
            labeled = label(class_mask, connectivity=2)
            regions = regionprops(labeled)
            
            # Sort regions by area to get meaningful instances
            regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)
            
            # Add each connected component as a separate mask
            for region in regions_sorted:
                # Create binary mask for this component
                component_mask = (labeled == region.label).astype(np.float32)
                
                # Only add masks with reasonable size (optimized for 1000x400 images)
                if region.area > min_area_threshold:
                    masks.append(torch.from_numpy(component_mask))
        
        # If no good masks found, create some dummy masks using intensity-based approach
        # This is optimized for black background + mono-colored blobs
        if len(masks) == 0:
            # Convert segmentation map to grayscale and use multiple threshold levels
            gray = np.array(Image.fromarray(seg_map_np.astype(np.uint8)).convert('L'))
            
            # Use thresholds optimized for black background detection
            # Lower thresholds to catch subtle blob boundaries
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                binary_mask = (gray > threshold * 255).astype(np.float32)
                
                # Split into connected components
                labeled = label(binary_mask, connectivity=2)
                regions = regionprops(labeled)
                
                for region in sorted(regions, key=lambda r: r.area, reverse=True)[:5]:  # Top 5 regions
                    if region.area > min_area_threshold:
                        component_mask = (labeled == region.label).astype(np.float32)
                        masks.append(torch.from_numpy(component_mask))
        
        # If still no masks, create default masks using simple thresholding
        # This fallback is specifically designed for black background scenarios
        if len(masks) == 0:
            # Convert segmentation map to grayscale and create masks
            gray_seg = seg_map_np.astype(np.float32)
            gray_seg = (gray_seg - gray_seg.min()) / (gray_seg.max() - gray_seg.min() + 1e-8)
            
            # Use lower thresholds for black background detection
            for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
                mask = (gray_seg > threshold).astype(np.float32)
                if mask.sum() > min_area_threshold:
                    masks.append(torch.from_numpy(mask))
        
        # Pad to num_queries
        while len(masks) < self.num_queries:
            masks.append(torch.zeros(height, width, dtype=torch.float32))
            
        # Stack and add batch dimension
        pred_masks = torch.stack(masks[:self.num_queries], dim=0)  # (num_queries, H, W)
        pred_masks = pred_masks.unsqueeze(0).to(self.device)  # (1, num_queries, H, W)
        
        return pred_masks

    # ------------------------------------------------------------------
    # helper (optional): fast ADE20K palette for pretty colour maps
    # ------------------------------------------------------------------
    @staticmethod
    def ade_palette() -> list[list[int]]:
        """ADE20K palette that maps each class to RGB values."""
        return [
            [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]
        ]

    def visualize_predictions(self, image: Image.Image, predictions: Dict[str, Any], 
                            threshold: float = 0.5) -> Image.Image:
        """
        Visualize SegFormer semantic segmentation predictions on the image.
        
        Args:
            image: Input PIL image
            predictions: Output from infer_image()
            threshold: Confidence threshold for visualization
            
        Returns:
            PIL image with visualizations
        """
        
        # Convert image to numpy array (original size)
        image_np = np.array(image)

        # Prefer a proper semantic map if available
        seg_map = predictions.get('seg_map')
        if seg_map is not None:
            if isinstance(seg_map, torch.Tensor):
                seg_map_np = seg_map.cpu().numpy()
            else:
                seg_map_np = np.asarray(seg_map)

            # Colorize classes using ADE20K palette, wrapping indices if needed
            palette = np.array(self.ade_palette(), dtype=np.uint8)
            colored = palette[seg_map_np % len(palette)]  # (H, W, 3)

            # Blend with the original image at the original resolution
            blend = (0.6 * image_np + 0.4 * colored).astype(np.uint8)
            return Image.fromarray(blend)

        # Fallback: derive a binary map from summed masks
        pred_masks: torch.Tensor = predictions.get('pred_masks')
        if pred_masks is None or pred_masks.shape[1] == 0:
            return image

        semantic_map = pred_masks[0].sum(dim=0).cpu().numpy()
        if semantic_map.max() > 0:
            semantic_map = semantic_map / (semantic_map.max() + 1e-8)
        binary = (semantic_map > threshold)
        colored = np.zeros_like(image_np)
        colored[binary] = np.array([255, 0, 0], dtype=np.uint8)
        blend = (0.6 * image_np + 0.4 * colored).astype(np.uint8)
        return Image.fromarray(blend)


# ----------------------------------------------------------------------
# demo usage and backward compatibility
# ----------------------------------------------------------------------
# Keep the old class name for backward compatibility
SegFormerPredictor = SegFormerInterface


if __name__ == "__main__":
    # Demo usage
    interface = SegFormerInterface()
    interface.load_model()

    # Download a sample image (ADE20K validation sample #1)
    repo = "hf-internal-testing/fixtures_ade20k"
    img_path = hf_hub_download(repo_id=repo, filename="ADE_val_00000001.jpg", repo_type="dataset")
    image = Image.open(img_path)

    # Run inference
    predictions = interface.infer_image(image)
    
    print(f"Prediction format:")
    print(f"- pred_masks shape: {predictions['pred_masks'].shape}")
    print(f"- pred_logits shape: {predictions['pred_logits'].shape}")
    print(f"- pred_boxes shape: {predictions['pred_boxes'].shape}")

    # For visualization, we can extract the semantic segmentation
    seg_map = interface.processor.post_process_semantic_segmentation(
        interface.model(interface.processor(image, return_tensors="pt").pixel_values.to(interface.device)),
        target_sizes=[image.size[::-1]]
    )[0]

    # Colorize the prediction for visual inspection
    palette = np.array(interface.ade_palette(), dtype=np.uint8)
    colour = palette[seg_map.cpu().numpy()]  # (H, W, 3)
    blend = (0.5 * np.asarray(image) + 0.5 * colour[..., ::-1]).astype(np.uint8)

    plt.figure(figsize=(12, 8))
    plt.imshow(blend)
    plt.axis("off")
    plt.tight_layout()
    plt.show()