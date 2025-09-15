"""
detr_interface.py

A model interface wrapper that provides DETR-compatible output from DETR segmentation models.
This interface standardizes model loading, inference, and output formatting across experiments.
Uses ONLY transformers library with the specified model_name.

Dependencies:
  pip install transformers safetensors huggingface_hub pillow matplotlib torch torchvision
"""

from pathlib import Path
from typing import Union, Dict, Any
import logging
import io
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from transformers import (
    DetrImageProcessor,
    DetrForSegmentation,
)
from transformers.models.detr.feature_extraction_detr import rgb_to_id
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


class DetrInterface(ModelInterface):
    """
    DETR segmentation model interface that provides instance segmentation outputs.
    Uses ONLY transformers library with the specified model_name.
    """

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50-panoptic",
        device: Union[str, torch.device, None] = None,
        num_queries: int = 100,
        logger: logging.Logger = None,
        confidence_threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.num_queries = num_queries
        self.confidence_threshold = confidence_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        self.device = (
            torch.device(device)
            if isinstance(device, str)
            else device
            if isinstance(device, torch.device)
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # DETR processor and model from transformers
        self.processor = None
        self.model = None
        
        self.logger.info(f"Initialized DETR Segmentation interface with device: {self.device}")
        self.logger.info(f"Will load model: {self.model_name}")

    def load_model(self, use_safetensors: bool = True) -> None:
        """Load the DETR segmentation model using transformers library ONLY."""
        self.logger.info(f"Loading DETR segmentation model: {self.model_name}")
        
        try:
            # Load processor and model from transformers
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = (
                DetrForSegmentation.from_pretrained(
                    self.model_name, use_safetensors=use_safetensors
                )
                .to(self.device)
                .eval()
            )
            self.logger.info("DETR segmentation model loaded successfully via transformers")
            
        except Exception as e:
            self.logger.error(f"Failed to load DETR model {self.model_name}: {e}")
            raise

    def infer_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference and return DETR-compatible predictions using transformers.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Call load_model() before infer_image().")

        # Get original image dimensions
        orig_width, orig_height = image.size
        
        try:
            # Run DETR segmentation inference
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use the original DETR outputs for logits and boxes
            pred_logits = outputs.logits  # (1, num_queries, num_classes)
            pred_boxes = outputs.pred_boxes  # (1, num_queries, 4)
            
            # Ensure masks are resized to original image dimensions
            pred_masks = outputs.pred_masks  # (1, num_queries, H, W)
            
            # Check if masks need resizing
            if pred_masks.shape[-2:] != (orig_height, orig_width):
                # Resize masks to original image dimensions
                resized_masks = []
                for i in range(pred_masks.shape[1]):
                    mask = pred_masks[0, i]  # (H, W)
                    mask_pil = Image.fromarray(mask.cpu().numpy())
                    mask_resized = mask_pil.resize((orig_width, orig_height), Image.BILINEAR)
                    resized_masks.append(torch.from_numpy(np.array(mask_resized)).float())
                
                pred_masks = torch.stack(resized_masks, dim=0).unsqueeze(0).to(self.device)
            
            return {
                'pred_masks': pred_masks,
                'pred_logits': pred_logits,
                'pred_boxes': pred_boxes
            }
            
        except Exception as e:
            self.logger.error(f"Error in DETR inference: {e}")
            self.logger.error("Full traceback:")
            for line in traceback.format_exc().splitlines():
                self.logger.error(line)
            # Return empty predictions to avoid crashing
            return self._get_empty_predictions(orig_height, orig_width)

    def _get_empty_predictions(self, height: int, width: int) -> Dict[str, Any]:
        """Return empty predictions when inference fails."""
        batch_size = 1
        pred_masks = torch.zeros(batch_size, self.num_queries, height, width, device=self.device)
        pred_logits = torch.zeros(batch_size, self.num_queries, 91, device=self.device)  # 91 COCO classes
        pred_boxes = torch.zeros(batch_size, self.num_queries, 4, device=self.device)
        
        return {
            'pred_masks': pred_masks,
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }

    def _convert_panoptic_to_masks(self, panoptic_seg_id: np.ndarray, 
                                  segments_info: list, height: int, width: int) -> torch.Tensor:
        """
        Convert panoptic segmentation to individual binary masks.
        
        Args:
            panoptic_seg_id: Panoptic segmentation map with segment IDs
            segments_info: Information about each segment
            height, width: Original image dimensions
            
        Returns:
            Tensor of shape (1, num_queries, H, W) with binary masks
        """
        
        # Resize panoptic segmentation to original image size
        if panoptic_seg_id.shape != (height, width):
            panoptic_pil = Image.fromarray(panoptic_seg_id.astype(np.uint8))
            panoptic_pil = panoptic_pil.resize((width, height), Image.NEAREST)
            panoptic_seg_id = np.array(panoptic_pil)
        
        # Initialize masks list
        masks = []
        
        # Create masks from segments
        for segment in segments_info:
            segment_id = segment['id']
            
            # Create binary mask for this segment
            mask = (panoptic_seg_id == segment_id).astype(np.float32)
            
            if mask.sum() > 0:  # Only add non-empty masks
                masks.append(torch.from_numpy(mask))
        
        # If we don't have enough masks, create additional masks by splitting larger segments
        if len(masks) < self.num_queries:
            # Find larger segments and split them into connected components
            for segment in segments_info:
                if len(masks) >= self.num_queries:
                    break
                    
                segment_id = segment['id']
                mask = (panoptic_seg_id == segment_id).astype(np.uint8)
                
                # Split into connected components
                labeled = label(mask, connectivity=2)
                regions = regionprops(labeled)
                
                # Add largest connected components as separate masks
                for region in sorted(regions, key=lambda r: r.area, reverse=True):
                    if len(masks) >= self.num_queries:
                        break
                    if region.area > 50:  # Minimum area threshold
                        component_mask = (labeled == region.label).astype(np.float32)
                        masks.append(torch.from_numpy(component_mask))
        
        # If still not enough masks, create simple grid-based masks as fallback
        while len(masks) < self.num_queries:
            grid_idx = len(masks) % 16  # Create up to 16 different grid positions
            grid_size = 4  # 4x4 grid
            row = grid_idx // grid_size
            col = grid_idx % grid_size
            
            # Create mask in grid cell
            mask = np.zeros((height, width), dtype=np.float32)
            
            cell_h = height // grid_size
            cell_w = width // grid_size
            y1 = row * cell_h
            y2 = min((row + 1) * cell_h, height)
            x1 = col * cell_w
            x2 = min((col + 1) * cell_w, width)
            
            # Add small central region in each grid cell
            center_y = (y1 + y2) // 2
            center_x = (x1 + x2) // 2
            mask_h = max(1, cell_h // 4)
            mask_w = max(1, cell_w // 4)
            
            mask_y1 = max(0, center_y - mask_h // 2)
            mask_y2 = min(height, center_y + mask_h // 2)
            mask_x1 = max(0, center_x - mask_w // 2)
            mask_x2 = min(width, center_x + mask_w // 2)
            
            if mask_y2 > mask_y1 and mask_x2 > mask_x1:
                mask[mask_y1:mask_y2, mask_x1:mask_x2] = 0.1  # Lower confidence for fallback masks
            
            masks.append(torch.from_numpy(mask))
        
        # Stack and add batch dimension
        pred_masks = torch.stack(masks[:self.num_queries], dim=0)  # (num_queries, H, W)
        pred_masks = pred_masks.unsqueeze(0).to(self.device)  # (1, num_queries, H, W)
        
        return pred_masks

    # ------------------------------------------------------------------
    # Helper methods for visualization and compatibility
    # ------------------------------------------------------------------
    def get_coco_labels(self) -> Dict[int, str]:
        """Get COCO class labels used by DETR."""
        return {
            0: 'N/A', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
            7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'N/A',
            13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog',
            19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
            25: 'giraffe', 26: 'N/A', 27: 'backpack', 28: 'umbrella', 29: 'handbag', 30: 'tie',
            31: 'suitcase', 32: 'frisbee', 33: 'skis', 34: 'snowboard', 35: 'sports ball',
            36: 'kite', 37: 'baseball bat', 38: 'baseball glove', 39: 'skateboard', 40: 'surfboard',
            41: 'tennis racket', 42: 'bottle', 43: 'wine glass', 44: 'cup', 45: 'fork', 46: 'knife',
            47: 'spoon', 48: 'bowl', 49: 'banana', 50: 'apple', 51: 'sandwich', 52: 'orange',
            53: 'broccoli', 54: 'carrot', 55: 'hot dog', 56: 'pizza', 57: 'donut', 58: 'cake',
            59: 'chair', 60: 'couch', 61: 'potted plant', 62: 'bed', 63: 'dining table',
            64: 'toilet', 65: 'tv', 66: 'laptop', 67: 'mouse', 68: 'remote', 69: 'keyboard',
            70: 'cell phone', 71: 'microwave', 72: 'oven', 73: 'toaster', 74: 'sink',
            75: 'refrigerator', 76: 'N/A', 77: 'book', 78: 'clock', 79: 'vase', 80: 'scissors',
            81: 'teddy bear', 82: 'hair drier', 83: 'toothbrush'
        }

    def visualize_predictions(self, image: Image.Image, predictions: Dict[str, Any], 
                            threshold: float = 0.5) -> Image.Image:
        """
        Visualize DETR segmentation predictions on the image.
        
        Args:
            image: Input PIL image
            predictions: Output from infer_image()
            threshold: Confidence threshold for visualization
            
        Returns:
            PIL image with visualizations
        """
        
        # Convert image to numpy array
        image_np = np.array(image)
        overlay = image_np.copy()
        
        pred_masks = predictions['pred_masks'][0]  # Remove batch dimension
        
        # Generate colors for masks
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),    # Dark Red
            (0, 128, 0),    # Dark Green
            (0, 0, 128),    # Dark Blue
            (128, 128, 0),  # Olive
        ]
        
        # Apply masks with different colors
        for i in range(pred_masks.shape[0]):
            mask = pred_masks[i].cpu().numpy()
            if mask.sum() > 50:  # Only visualize masks with some content
                # Get color for this mask
                color = colors[i % len(colors)]
                
                # Apply color to mask region with transparency
                mask_bool = mask > threshold
                overlay[mask_bool] = overlay[mask_bool] * 0.6 + np.array(color) * 0.4
        
        # Create a simple side-by-side comparison
        # Resize both images to the same height for comparison
        height = image_np.shape[0]
        width = image_np.shape[1]
        
        # Create a combined image: original | overlay
        combined_width = width * 2
        combined_image = np.zeros((height, combined_width, 3), dtype=np.uint8)
        
        # Place original image on the left
        combined_image[:, :width] = image_np
        
        # Place overlay on the right
        combined_image[:, width:] = overlay.astype(np.uint8)
        
        # Add text labels (simple colored rectangles)
        label_height = 30
        combined_image[:label_height, :width] = [200, 200, 200]  # Gray background
        combined_image[:label_height, width:] = [200, 200, 200]  # Gray background
        
        return Image.fromarray(combined_image)


# ----------------------------------------------------------------------
# demo usage and backward compatibility
# ----------------------------------------------------------------------
# Keep the old class name for backward compatibility
DetrPredictor = DetrInterface


if __name__ == "__main__":
    # Demo usage
    interface = DetrInterface()
    interface.load_model()

    # Download a sample image (COCO validation sample)
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    try:
        response = requests.get(url)
        from io import BytesIO
        image = Image.open(BytesIO(response.content))
    except:
        # Fallback to HuggingFace sample if COCO is not accessible
        repo = "hf-internal-testing/fixtures_ade20k"
        img_path = hf_hub_download(repo_id=repo, filename="ADE_val_00000001.jpg", repo_type="dataset")
        image = Image.open(img_path)

    # Run inference
    predictions = interface.infer_image(image)
    
    print(f"Prediction format:")
    print(f"- pred_masks shape: {predictions['pred_masks'].shape}")
    print(f"- pred_logits shape: {predictions['pred_logits'].shape}")
    print(f"- pred_boxes shape: {predictions['pred_boxes'].shape}")
    print(f"- Number of non-empty masks: {(predictions['pred_masks'].sum(dim=(-2, -1)) > 0).sum().item()}")

    # For visualization
    results_vis = interface.visualize_predictions(image, predictions)
    
    # Save visualization
    results_vis.save("detr_segmentation_demo_output.png")
    print("Saved visualization to detr_segmentation_demo_output.png")