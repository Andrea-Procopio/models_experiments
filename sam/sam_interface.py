"""
sam_interface.py

A model interface wrapper that provides DETR-compatible output from SAM (Segment Anything Model).
This interface standardizes model loading, inference, and output formatting across experiments.

Dependencies:
  pip install transformers safetensors huggingface_hub pillow matplotlib torch torchvision
"""

from pathlib import Path
from typing import Union, Dict, Any, List, Tuple, Optional
import logging
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor
from skimage.measure import label, regionprops, find_contours
import io


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


class SAMInterface(ModelInterface):
    """
    SAM (Segment Anything Model) interface that provides DETR-compatible output format.
    Uses automatic point generation to create segmentation masks without manual prompts.
    
    Polygon Detection Modes:
    - "auto": Automatically detect and choose best strategy (default)
    - "single": Force single-polygon mode (optimized for exp3b experiments)
    - "two_polygon": Force two-polygon mode (optimized for exp1 collision frames)
    
    The interface automatically generates optimal points for each mode:
    - Single-polygon: Center + boundary + interior points for one object
    - Two-polygon: Center + boundary + interior points for each of two objects
    """

    def __init__(
        self,
        model_name: str = "facebook/sam-vit-huge",
        device: Union[str, torch.device, None] = None,
        num_queries: int = 100,
        logger: logging.Logger = None,
        num_points_per_side: int = 16,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: List[List[float]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        optimize_points: bool = True,  # New parameter for point optimization
        perfect_mask_threshold: float = 0.95,  # SAM confidence threshold to consider mask "perfect"
        polygon_detection_mode: str = "auto",  # "auto", "single", "two_polygon"
    ):
        self.model_name = model_name
        self.num_queries = num_queries
        self.logger = logger or logging.getLogger(__name__)
        
        # SAM-specific parameters
        self.num_points_per_side = num_points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.point_grids = point_grids
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.optimize_points = optimize_points
        self.perfect_mask_threshold = perfect_mask_threshold
        self.polygon_detection_mode = polygon_detection_mode  # New parameter
        
        self.device = (
            torch.device(device)
            if isinstance(device, str)
            else device
            if isinstance(device, torch.device)
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self.model: SamModel = None
        self.processor: SamProcessor = None
        
        self.logger.info(f"Initialized SAM interface with device: {self.device}")
        self.logger.info(f"Polygon detection mode: {self.polygon_detection_mode}")

    def set_polygon_detection_mode(self, mode: str) -> None:
        """
        Set the polygon detection mode dynamically.
        
        Args:
            mode: "auto", "single", or "two_polygon"
                - "auto": Automatically detect and choose best strategy
                - "single": Force single-polygon mode (for exp3b)
                - "two_polygon": Force two-polygon mode (for exp1)
        """
        if mode not in ["auto", "single", "two_polygon"]:
            raise ValueError(f"Invalid polygon detection mode: {mode}. Must be 'auto', 'single', or 'two_polygon'")
        
        self.polygon_detection_mode = mode
        self.logger.info(f"Polygon detection mode changed to: {mode}")

    def load_model(self, use_safetensors: bool = True) -> None:
        """Downloads and loads the SAM model."""
        self.logger.info(f"Loading SAM model: {self.model_name}")
        
        try:
            self.processor = SamProcessor.from_pretrained(self.model_name)
            self.model = (
                SamModel.from_pretrained(
                    self.model_name, use_safetensors=use_safetensors
                )
                .to(self.device)
                .eval()
            )
            self.logger.info("SAM model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load SAM model {self.model_name}: {e}")
            raise

    def infer_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference and return DETR-compatible predictions.
        
        Optimized: First tries center point, if IoU is near-perfect, skips remaining points.
        Otherwise, continues with adaptive points for better coverage.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Call load_model() before infer_image().")

        # Get original image dimensions
        orig_width, orig_height = image.size
        
        # Start with center point first (most likely to be perfect for centered polygons)
        center_x, center_y = orig_width // 2, orig_height // 2
        center_point = [center_x, center_y]
        
        all_masks: List[torch.Tensor] = []   # each of shape [3, 1, H, W]
        all_scores: List[torch.Tensor] = []  # each of shape [3]
        
        # Try center point first
        center_masks, center_scores = self._run_sam_at_point(image, center_point)
        if center_masks is not None:
            all_masks.extend(center_masks)
            all_scores.extend(center_scores)
            
            # Check if center point gave near-perfect masks (SAM confidence > threshold)
            best_center_score = self._get_best_iou_score(center_scores)
            self.logger.info(f"Center point best SAM confidence: {best_center_score:.3f}")
            
            if self._has_perfect_mask(center_scores):
                self.logger.info(f"Center point produced high-confidence masks (SAM score > {self.perfect_mask_threshold}), skipping additional points")
            else:
                # Center point wasn't perfect, try adaptive points
                self.logger.info(f"Center point masks not confident enough (best SAM score <= {self.perfect_mask_threshold}), trying adaptive points for better coverage")
                additional_masks, additional_scores = self._run_adaptive_points(image)
                all_masks.extend(additional_masks)
                all_scores.extend(additional_scores)
        else:
            # Center point failed, fall back to adaptive points
            self.logger.warning("Center point failed, falling back to adaptive points")
            additional_masks, additional_scores = self._run_adaptive_points(image)
            all_masks.extend(additional_masks)
            all_scores.extend(additional_scores)

        # If nothing valid was produced, return empty DETR format
        if not all_masks:
            empty_pred_masks = torch.zeros(1, self.num_queries, orig_height, orig_width, device=self.device)
            pred_logits = torch.zeros(1, self.num_queries, 91, device=self.device)
            pred_boxes = torch.zeros(1, self.num_queries, 4, device=self.device)
            return {
                'pred_masks': empty_pred_masks,
                'pred_logits': pred_logits,
                'pred_boxes': pred_boxes
            }

        # Concatenate across all points → shapes: [P*3, 1, H, W] and [P*3]
        masks_cat = torch.cat(all_masks, dim=0)
        scores_cat = torch.cat(all_scores, dim=0)

        # Convert to DETR-compatible format
        pred_masks = self._convert_sam_masks_to_detr_format(
            masks_cat, scores_cat, orig_height, orig_width
        )

        # Generate dummy logits and boxes for compatibility
        batch_size = 1
        num_masks = pred_masks.shape[1]
        pred_logits = torch.zeros(batch_size, num_masks, 91, device=self.device)
        pred_boxes = torch.zeros(batch_size, num_masks, 4, device=self.device)

        if num_masks > 0:
            pred_logits[0, :num_masks, 0] = 10.0

        return {
            'pred_masks': pred_masks,
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }
    
    def _run_sam_at_point(self, image: Image.Image, point: List[int]) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """Run SAM inference at a specific point and return masks and scores."""
        try:
            x, y = point
            inputs = self.processor(
                image,
                input_points=[[[int(x), int(y)]]],  # shape: [1, 1, 2]
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process masks to original image size
            processed_masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]  # shape: [3, 1, H, W]

            # IoU scores for the 3 masks
            scores = outputs.iou_scores.squeeze()  # shape: [3]

            # Return as lists for consistency
            if isinstance(processed_masks, torch.Tensor) and processed_masks.ndim == 4:
                masks_list = [processed_masks]
                scores_list = [scores.detach().cpu() if isinstance(scores, torch.Tensor) else torch.tensor(scores)]
                return masks_list, scores_list
            else:
                return None, None
                
        except Exception as e:
            self.logger.warning(f"SAM inference failed at point ({x},{y}): {e}")
            return None, None
    
    def _run_adaptive_points(self, image: Image.Image) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run SAM inference on adaptive points (excluding center)."""
        # Generate adaptive points based on polygon detection mode
        if self.polygon_detection_mode == "two_polygon":
            # Force two-polygon mode for exp1 experiments
            points_xy: List[List[int]] = self._generate_two_polygon_points(image)
            self.logger.info(f"Using two-polygon point generation mode for exp1")
        elif self.polygon_detection_mode == "single":
            # Force single-polygon mode for exp3b experiments
            points_xy: List[List[int]] = self._generate_adaptive_points(image)
            self.logger.info(f"Using single-polygon point generation mode for exp3b")
        else:  # "auto" mode - detect automatically
            # Try two-polygon first, fall back to single if needed
            points_xy: List[List[int]] = self._generate_two_polygon_points(image)
            # Note: _generate_two_polygon_points already falls back to _generate_adaptive_points if needed
            self.logger.info(f"Using auto polygon detection mode")
        
        # Remove center point if it's in the list (since we already tried it)
        center_x, center_y = image.width // 2, image.height // 2
        points_xy = [p for p in points_xy if p != [center_x, center_y]]
        
        all_masks: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []

        # Run SAM independently for each point
        for x, y in points_xy:
            try:
                inputs = self.processor(
                    image,
                    input_points=[[[int(x), int(y)]]],  # shape: [1, 1, 2]
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Post-process masks to original image size
                processed_masks = self.processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )[0]  # shape: [3, 1, H, W]

                # IoU scores for the 3 masks
                scores = outputs.iou_scores.squeeze()  # shape: [3]

                # Accumulate
                if isinstance(processed_masks, torch.Tensor) and processed_masks.ndim == 4:
                    all_masks.append(processed_masks)
                    all_scores.append(scores.detach().cpu() if isinstance(scores, torch.Tensor) else torch.tensor(scores))
            except Exception as e:
                self.logger.warning(f"SAM single-point inference failed at ({x},{y}): {e}")
                continue
        
        return all_masks, all_scores
    
    def _has_perfect_mask(self, scores: List[torch.Tensor]) -> bool:
        """Check if the best mask from center point has SAM confidence score > threshold."""
        if not scores:
            return False
            
        # Find the best score across all masks from the center point
        best_score = 0.0
        
        for score_tensor in scores:
            if isinstance(score_tensor, torch.Tensor):
                # Get the maximum score from this tensor
                max_score = score_tensor.max().item()
                best_score = max(best_score, max_score)
            elif isinstance(score_tensor, (list, np.ndarray)):
                # Handle other numeric types
                max_score = max(score_tensor) if score_tensor else 0
                best_score = max(best_score, max_score)
        
        # Only return True if the BEST mask exceeds the threshold
        return best_score > self.perfect_mask_threshold
    
    def _get_best_iou_score(self, scores: List[torch.Tensor]) -> float:
        """Get the best SAM confidence score from a list of score tensors."""
        if not scores:
            return 0.0
            
        best_score = 0.0
        
        for score_tensor in scores:
            if isinstance(score_tensor, torch.Tensor):
                # Get the maximum score from this tensor
                max_score = score_tensor.max().item()
                best_score = max(best_score, max_score)
            elif isinstance(score_tensor, (list, np.ndarray)):
                # Handle other numeric types
                max_score = max(score_tensor) if score_tensor else 0
                best_score = max(best_score, max_score)
        
        return best_score

    def _generate_adaptive_points(self, image: Image.Image) -> List[List[int]]:
        """Generate optimized, minimal points for geometric shapes."""
        points: List[List[int]] = []
        
        try:
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Find the colored polygon (non-black pixels)
            # Use a higher threshold to avoid noise
            non_black = np.any(img_array > 50, axis=2)
            
            if non_black.sum() > 100:  # Significant polygon detected
                # Find polygon boundaries using contour detection
                from skimage.measure import find_contours
                contours = find_contours(non_black, 0.5)
                
                if contours:
                    # Get the largest contour (main polygon)
                    main_contour = max(contours, key=len)
                    
                    # 1. Center of mass of the actual polygon (most important)
                    center_y, center_x = np.mean(main_contour, axis=0)
                    points.append([int(center_x), int(center_y)])
                    
                    # 2. Boundary points (configurable density)
                    if self.optimize_points:
                        num_boundary_points = 4  # Optimized: fewer points
                    else:
                        num_boundary_points = 8  # Original: more points
                    
                    for i in range(num_boundary_points):
                        idx = int(i * len(main_contour) / num_boundary_points)
                        y, x = main_contour[idx]
                        points.append([int(x), int(y)])
                    
                    # 3. Interior points (configurable density)
                    y_coords, x_coords = np.where(non_black)
                    if len(y_coords) > 0:
                        if self.optimize_points:
                            # Optimized: only 1 interior point
                            mid_idx = len(y_coords) // 2
                            points.append([int(x_coords[mid_idx]), int(y_coords[mid_idx])])
                        else:
                            # Original: 2-3 interior points
                            for i in range(min(3, len(y_coords) // 100)):
                                idx = i * len(y_coords) // 3
                                points.append([int(x_coords[idx]), int(y_coords[idx])])
                    
                    mode = "optimized" if self.optimize_points else "full"
                    self.logger.info(f"Generated {len(points)} {mode} points for polygon")
                    return points
                    
        except Exception as e:
            self.logger.warning(f"Optimized point generation failed: {e}, falling back to center-based")
        
        # Fallback: center-based approach if adaptive fails
        center_y, center_x = image.height // 2, image.width // 2
        points = [[center_x, center_y]]
        
        # Add strategic points around center (configurable density)
        radius = min(image.height, image.width) // 8
        if self.optimize_points:
            num_directions = 4  # Optimized: fewer points
            angle_step = np.pi / 2  # 4 directions: up, right, down, left
        else:
            num_directions = 8  # Original: more points
            angle_step = np.pi / 4  # 8 directions around center
            
        for i in range(num_directions):
            angle = i * angle_step
            x = center_x + int(radius * np.cos(angle))
            y = center_y + int(radius * np.sin(angle))
            points.append([x, y])
        
        return points

    def _generate_two_polygon_points(self, image: Image.Image) -> List[List[int]]:
        """
        Generate points optimized for two-polygon collision frames.
        Specifically designed for exp1 causality experiments with two objects.
        Enhanced to avoid background inclusion and ensure proper separation.
        """
        points: List[List[int]] = []
        
        try:
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Enhanced polygon detection with better thresholding
            # Use multiple color channels to detect non-black regions more accurately
            gray = np.mean(img_array, axis=2)
            non_black = gray > 30  # More conservative threshold
            
            if non_black.sum() < 200:  # Not enough content
                self.logger.warning("Insufficient polygon content detected")
                return self._generate_adaptive_points(image)
            
            # Apply morphological operations to clean up the mask
            from skimage.morphology import binary_opening, binary_closing
            from skimage.measure import find_contours
            
            # Clean up the mask
            cleaned_mask = binary_opening(non_black, footprint=np.ones((3, 3)))
            cleaned_mask = binary_closing(cleaned_mask, footprint=np.ones((5, 5)))
            
            # Find contours with better parameters
            contours = find_contours(cleaned_mask, 0.5)
            
            if len(contours) >= 2:
                # Sort contours by area (largest first)
                contours = sorted(contours, key=len, reverse=True)
                
                self.logger.info(f"Detected {len(contours)} contours, using top 2 for two-polygon mode")
                
                # Generate points for each of the top 2 polygons with better separation
                for i, contour in enumerate(contours[:2]):
                    # Calculate contour properties
                    contour_area = len(contour)
                    if contour_area < 50:  # Skip very small contours
                        continue
                    
                    # Center of mass for this polygon (more robust calculation)
                    center_y, center_x = np.mean(contour, axis=0)
                    
                    # Ensure the center point is within the actual polygon
                    if self._is_point_in_polygon(center_x, center_y, contour):
                        points.append([int(center_x), int(center_y)])
                    
                    # Add boundary points with better distribution
                    num_boundary_points = 4 if self.optimize_points else 6
                    for j in range(num_boundary_points):
                        idx = int(j * len(contour) / num_boundary_points)
                        y, x = contour[idx]
                        
                        # Ensure boundary points are on the actual contour
                        if 0 <= int(x) < image.width and 0 <= int(y) < image.height:
                            points.append([int(x), int(y)])
                    
                    # Add interior points with better selection
                    interior_points = self._find_interior_points_for_contour(contour, cleaned_mask, image.width, image.height)
                    points.extend(interior_points)
                    
                    self.logger.info(f"Generated {len(interior_points) + num_boundary_points + 1} points for polygon {i+1}")
                
                if len(points) >= 4:  # Ensure we have enough points
                    self.logger.info(f"Generated {len(points)} total points for 2 polygons (two-polygon mode)")
                    return points
                    
            elif len(contours) == 1:
                self.logger.info("Only 1 contour detected, attempting to split it")
                # Try to split the single contour into two parts
                split_points = self._split_single_contour(contours[0], cleaned_mask, image.width, image.height)
                if split_points:
                    points.extend(split_points)
                    self.logger.info(f"Split single contour into {len(split_points)} points")
                    return points
                else:
                    self.logger.info("Failed to split contour, falling back to single-polygon mode")
                    return self._generate_adaptive_points(image)
            
            # Fallback if we don't have enough points
            self.logger.warning("Insufficient points generated, falling back to adaptive mode")
            return self._generate_adaptive_points(image)
                    
        except Exception as e:
            self.logger.warning(f"Two-polygon point generation failed: {e}, falling back to adaptive")
            return self._generate_adaptive_points(image)

    def _is_point_in_polygon(self, x: float, y: float, contour: np.ndarray) -> bool:
        """Check if a point is inside a polygon contour."""
        from matplotlib.path import Path
        polygon_path = Path(contour)
        return polygon_path.contains_point((x, y))

    def _find_interior_points_for_contour(self, contour: np.ndarray, mask: np.ndarray, width: int, height: int) -> List[List[int]]:
        """Find interior points specifically within a contour."""
        interior_points = []
        
        try:
            # Create a mask for this specific contour
            from skimage.draw import polygon
            contour_coords = contour.astype(int)
            rr, cc = polygon(contour_coords[:, 0], contour_coords[:, 1], shape=mask.shape)
            
            # Find points within the contour that are also in the cleaned mask
            valid_points = []
            for r, c in zip(rr, cc):
                if 0 <= r < height and 0 <= c < width and mask[r, c]:
                    valid_points.append([c, r])  # Note: x=c, y=r
            
            if len(valid_points) > 0:
                # Select interior points with good distribution
                num_interior = 2 if self.optimize_points else 3
                step = max(1, len(valid_points) // num_interior)
                
                for i in range(0, len(valid_points), step):
                    if len(interior_points) < num_interior:
                        interior_points.append(valid_points[i])
            
        except Exception as e:
            self.logger.warning(f"Interior point generation failed: {e}")
        
        return interior_points

    def _split_single_contour(self, contour: np.ndarray, mask: np.ndarray, width: int, height: int) -> List[List[int]]:
        """Attempt to split a single large contour into two parts."""
        try:
            # Find the longest axis of the contour
            contour_coords = contour.astype(int)
            y_coords = contour_coords[:, 0]
            x_coords = contour_coords[:, 1]
            
            y_range = y_coords.max() - y_coords.min()
            x_range = x_coords.max() - x_coords.min()
            
            split_points = []
            
            if y_range > x_range:
                # Split vertically
                mid_y = (y_coords.max() + y_coords.min()) // 2
                
                # Find points on the left and right sides
                left_points = [(x, y) for x, y in zip(x_coords, y_coords) if y < mid_y]
                right_points = [(x, y) for x, y in zip(x_coords, y_coords) if y >= mid_y]
                
                if left_points and right_points:
                    # Add center points for each side
                    left_center_x = sum(x for x, y in left_points) // len(left_points)
                    left_center_y = sum(y for x, y in left_points) // len(left_points)
                    right_center_x = sum(x for x, y in right_points) // len(right_points)
                    right_center_y = sum(y for x, y in right_points) // len(right_points)
                    
                    split_points.extend([[left_center_x, left_center_y], [right_center_x, right_center_y]])
            else:
                # Split horizontally
                mid_x = (x_coords.max() + x_coords.min()) // 2
                
                # Find points on the top and bottom sides
                top_points = [(x, y) for x, y in zip(x_coords, y_coords) if x < mid_x]
                bottom_points = [(x, y) for x, y in zip(x_coords, y_coords) if x >= mid_x]
                
                if top_points and bottom_points:
                    # Add center points for each side
                    top_center_x = sum(x for x, y in top_points) // len(top_points)
                    top_center_y = sum(y for x, y in top_points) // len(top_points)
                    bottom_center_x = sum(x for x, y in bottom_points) // len(bottom_points)
                    bottom_center_y = sum(y for x, y in bottom_points) // len(bottom_points)
                    
                    split_points.extend([[top_center_x, top_center_y], [bottom_center_x, bottom_center_y]])
            
            return split_points
            
        except Exception as e:
            self.logger.warning(f"Contour splitting failed: {e}")
            return []

    def _generate_automatic_points(self, height: int, width: int) -> List[List[int]]:
        """Generate adaptive points based on actual image content for geometric shapes."""
        points: List[List[int]] = []
        
        # Start with center point as fallback
        center_y, center_x = height // 2, width // 2
        points.append([center_x, center_y])
        
        # Add a few strategic points around center for robustness
        radius = min(height, width) // 8
        for i in range(4):
            angle = i * np.pi / 2  # 4 directions: up, right, down, left
            x = center_x + int(radius * np.cos(angle))
            y = center_y + int(radius * np.sin(angle))
            points.append([x, y])
        
        return points

    def _convert_sam_masks_to_detr_format(
        self, 
        masks: torch.Tensor, 
        scores: torch.Tensor, 
        height: int, 
        width: int
    ) -> torch.Tensor:
        """
        Convert SAM masks to DETR-compatible format.
        
        Args:
            masks: SAM masks of shape [num_points, 1, H, W] or [num_points, 3, H, W] or [3, 1, H, W] or [1, 3, H, W]
            scores: IoU scores of shape [num_points*3] or [num_points]
            height, width: Original image dimensions
            
        Returns:
            Tensor of shape [1, num_queries, H, W] with binary masks
        """
        # Ensure scores is a 1D CPU tensor
        if isinstance(scores, np.ndarray):
            scores = torch.from_numpy(scores)
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores)
        scores = scores.detach().cpu().view(-1)

        # Normalize masks to shape [N, 1, H, W]
        if not isinstance(masks, torch.Tensor):
            masks = torch.as_tensor(masks)
        # Move to CPU for safety in shape ops
        masks = masks.detach().cpu()

        if masks.ndim == 4 and masks.shape[1] == 1:
            # [N, 1, H, W] - already normalized
            normalized_masks = masks
        elif masks.ndim == 4 and masks.shape[1] == 3:
            # [P, 3, H, W] -> flatten first two dims to [P*3, 1, H, W]
            P, three, Hm, Wm = masks.shape
            normalized_masks = masks.reshape(P * three, 1, Hm, Wm)
        elif masks.ndim == 4 and masks.shape[0] == 1 and masks.shape[1] == 3:
            # [1, 3, H, W] -> [3, 1, H, W]
            normalized_masks = masks.permute(1, 0, 2, 3)
        elif masks.ndim == 3 and masks.shape[0] == 3:
            # [3, H, W] -> [3, 1, H, W]
            normalized_masks = masks.unsqueeze(1)
        else:
            # As a last resort, try to flatten any 4D [A, B, H, W] to [A*B, 1, H, W]
            if masks.ndim == 4:
                A, B, Hm, Wm = masks.shape
                normalized_masks = masks.reshape(A * B, 1, Hm, Wm)
            else:
                # Unexpected shape: return empty tensor
                return torch.zeros(1, self.num_queries, height, width, device=self.device)

        # After normalization, shapes should match: len(scores) == normalized_masks.shape[0]
        N = normalized_masks.shape[0]
        if scores.numel() != N:
            # If scores length mismatches (e.g., scores is per-point and N is per-proposal), try repeating
            # Common case: scores length == N // 3
            if scores.numel() * 3 == N:
                scores = scores.repeat_interleave(3)
            else:
                # Fallback: truncate or pad scores to N
                if scores.numel() > N:
                    scores = scores[:N]
                else:
                    pad = torch.zeros(N - scores.numel())
                    scores = torch.cat([scores, pad], dim=0)

        # Filter masks by IoU threshold
        valid_indices = scores > self.pred_iou_thresh
        if valid_indices.numel() != N:
            valid_indices = valid_indices.view(-1)
        valid_masks = normalized_masks[valid_indices]
        valid_scores = scores[valid_indices]

        if len(valid_masks) == 0:
            # Return empty tensor if no valid masks
            return torch.zeros(1, self.num_queries, height, width, device=self.device)

        # Sort by score and take top masks
        sorted_indices = torch.argsort(valid_scores, descending=True)
        top_masks = valid_masks[sorted_indices[:self.num_queries]]

        # Convert to binary masks
        binary_masks = (top_masks > 0.5).float()

        # Pad or truncate to match num_queries
        if binary_masks.shape[0] < self.num_queries:
            # Pad with zeros
            padding = torch.zeros(
                self.num_queries - binary_masks.shape[0], 
                1, height, width, 
                device=binary_masks.device
            )
            binary_masks = torch.cat([binary_masks, padding], dim=0)
        elif binary_masks.shape[0] > self.num_queries:
            # Truncate
            binary_masks = binary_masks[:self.num_queries]

        # Remove channel dim to get [N, H, W]
        if binary_masks.ndim == 4 and binary_masks.shape[1] == 1:
            binary_masks = binary_masks[:, 0, :, :]
        # Reshape to DETR format: [1, num_queries, H, W]
        return binary_masks.unsqueeze(0)

    def visualize_predictions(
        self, 
        image: Image.Image, 
        predictions: Dict[str, Any], 
        threshold: float = 0.5
    ) -> Image.Image:
        """Visualize SAM predictions on the image."""
        import matplotlib.pyplot as plt
        
        # Convert PIL image to numpy for visualization
        img_array = np.array(image)
        
        # Get masks [N, H, W] or [N, 1, H, W]
        masks = predictions['pred_masks'][0]  # Remove batch dimension
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0, :, :]
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Image with masks overlay
        axes[1].imshow(img_array)
        
        # Overlay masks
        for i in range(min(masks.shape[0], 10)):
            mask = masks[i].cpu().numpy()
            if mask.sum() > 0:  # Only show non-empty masks
                axes[1].imshow(mask, alpha=0.3, cmap='jet')
        
        axes[1].set_title(f"SAM Segmentation (showing {min(masks.shape[0], 10)} masks)")
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Convert matplotlib figure to PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        vis_image = Image.open(buf)
        plt.close()
        
        return vis_image
