# SAM (Segment Anything Model) Interface

This directory contains the SAM model interface for use with the vision experiments. The interface provides DETR-compatible output format from SAM models for seamless integration with existing experiment pipelines.

## Features

- **Automatic Segmentation**: Uses SAM's automatic point generation to create segmentation masks without manual prompts
- **DETR-Compatible Output**: Converts SAM outputs to match DETR format for experiment compatibility
- **Pre-trained Models**: Uses Hugging Face's pre-trained SAM models
- **Multiple Variants**: Supports different SAM model sizes (vit-huge, vit-large, vit-base)
- **Quality Filtering**: Automatically filters masks by IoU threshold for better quality

## Usage

### Basic Usage

```python
from sam.sam_interface import SAMInterface
from PIL import Image

# Initialize interface
interface = SAMInterface(model_name="facebook/sam-vit-huge")
interface.load_model()

# Run inference
image = Image.open("your_image.jpg")
predictions = interface.infer_image(image)

print(f"Prediction format:")
print(f"- pred_masks shape: {predictions['pred_masks'].shape}")
print(f"- pred_logits shape: {predictions['pred_logits'].shape}")
print(f"- pred_boxes shape: {predictions['pred_boxes'].shape}")
```

### Using with Experiments

The SAM interface can be used with any of the three experiments:

#### Experiment 1: Causality Analysis
```bash
python exp1Causality.py --model_interface sam --videos_dir /path/to/videos --output_dir /path/to/output
```

#### Experiment 2: Time-to-Collision (TTC)
```bash
python exp2TTC.py --model_interface sam --videos_dir /path/to/videos --csv_path /path/to/data.csv --output_dir /path/to/output
```

#### Experiment 3: Change Detection
```bash
python exp3Change.py --model_interface sam --images_dir /path/to/images --output_dir /path/to/output
```

#### Experiment 3B: Correlation Analysis
```bash
python exp3b_correlation.py --model_interface sam --images_dir /path/to/images --output_dir /path/to/output
```

### Custom Model Names

Currently supported models:
- `facebook/sam-vit-huge` (default) - Largest model, best quality
- `facebook/sam-vit-large` - Medium size, good balance
- `facebook/sam-vit-base` - Smallest model, fastest inference

### Configuration Parameters

The SAM interface supports several configuration parameters:

```python
interface = SAMInterface(
    model_name="facebook/sam-vit-huge",
    num_points_per_side=32,        # Grid density for automatic points
    pred_iou_thresh=0.88,          # IoU threshold for mask filtering
    stability_score_thresh=0.95,    # Stability threshold for mask filtering
    num_queries=100                 # Maximum number of output masks
)
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How It Works

1. **Automatic Point Generation**: Creates a grid of points across the input image
2. **SAM Inference**: Runs SAM model on these points to generate segmentation masks
3. **Quality Filtering**: Filters masks by IoU score and stability
4. **Format Conversion**: Converts SAM outputs to DETR-compatible format
5. **Output Standardization**: Provides consistent interface across all model types

## Advantages of SAM

- **Zero-shot Performance**: Works on any image without training
- **High Quality Masks**: Produces precise, detailed segmentation masks
- **Automatic Operation**: No need for manual prompts or bounding boxes
- **Robust Detection**: Handles various object types and image conditions

## Limitations

- **Slower Inference**: Larger models can be computationally expensive
- **Memory Usage**: Requires significant GPU memory for large models
- **Point Density**: Quality depends on the density of automatic points
- **No Class Information**: Only provides segmentation masks, not object classes
