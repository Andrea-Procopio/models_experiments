import io, requests, numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# ------------- config -------------
model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
image_url = "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800&q=80"  # replace with your URL
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
# ----------------------------------

# fetch image
resp = requests.get(image_url, timeout=15)
resp.raise_for_status()
image = Image.open(io.BytesIO(resp.content)).convert("RGB")

# load processor + model
processor = AutoImageProcessor.from_pretrained(model_id)
model = SegformerForSemanticSegmentation.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

# preprocess & forward
inputs = processor(images=image, return_tensors="pt").to(device)
with torch.inference_mode():
    outputs = model(**inputs)  # logits: (B, num_labels, h_out, w_out)

# post-process to original size -> integer class map (H, W)
pred_seg = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]  # (H, W)
)[0].cpu().numpy().astype(np.int32)

# visualize: random palette per label id
num_labels = model.config.num_labels
rng = np.random.default_rng(0)
palette = rng.integers(0, 255, size=(num_labels, 3), dtype=np.uint8)
color_mask = palette[pred_seg]                                  # (H, W, 3)
overlay = (0.6 * np.asarray(image) + 0.4 * color_mask).astype(np.uint8)

# save results
Image.fromarray(overlay).save("segformer_overlay.png")
with open("labels.txt", "w") as f:
    for i, name in model.config.id2label.items():
        f.write(f"{i}\t{name}\n")

# quick sanity prints
present = np.unique(pred_seg)
print("Classes present:", present.tolist())
print("Saved: segformer_overlay.png, labels.txt")