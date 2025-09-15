import os, torch
from transformers import SegformerForSemanticSegmentation, TrainingArguments, Trainer
from transformers.trainer_utils import IntervalStrategy
from torch.utils.data import DataLoader
from typing import Dict, Any

from .dataset import load_datasets
from .losses import dice_loss_from_logits
from eval.metrics import iou_dice, boundary_f1
from config import (
    MODEL_ID, NUM_LABELS, DATA_DIR,
    LR, EPOCHS, BATCH_TRAIN, BATCH_EVAL,
    WEIGHT_DECAY, LOG_STEPS, EVAL_STEPS, SAVE_STEPS
)

ID2LABEL = {0: "background", 1: "polygon"}
LABEL2ID = {"background": 0, "polygon": 1}

class CEDiceTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs: Dict[str, Any], return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        # Drop non-model keys like 'meta'
        inputs = {k: v for k, v in inputs.items() if k in {"pixel_values", "attention_mask"}}
        outputs = model(**inputs)           # logits: (B,2,h,w)
        logits = outputs.logits
        logits = logits.contiguous()
        # resize labels to logits shape if needed
        if logits.shape[-2:] != labels.shape[-2:]:
            labels = torch.nn.functional.interpolate(
                labels.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest"
            ).squeeze(1).long()
        # Ensure labels are on the same device as logits
        if labels.device != logits.device:
            labels = labels.to(logits.device)
        labels = labels.contiguous()
        # MPS has issues with class-weighted CE; disable weights on MPS
        weight = self.class_weights
        if weight is not None:
            try:
                weight = weight.to(logits.device)
            except Exception:
                weight = None
        if logits.device.type == "mps":
            weight = None
        ce = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        # Flatten to (N*C, num_classes) and (N*C,) to avoid internal view issues
        if logits.dim() == 4:
            b, c, h, w = logits.shape
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, c).contiguous()
            labels_flat = labels.reshape(-1).contiguous()
            loss_ce = ce(logits_flat, labels_flat)
        else:
            loss_ce = ce(logits, labels)
        # Dice can trigger backend stride issues on some platforms; guard it
        try:
            loss_dice = dice_loss_from_logits(logits, labels)
            loss = loss_ce + 0.5 * loss_dice
        except Exception:
            loss = loss_ce
        return (loss, outputs) if return_outputs else loss

def estimate_class_weights(ds, max_samples=400):
    # crude inverse frequency to avoid all-background collapse
    import numpy as np
    n = min(max_samples, len(ds))
    bg = fg = 0
    for i in range(n):
        fg += (ds[i]["labels"].numpy() == 1).sum()
        bg += (ds[i]["labels"].numpy() == 0).sum()
    tot = max(1, bg + fg)
    w_bg = tot / (2 * max(1, bg))
    w_fg = tot / (2 * max(1, fg))
    w = torch.tensor([w_bg, w_fg], dtype=torch.float32)
    return torch.clamp(w, 0.25, 4.0)

def main(output_dir="runs/segformer_b0_polygons"):
    train_ds, val_ds, collator, processor = load_datasets()

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_ID,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # reset head
    )
    # Force CPU to avoid MPS tensor stride issues with SegFormer
    device = "cpu"
    model.to(device)

    class_weights = estimate_class_weights(train_ds).to(model.device)

    # Manual training loop to avoid backend view/stride issues in Trainer/accelerate
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=0,
        pin_memory=False, collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_EVAL, shuffle=False, num_workers=0,
        pin_memory=False, collate_fn=collator
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Cosine schedule with warmup (approximate): warmup 5% of steps
    acc_steps = max(1, int(os.environ.get("ACC_STEPS", os.environ.get("GRAD_ACCUM", 1))))
    total_steps = max(1, int(len(train_ds) / max(1, BATCH_TRAIN)) * int(EPOCHS) / acc_steps)
    warmup_steps = max(1, int(0.05 * total_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine from 1 -> 0
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        import math
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -----------------------------
    # Resume checkpoint if present
    # -----------------------------
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
    start_epoch = 0
    global_step = 0
    if os.path.isfile(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=model.device)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"]) 
            if "optim" in ckpt:
                optimizer.load_state_dict(ckpt["optim"]) 
            if "sched" in ckpt:
                scheduler.load_state_dict(ckpt["sched"]) 
            start_epoch = int(ckpt.get("epoch", 0))
            global_step = int(ckpt.get("global_step", 0))
            print(f"Resumed from checkpoint: epoch={start_epoch}, global_step={global_step}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint at {checkpoint_path}: {e}")
    model.train()
    for epoch in range(int(start_epoch), int(EPOCHS)):
        running = 0.0
        count = 0
        micro_idx = 0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(model.device).contiguous()
            labels = batch["labels"].to(model.device).contiguous()

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits.contiguous()
            # resize labels to logits size
            if logits.shape[-2:] != labels.shape[-2:]:
                labels = torch.nn.functional.interpolate(
                    labels.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest"
                ).squeeze(1).long().contiguous()

            # Cross-entropy on flattened tensors
            b, c, h, w = logits.shape
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, c).contiguous()
            labels_flat = labels.reshape(-1).contiguous()
            weight = class_weights if model.device.type != "mps" else None
            ce = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=255)
            loss_ce = ce(logits_flat, labels_flat)

            # Dice (guarded)
            try:
                loss_dice = dice_loss_from_logits(logits, labels)
                loss = loss_ce + 0.5 * loss_dice
            except Exception:
                loss = loss_ce

            # Gradient accumulation
            loss_to_backprop = loss / acc_steps
            loss_to_backprop.backward()
            micro_idx += 1

            if (micro_idx % acc_steps == 0) or (micro_idx == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running += float(loss.detach().cpu())
            count += 1

        # Validation with metrics (IoU, Dice, BF1)
        model.eval()
        with torch.no_grad():
            val_running = 0.0
            val_count = 0
            val_iou = 0.0
            val_dice = 0.0
            val_bf1 = 0.0
            n_imgs_seen = 0
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(model.device).contiguous()
                labels = batch["labels"].to(model.device).contiguous()
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits.contiguous()
                if logits.shape[-2:] != labels.shape[-2:]:
                    labels = torch.nn.functional.interpolate(
                        labels.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest"
                    ).squeeze(1).long().contiguous()
                b, c, h, w = logits.shape
                logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, c).contiguous()
                labels_flat = labels.reshape(-1).contiguous()
                ce = torch.nn.CrossEntropyLoss(ignore_index=255)
                val_loss = ce(logits_flat, labels_flat)
                val_running += float(val_loss.detach().cpu())
                val_count += 1

                # === FULL-RES METRICS (use HF post-process to (H,W)) ===
                # Keep the original full-res labels BEFORE we resized them for loss
                labels_full = batch["labels"].to(model.device).contiguous()  # (B,H,W)
                B = labels_full.shape[0]
                H, W = labels_full.shape[-2], labels_full.shape[-1]

                # Upsample model outputs back to (H,W) for each sample
                pred_ids_list = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[(H, W)] * B
                )  # list of length B; each is (H,W) with class ids

                for i in range(B):
                    pred_ids = pred_ids_list[i].cpu().numpy().astype("uint8")   # (H,W)
                    pred = (pred_ids == 1).astype("uint8")
                    gt   = (labels_full[i] == 1).to(torch.uint8).cpu().numpy()

                    iou, d = iou_dice(pred, gt)
                    bf1    = boundary_f1(pred, gt)
                    val_iou += iou
                    val_dice += d
                    val_bf1 += bf1
                n_imgs_seen += B
            val_iou /= max(1, n_imgs_seen)
            val_dice /= max(1, n_imgs_seen)
            val_bf1  /= max(1, n_imgs_seen)
        print(f"Epoch {epoch+1}/{int(EPOCHS)} - train_loss={running/max(1,count):.4f} val_loss={val_running/max(1,val_count):.4f} IoU={val_iou:.3f} Dice={val_dice:.3f} BF1={val_bf1:.3f}")

        # -----------------------------
        # Save checkpoint (epoch-level)
        # -----------------------------
        state = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
        }
        torch.save(state, checkpoint_path)
        # Also keep a rolling per-epoch copy (optional)
        torch.save(state, os.path.join(output_dir, f"checkpoint-epoch{epoch+1}.pt"))
        model.train()

    # Save artifacts similar to Trainer
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "id2label.json"), "w") as f:
        f.write('{"0":"background","1":"polygon"}')

if __name__ == "__main__":
    main()
