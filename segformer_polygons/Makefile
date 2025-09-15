# ==== CONFIG (override at call-time, e.g., `make full CKPT_DIR=runs/exp1`) ====
PY ?= python3

RAW_DIR   ?= Exp3b_Images         # from config.py (used only by the scripts)
DATA_DIR  ?= data                 # from config.py
CKPT_DIR  ?= runs/segformer_b0_polygons
RADII_JSON ?= runs/radii.json
SPLIT     ?= val                  # val | train

# ==============================================================================
.PHONY: help prep deps build train calib eval eval-train eval-val full clean distclean show

help:
	@echo "Targets:"
	@echo "  prep          - ensure package dirs have __init__.py"
	@echo "  deps          - pip install requirements.txt"
	@echo "  build         - build dataset (thresholded GT, split by shape)"
	@echo "  train         - fine-tune SegFormer (writes checkpoint to CKPT_DIR)"
	@echo "  calib         - calibrate closing radii (writes $(RADII_JSON))"
	@echo "  eval          - evaluate checkpoint (IoU/Dice/BF1/CFI) on SPLIT=$(SPLIT)"
	@echo "  eval-train    - evaluate on train split"
	@echo "  eval-val      - evaluate on val split"
	@echo "  full          - prep -> deps -> build -> train -> calib -> eval(val)"
	@echo "  clean         - remove overlays and eval JSONs"
	@echo "  distclean     - also remove runs/ and data/ (DANGEROUS)"
	@echo ""
	@echo "Variables (override as needed):"
	@echo "  CKPT_DIR=$(CKPT_DIR)  RADII_JSON=$(RADII_JSON)  SPLIT=$(SPLIT)"

prep:
	@touch data_io/__init__.py training/__init__.py eval/__init__.py viz/__init__.py

deps:
	$(PY) -m pip install -r requirements.txt

build: prep
	$(PY) -m data_io.build_dataset

# train.py doesn't parse CLI args; call main() with CKPT_DIR explicitly.
train: prep
	@mkdir -p $(CKPT_DIR)
	$(PY) -c "import os; from training.train import main; main(output_dir=os.environ.get('CKPT_DIR','runs/segformer_b0_polygons'))"

calib: prep
	@mkdir -p $(dir $(RADII_JSON))
	$(PY) -m eval.calibration
	@echo "Radii written to $(RADII_JSON)"

eval: prep
	$(PY) -m eval.evaluate --ckpt $(CKPT_DIR) --radii $(RADII_JSON) --split $(SPLIT)

eval-train: prep
	$(MAKE) eval SPLIT=train

eval-val: prep
	$(MAKE) eval SPLIT=val

full: prep deps build train calib eval-val

clean:
	@rm -rf overlays || true
	@find $(CKPT_DIR) -maxdepth 1 -name 'eval_*.json' -delete || true

distclean: clean
	@rm -rf runs || true
	@rm -rf data || true

show:
	@echo "RAW_DIR=$(RAW_DIR)"
	@echo "DATA_DIR=$(DATA_DIR)"
	@echo "CKPT_DIR=$(CKPT_DIR)"
	@echo "RADII_JSON=$(RADII_JSON)"
	@echo "SPLIT=$(SPLIT)"
