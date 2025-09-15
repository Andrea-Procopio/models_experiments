#!/usr/bin/env bash
set -euo pipefail

# defaults (can be overridden via env or flags)
CKPT_DIR="${CKPT_DIR:-runs/segformer_b0_polygons}"
RADII_JSON="${RADII_JSON:-runs/radii.json}"
SPLIT="${SPLIT:-val}"

cmd="${1:-help}"
shift || true

case "$cmd" in
  full)        make full CKPT_DIR="$CKPT_DIR" RADII_JSON="$RADII_JSON" ;;
  build)       make build ;;
  train)       make train CKPT_DIR="$CKPT_DIR" ;;
  calib)       make calib RADII_JSON="$RADII_JSON" ;;
  eval)        make eval CKPT_DIR="$CKPT_DIR" RADII_JSON="$RADII_JSON" SPLIT="${1:-$SPLIT}" ;;
  eval-train)  make eval-train CKPT_DIR="$CKPT_DIR" RADII_JSON="$RADII_JSON" ;;
  eval-val)    make eval-val CKPT_DIR="$CKPT_DIR" RADII_JSON="$RADII_JSON" ;;
  clean)       make clean ;;
  distclean)   make distclean ;;
  *)           make help ;;
esac