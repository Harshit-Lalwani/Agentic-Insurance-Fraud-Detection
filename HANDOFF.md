# Handoff — Automotive Fraud Detection System

## Goal
Get the full fraud detection pipeline working end-to-end via the Gradio UI at `localhost:7860`.

## Current State
- All model weights present (Detectron2 x2, Keras x2)
- GPU (RTX 3050 4GB) is available and detected by PyTorch
- `.env` updated — base URL fixed from `api.build.nvidia.com` (DNS dead) to `integrate.api.nvidia.com`
- Description matching via NVIDIA NIM works when tested in isolation

## Remaining Issue
The app crashes during initialization in `FraudDetection/input.py`. Logs stop after TensorFlow loading messages (TF can't find CUDA drivers — harmless, falls back to CPU). The crash happens later — probably while loading Detectron2 or CLIP models. The exact error message is not visible because the process dies silently. Investigate the init sequence in `FraudDetectionPipeline.__init__()` starting at line 53 of `input.py`.

## Key Files
- `FraudDetection/input.py` — main app, init sequence at lines 53-90
- `FraudDetection/.env` — credentials and API config
- `FraudDetection/description_check.py` — NIM integration (already working)
- `FraudDetection/combined_damage_detector.py` — Detectron2 loader (likely crash point)
- `FraudDetection/duplication_check.py` — CLIP loader (possible crash point)

## Environment
- Virtualenv: `/root/Megathon25/venv/bin/activate`
- Working directory: `/root/Megathon25`
