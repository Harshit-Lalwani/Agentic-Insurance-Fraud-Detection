# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Warning:** This repo is heavily "vibe coded" — code, docs, and comments may be unreliable. Verify before trusting.

## Commands

```bash
# Run the web app (from repo root)
source /root/Megathon25/venv/bin/activate && python /root/Megathon25/FraudDetection/input.py
# Opens Gradio UI at http://localhost:7860

# Run individual pipeline stages via CLI (from FraudDetection/)
python cli.py --step all --image path/to/img.jpg --description "scratched door" --customer "Jane Doe" --car "Honda Civic 2020"
python cli.py --step {ai|tampering|description|duplication|damage} --image path/to/img.jpg

# Install deps (from FraudDetection/)
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install git+https://github.com/openai/CLIP.git
```

There is no test suite, linter, or build step in this repo.

## Hard prerequisites

The app will fail to start or crash mid-pipeline without these:
- `FraudDetection/.env` with at least one of `GOOGLE_API_KEY` (Gemini, primary) or `NVIDIA_API_KEY` (NVIDIA NIM, fallback) — see `FraudDetection/.env.example`; also supports `GEMINI_MODEL` / `NVIDIA_NIM_BASE_URL` / `NVIDIA_NIM_MODEL` overrides
- `damage-det/model_parts.pth` and `damage-det/model_damage.pth` — Detectron2 Mask R-CNN weights (~335MB each), not in git
- `FraudDetection/Image-Tampering-Detection-using-ELA-and-Metadata-Analysis/{ELA_Training/model_ela.h5, WeatherCNNTraining/Weather_Model.h5}` — TF/Keras weights, not in git
- A CUDA-capable NVIDIA GPU. `input.py` calls `torch.cuda.is_available()` at import time and `sys.exit(1)`s immediately if false — CPU fallback is deliberately disabled because Detectron2 + HuggingFace + CLIP inference together is too slow on CPU for an interactive app.

## Architecture

The system is a **5-stage sequential fraud detection pipeline** for automotive insurance claims, orchestrated by `FraudDetectionPipeline` in `FraudDetection/input.py`. Each stage is a separately-instantiated detector class; the pipeline wires them together and can reject a claim early at any stage:

1. **AI-generation check** (`ai_detector.py`) — HuggingFace ensemble (`Organika/sdxl-detector`, `umm-maybe/AI-image-detector`) flags synthetic images.
2. **Tampering check** (`tampering_check.py`) — ELA (Error Level Analysis) via a DenseNet121 Keras model, plus EXIF/GPS/timestamp cross-validation against a weather API. Rejects above a 60% tampering score.
3. **Description matching** (`description_check.py`) — sends image + user-provided description to Gemini (`gemini-3.5-flash`, primary VLM) to verify the described damage matches what's visible; automatically falls back to NVIDIA NIM (Llama 3.2 Vision 11B, OpenAI-compatible API) if the Gemini call raises an exception.
4. **Duplication check** (`duplication_check.py`) — perceptual hashes (aHash/pHash/dHash) plus CLIP (ViT-B/32) embedding similarity against every image under `fraud_detection_data/`, to catch resubmitted claim photos.
5. **Damage analysis & cost estimation** (`combined_damage_detector.py`, built on `car_parts_detector.py`) — two separate Detectron2 Mask R-CNN models (21 car-part classes, 8 damage-type classes), overlap between part and damage masks determines severity (Low/Med/High by % of part area), which maps to a repair cost in INR.

`FraudDetectionPipeline.__init__()` wraps each detector's construction in its own `try/except` — a missing model file disables that stage (sets it to `None`) rather than crashing the whole app, so partial pipelines run in degraded mode. Check `input.py` around the init sequence when a stage silently doesn't run.

Each processed submission is written to `fraud_detection_data/<customer>_<timestamp>/` (images + `metadata.json`), which also serves as the duplicate-detection database for stage 4. This directory is gitignored.

`cli.py` is a thin argparse wrapper that either runs the full pipeline (importing the singleton `pipeline` from `input.py`) or exercises a single stage's detector directly — useful for isolating which stage is failing without booting the Gradio UI.

## Known quirks

- `tampering_check.py` imports both `tf_keras` and `tf.keras` as a fallback — Keras version mismatches are a common failure point.
- TensorFlow logging "can't find CUDA drivers" during startup is expected/harmless — TF falls back to CPU while PyTorch still uses the GPU.
