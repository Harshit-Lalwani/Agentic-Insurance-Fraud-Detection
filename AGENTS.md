# AGENTS.md — Automotive Fraud Detection System

**Warning:** This repo is heavily "vibe coded" — code, docs, and comments may be unreliable. Verify before trusting.

## Quick start

```bash
source /root/Megathon25/venv/bin/activate && python /root/Megathon25/FraudDetection/input.py
```

## Key structure

| Path | Role |
|------|------|
| `FraudDetection/input.py` | Main app — Gradio UI + `FraudDetectionPipeline` orchestrator |
| `FraudDetection/ai_detector.py` | AI-generated image detection (HuggingFace ensemble) |
| `FraudDetection/tampering_check.py` | ELA + metadata tampering detection (TensorFlow/Keras models) |
| `FraudDetection/description_check.py` | NVIDIA NIM VLM description matching (Llama 3.2 Vision 11B) |
| `FraudDetection/duplication_check.py` | Perceptual hash + CLIP duplicate detection |
| `FraudDetection/combined_damage_detector.py` | Car parts + damage segmentation via Detectron2 |
| `FraudDetection/car_parts_detector.py` | Standalone Detectron2 parts detector |
| `damage-det/` | Expects `model_parts.pth` and `model_damage.pth` |

## Requirements

**Hard prerequisites (not optional):**
- `FraudDetection/.env` with `NVIDIA_API_KEY=...`
- `damage-det/model_parts.pth` — 351 MB car parts Mask R-CNN weights
- `damage-det/model_damage.pth` — 351 MB damage Mask R-CNN weights
- TensorFlow/Keras models under `FraudDetection/Image-Tampering-Detection-using-ELA-and-Metadata-Analysis/` (`model_ela.h5`, `Weather_Model.h5`)

## Developer commands

```bash
# Install (from FraudDetection/)
pip install -r requirements.txt

# Special installs: detectron2 + CLIP (both from GitHub)
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install git+https://github.com/openai/CLIP.git
```

## Quirks & gotchas

- `tampering_check.py` imports both `tf_keras` and `tf.keras` as fallback — Keras compatibility issues are common.
- NVIDIA GPU with CUDA 11.8+ required (app errors out if no GPU detected).
- All submission data saved to `fraud_detection_data/` (gitignored).
