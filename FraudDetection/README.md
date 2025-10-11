# Automotive Fraud Detection System

A comprehensive AI-powered system to detect fraudulent automotive insurance claims by analyzing damage images through multiple validation stages.

## Features

### 🔍 Four-Stage Fraud Detection Pipeline

1. **AI Generation Detection**
   - Uses ensemble of 2 models (Organika/sdxl-detector, umm-maybe/AI-image-detector)
   - Identifies AI-generated fake images
   - High confidence threshold prevents processing of synthetic images

2. **Tampering Detection**
   - Error Level Analysis (ELA) using DenseNet121
   - Weather metadata validation using EXIF data
   - Detects photo manipulation and editing

3. **Description Matching**
   - Powered by Google Gemini AI
   - Validates if images match damage descriptions
   - Identifies car parts and damage types

4. **Duplication Detection**
   - Perceptual hashing (aHash, pHash, dHash) for fast comparison
   - CLIP embeddings for semantic similarity
   - Checks against all previously submitted images in database
   - Prevents duplicate claim submissions

### 🌐 Web Interface

- User-friendly Gradio interface
- Upload multiple images at once
- Real-time processing and detailed reports
- All submissions saved with timestamps

## System Architecture

```
input.py (Gradio Web Interface)
    ↓
    ├─→ ai_detector.py (AI Generation Check)
    │   ├─ If AI-generated with high confidence → REJECT
    │   └─ Otherwise → Continue
    │
    ├─→ tampering_check.py (Tampering Detection)
    │   ├─ If tampered (score > 60%) → REJECT
    │   └─ Otherwise → Continue
    │
    ├─→ description_check.py (Description Match)
    │   ├─ If description mismatch → REJECT
    │   └─ Otherwise → Continue
    │
    └─→ duplication_check.py (Duplication Check)
        ├─ If duplicate found → REJECT
        └─ If all pass → APPROVE & SAVE TO DATABASE
```

## Installation

### 1. Clone Repository
```bash
cd /home/alookaladdoo/Megathon25/FraudDetection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Create a `.env` file in the FraudDetection directory:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your Gemini API key from: https://makersuite.google.com/app/apikey

### 4. Download Model Files
Ensure these model files are in place:
- `Image-Tampering-Detection-using-ELA-and-Metadata-Analysis/ELA_Training/model_ela.h5`
- `Image-Tampering-Detection-using-ELA-and-Metadata-Analysis/WeatherCNNTraining/Weather_Model.h5`

## Usage

### Start the Web Interface
```bash
python input.py
```

The system will:
1. Initialize all detection models
2. Start a web server at `http://localhost:7860`
3. Open the interface in your browser

### Using the Interface

1. **Enter Customer Information**
   - Customer Name: Full name of the claimant
   - Car Details: Vehicle make, model, year, license plate

2. **Upload Images**
   - Click "Upload Car Damage Images"
   - Select one or multiple images
   - Supported formats: JPG, JPEG, PNG

3. **Provide Descriptions**
   - Enter damage descriptions (one per image)
   - Separate with commas or new lines
   - Example: "Damaged front bumper, Scratched door"

4. **Analyze**
   - Click "Analyze for Fraud"
   - Wait for processing (may take 1-2 minutes per image)
   - Review detailed report

### Example Input
```
Customer Name: John Smith
Car Details: Toyota Camry 2020, License: ABC-1234
Images: [front_damage.jpg, side_scratch.jpg]
Descriptions: Damaged front bumper with dent, Scratched passenger door
```

## File Structure

```
FraudDetection/
├── input.py                    # Main Gradio interface
├── ai_detector.py              # AI generation detection module
├── tampering_check.py          # Image tampering detection module
├── description_check.py        # Image-description matching module
├── duplication_check.py        # Image duplication detection module
├── tampering_check.py          # Image tampering detection module
├── description_check.py        # Image-description matching module
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .env                        # Environment variables (create this)
│
├── fraud_detection_data/       # Auto-created submission storage
│   └── CustomerName_20241012_143025/
│       ├── images/             # Uploaded images
│       │   ├── image_1_front.jpg
│       │   └── image_2_side.jpg
│       └── metadata.json       # Submission details and results
│
└── Image-Tampering-Detection-using-ELA-and-Metadata-Analysis/
    ├── ELA_Training/
    │   └── model_ela.h5        # ELA detection model
    └── WeatherCNNTraining/
        └── Weather_Model.h5     # Weather classification model
```

## Output Format

### Success Case
```
============================================================
AUTOMOTIVE FRAUD DETECTION REPORT
============================================================
Customer Name: John Smith
Car Details: Toyota Camry 2020
Submission Date: 2024-10-12 14:30:25
Number of Images: 2
============================================================

[Image 1/2]
[STEP 1/3] AI GENERATION CHECK
AI Generation Probability: 12.34%
Verdict: Human-Generated
RESULT: Passed AI check

[STEP 2/3] TAMPERING CHECK
Tampering Score: 23.45%
ELA Prediction: Real
RESULT: Passed tampering check

[STEP 3/3] DESCRIPTION MATCHING
Match Type: Exact Match
Confidence: 0.92
Car Part: Front-bumper
RESULT: Passed description check

============================================================
FINAL VERDICT
============================================================
Images Analyzed: 2
Images Passed: 2
Images Rejected: 0

OVERALL STATUS: PASSED - No fraud detected
============================================================
```

### Fraud Detected Case
```
[STEP 1/3] AI GENERATION CHECK
AI Generation Probability: 87.65%
Verdict: AI-Generated
Confidence: High

RESULT: FRAUD DETECTED - Image is AI-generated
STATUS: REJECTED
```

## API Details

### AIImageDetector
```python
from ai_detector import AIImageDetector

detector = AIImageDetector()
result = detector.detect("image.jpg")
# Returns: {'is_ai_generated': bool, 'ai_percentage': float, ...}
```

### TamperingDetector
```python
from tampering_check import TamperingDetector

detector = TamperingDetector(ela_model_path, weather_model_path)
result = detector.detect("image.jpg")
# Returns: {'is_tampered': bool, 'tampering_score': float, ...}
```

### DescriptionMatcher
```python
from description_check import DescriptionMatcher

matcher = DescriptionMatcher(api_key="your_key")
result = matcher.verify("image.jpg", "damaged bumper")
# Returns: {'matches': bool, 'confidence': float, ...}
```

## Troubleshooting

### Models Not Loading
- Ensure TensorFlow 2.15+ is installed: `pip install tensorflow==2.15.0`
- Check if model files exist in correct paths
- For Keras compatibility issues: `pip install tf-keras`

### Gemini API Errors
- Verify API key in `.env` file
- Check API quota: https://console.cloud.google.com/
- Ensure internet connection for API calls

### Memory Issues
- Process images one at a time
- Reduce image resolution if needed
- Close other applications to free RAM

### Gradio Port Already in Use
Change port in `input.py`:
```python
app.launch(server_port=7861)  # Use different port
```

## Performance

- **AI Detection**: ~5-10 seconds per image
- **Tampering Check**: ~3-5 seconds per image
- **Description Matching**: ~2-3 seconds per image
- **Total**: ~10-18 seconds per image

## Security & Privacy

- All data stored locally in `fraud_detection_data/`
- Images sent to Gemini API for description matching only
- No data shared with third parties
- Submissions organized by customer and timestamp

## Limitations

- AI detection accuracy: ~85-90%
- Tampering detection accuracy: ~87%
- Description matching depends on Gemini API availability
- Weather validation requires EXIF metadata in images
- Large batch processing may be slow

## Future Enhancements

- [ ] Batch processing optimization
- [ ] Database integration
- [ ] PDF report generation
- [ ] Email notifications
- [ ] Multi-language support
- [ ] Mobile app interface

## Credits

- ELA Detection: Based on CASIA2.0 dataset training
- Weather CNN: Custom trained model
- AI Detection: Hugging Face transformers
- Description Matching: Google Gemini AI

## License

For educational and research purposes.

## Support

For issues or questions, check the console output for detailed error messages.

---

**Version**: 1.0.0  
**Last Updated**: October 2024
