"""
AI Image Detection Module
Checks if images are AI-generated using ensemble models
"""

import os
from PIL import Image
from transformers import pipeline
import torch


class AIImageDetector:
    MODEL_1 = "Organika/sdxl-detector"
    MODEL_2 = "umm-maybe/AI-image-detector"
    
    def __init__(self):
        """Initialize AI detection models"""
        self.detector1 = None
        self.detector2 = None
        self.models_loaded = {"model1": False, "model2": False}
        self._load_models()
    
    def _load_models(self):
        """Load AI detection models with error handling"""
        print("Loading AI detection models...")
        
        # Try loading Model 1
        try:
            print(f"  Loading Model 1 ({self.MODEL_1})...")
            self.detector1 = pipeline("image-classification", model=self.MODEL_1, trust_remote_code=True)
            self.models_loaded["model1"] = True
            print("  Model 1 loaded successfully!")
        except Exception as e:
            print(f"  Error loading Model 1: {e}")
            self.detector1 = None

        # Try loading Model 2
        try:
            print(f"  Loading Model 2 ({self.MODEL_2})...")
            self.detector2 = pipeline("image-classification", model=self.MODEL_2, trust_remote_code=True)
            self.models_loaded["model2"] = True
            print("  Model 2 loaded successfully!")
        except Exception as e:
            print(f"  Error loading Model 2: {e}")
            # Try fallback model
            try:
                print("  Trying fallback model...")
                self.detector2 = pipeline("image-classification", model="microsoft/resnet-50")
                self.models_loaded["model2"] = True
                print("  Fallback model loaded successfully!")
            except Exception as e2:
                print(f"  Fallback model also failed: {e2}")
                self.detector2 = None

        if not any(self.models_loaded.values()):
            raise Exception("CRITICAL: No AI detection models loaded")
        
        print("AI detection models initialized!\n")
    
    def _normalize_prediction(self, predictions, model_name):
        """Extract AI-generated probability from model predictions"""
        pred_dict = {pred["label"].lower(): pred["score"] for pred in predictions}

        if "organika" in model_name.lower():
            ai_score = pred_dict.get('ai', pred_dict.get('artificial', 0))
        elif "umm-maybe" in model_name.lower():
            ai_score = pred_dict.get('artificial', 0)
        elif "resnet" in model_name.lower():
            ai_score = predictions[0]["score"] if predictions else 0.5
        else:
            ai_score = pred_dict.get('ai', pred_dict.get('artificial', predictions[0]["score"] if predictions else 0.5))

        label = "AI-Generated" if ai_score > 0.5 else "Human-Generated"
        return label, ai_score
    
    def detect(self, image_path):
        """
        Detect if an image is AI-generated
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: {
                'is_ai_generated': bool,
                'ai_percentage': float,
                'confidence': str,
                'verdict': str,
                'details': dict
            }
        """
        result = {
            'is_ai_generated': False,
            'ai_percentage': 0.0,
            'confidence': 'N/A',
            'verdict': 'Error',
            'details': {}
        }
        
        # Check if file exists
        if not os.path.exists(image_path):
            result['details']['error'] = f"File not found: {image_path}"
            return result
        
        # Load image
        try:
            image = Image.open(image_path)
        except Exception as e:
            result['details']['error'] = f"Failed to load image: {str(e)}"
            return result
        
        # Validate image
        if image.size[0] < 50 or image.size[1] < 50:
            result['details']['error'] = "Image too small (minimum 50x50 pixels)"
            return result

        # Check if any models are loaded
        if not any(self.models_loaded.values()):
            result['details']['error'] = "No AI detection models are available"
            return result

        predictions = []

        # Get prediction from Model 1
        if self.models_loaded["model1"] and self.detector1:
            try:
                pred1 = self.detector1(image, top_k=2)
                label1, score1 = self._normalize_prediction(pred1, self.MODEL_1)
                predictions.append(("Model 1", label1, score1))
                result['details']['model1_label'] = label1
                result['details']['model1_ai_score'] = f"{score1:.2%}"
            except Exception as e:
                result['details']['model1_error'] = str(e)

        # Get prediction from Model 2
        if self.models_loaded["model2"] and self.detector2:
            try:
                pred2 = self.detector2(image, top_k=2)
                label2, score2 = self._normalize_prediction(pred2, self.MODEL_2)
                predictions.append(("Model 2", label2, score2))
                result['details']['model2_label'] = label2
                result['details']['model2_ai_score'] = f"{score2:.2%}"
            except Exception as e:
                result['details']['model2_error'] = str(e)

        # Determine Final Verdict
        if len(predictions) == 2:
            # Both models available - ensemble prediction
            label1, score1 = predictions[0][1], predictions[0][2]
            label2, score2 = predictions[1][1], predictions[1][2]

            # Average the AI confidence scores
            avg_score = (score1 + score2) / 2
            result['ai_percentage'] = avg_score * 100

            # Decision boundary: 0.3 is the tipping point
            final_verdict = "AI-Generated" if avg_score > 0.3 else "Human-Generated"
            
            if final_verdict == "AI-Generated":
                result['is_ai_generated'] = True
                if avg_score > 0.8:
                    confidence = "High"
                elif avg_score > 0.6:
                    confidence = "Medium"
                else:
                    confidence = "Low"
            else:
                if avg_score < 0.2:
                    confidence = "High"
                else:
                    confidence = "Medium"

            result['verdict'] = final_verdict
            result['confidence'] = confidence
            result['details']['agreement'] = "Yes" if label1 == label2 else "No"

        elif len(predictions) == 1:
            # Only one model available
            model_name, label, score = predictions[0][0], predictions[0][1], predictions[0][2]
            result['ai_percentage'] = score * 100
            result['verdict'] = label
            result['is_ai_generated'] = (label == "AI-Generated")
            
            if score > 0.7 or score < 0.3:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            result['confidence'] = confidence
            result['details']['note'] = "Only one model available"

        else:
            result['details']['error'] = "Both models failed to process the image"

        return result


if __name__ == "__main__":
    # Test
    detector = AIImageDetector()
    result = detector.detect("test_image.jpg")
    print(result)
