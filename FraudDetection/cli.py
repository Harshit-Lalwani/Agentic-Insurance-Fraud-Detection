import argparse
import os
import sys
import json
from pathlib import Path
import numpy as np

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def print_result(result, title):
    print("=" * 60)
    print(f"[{title}]")
    print("=" * 60)
    serializable_result = convert_to_serializable(result)
    print(json.dumps(serializable_result, indent=2))
    print()

def main():
    parser = argparse.ArgumentParser(description="Automotive Fraud Detection CLI")
    parser.add_argument("--step", choices=["all", "ai", "tampering", "description", "duplication", "damage"], 
                        default="all", help="Which step to run")
    parser.add_argument("--image", nargs="+", required=True, help="Path(s) to image(s)")
    parser.add_argument("--description", help="Description of the image(s), separated by commas or newlines")
    parser.add_argument("--customer", default="CLI User", help="Customer name (for 'all' step)")
    parser.add_argument("--car", default="CLI Car", help="Car details (for 'all' step)")
    
    args = parser.parse_args()

    # Verify images exist
    for img_path in args.image:
        if not os.path.exists(img_path):
            print(f"Error: Image not found at {img_path}")
            sys.exit(1)

    if args.step == "all":
        # Simulate Gradio File object since input.py expects it
        class DummyFile:
            def __init__(self, path):
                self.name = path
        
        images = [DummyFile(p) for p in args.image]
        
        from input import pipeline
        report = pipeline.process_submission(
            images, 
            args.description if args.description else "Car damage", 
            args.customer, 
            args.car
        )
        print(report)
        return

    # For individual steps, we just take the first image if multiple are provided
    image_path = args.image[0]
    desc = args.description if args.description else "Car damage"

    if args.step == "ai":
        from ai_detector import AIImageDetector
        detector = AIImageDetector()
        result = detector.detect(image_path)
        print_result(result, "AI GENERATION CHECK")
        
    elif args.step == "tampering":
        from tampering_check import TamperingDetector
        script_dir = os.path.dirname(os.path.abspath(__file__))
        detector = TamperingDetector(
            ela_model_path=os.path.join(script_dir, "Image-Tampering-Detection-using-ELA-and-Metadata-Analysis/ELA_Training/model_ela.h5"),
            weather_model_path=os.path.join(script_dir, "Image-Tampering-Detection-using-ELA-and-Metadata-Analysis/WeatherCNNTraining/Weather_Model.h5")
        )
        result = detector.detect(image_path)
        print_result(result, "TAMPERING CHECK")
        
    elif args.step == "description":
        from description_check import DescriptionMatcher
        detector = DescriptionMatcher()
        result = detector.verify(image_path, desc)
        print_result(result, "DESCRIPTION MATCHING")
        
    elif args.step == "duplication":
        from duplication_check import DuplicationDetector
        detector = DuplicationDetector()
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fraud_detection_data")
        os.makedirs(base_dir, exist_ok=True)
        result = detector.check_for_duplicates(image_path, base_dir)
        print_result(result, "DUPLICATION CHECK")
        
    elif args.step == "damage":
        from combined_damage_detector import CombinedDamageDetector
        script_dir = os.path.dirname(os.path.abspath(__file__))
        detector = CombinedDamageDetector(
            parts_model_path=os.path.join(script_dir, "../damage-det/model_parts.pth"),
            damage_model_path=os.path.join(script_dir, "../damage-det/model_damage.pth"),
            confidence_threshold=0.7,
            verbose=True
        )
        result = detector.detect_damage_and_parts(image_path)
        print_result(result, "DAMAGE ANALYSIS")

if __name__ == "__main__":
    main()
