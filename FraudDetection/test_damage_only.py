"""
Quick test script for damage detection only
"""
import os
from combined_damage_detector import CombinedDamageDetector

# Initialize combined detector
print("Initializing Combined Damage Detector...")
detector = CombinedDamageDetector(
    parts_model_path="../part_detection_model.pth",
    damage_model_path="../damage_model.pth",
    confidence_threshold=0.5,
    verbose=True
)

# Test with an image
test_image = "../geminitest/Car damages 100.png"

if os.path.exists(test_image):
    print(f"\nTesting with image: {test_image}")
    result = detector.detect_damage_and_parts(test_image)
    
    # Generate report
    report = detector.generate_report(result)
    print("\n" + report)
    
    # Save visualization
    output_dir = "test_damage_output"
    os.makedirs(output_dir, exist_ok=True)
    vis_path = os.path.join(output_dir, "test_damage_analysis.jpg")
    detector.save_visualization(test_image, result, vis_path)
    print(f"\nVisualization saved to: {vis_path}")
else:
    print(f"Test image not found: {test_image}")
