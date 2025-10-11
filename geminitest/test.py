from dotenv import load_dotenv
import os
from main import CarDamageDetector

load_dotenv()
detector = CarDamageDetector(os.getenv("GEMINI_API_KEY"))

# Test with your image
result = detector.verify_image_description(
    "Car damages 101.png",  # Change this
    "dent on the rear bumper of the car"        # Change this
)

print(result)