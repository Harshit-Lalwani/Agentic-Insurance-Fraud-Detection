import google.generativeai as genai
from PIL import Image
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class CarDamageDetector:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the Car Damage Detector with Gemini API.
        
        Args:
            api_key: Your Gemini API key
            model_name: Gemini model to use (default: gemini-2.0-flash)
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def verify_image_description(self, 
                                 image_path: str, 
                                 description: str,
                                 include_reasoning: bool = True) -> Dict:
        """
        Verify if an image matches the given description(specifically car parts).
        Be very rigorous in terms of matching the description
        Act like an insurance company assessor and verify only if the image exactly or almost exactly matches the description
        Args:
            image_path: Path to the car part image
            description: Description to verify (e.g., "damaged front bumper")
            include_reasoning: Whether to include detailed reasoning
            
        Returns:
            Dictionary with verification results
        """

        # Removed inappropriate comment
        try:
            # Load image
            img = Image.open(image_path)


            # Build prompt text
            prompt_text = (
                "AI-Powered Multimodal Claims Assessment\n"
                "You are a highly-skilled and meticulous automotive insurance claims assessor. Your task is to analyze an image and a corresponding text description to determine if they match, focusing on specific car parts and visible damage. You must adhere to a strict, rigorous process, providing a detailed assessment.\n"
                "Task: Analyze the provided image and description to verify if the damage depicted in the image accurately matches the damage described. Your response must be objective and data-driven.\n"
                f"Description: {description}\n"
                "Provide a JSON response with the following keys. Do not deviate from this format.\n"
                "1. MATCH: A single value from the set ['Exact Match', 'Partial Match', 'No Match'].\n"
                "2. CONFIDENCE: A numeric value from 0.0 to 1.0 representing your confidence in the assessment.\n"
                "3. CAR_PART: Identify the specific car part shown in the image.\n"
                "4. DAMAGE_STATUS: A single value from the set ['Damaged', 'Not Damaged', 'Unclear'].\n"
                "5. DAMAGE_TYPE: A list of all types of damage visible in the image (e.g., 'dents', 'scratches', 'cracks').\n"
                "6. REASONING: A concise, detailed explanation of your assessment, referencing specific visual evidence from the image and how it aligns or conflicts with the provided description.\n"
                "7. SEVERITY: A single value from the set ['Low', 'Medium', 'High'] based on the damage observed in the image."
            )

            # Gemini expects a list of parts: text and image
            prompt_parts = [
                {"text": prompt_text},
                img
            ]

            # Generate response
            response = self.model.generate_content(prompt_parts)

            # Parse response
            result = self._parse_response(response.text)
            result['image_path'] = image_path
            result['description'] = description
            result['raw_response'] = response.text

            return result

        except Exception as e:
            return {
                'error': str(e),
                'image_path': image_path,
                'description': description
            }
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse Gemini's response into structured format."""
        try:
            # Try to extract JSON from response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
            else:
                # Fallback: create structured response from text
                return {
                    'match': 'Unclear',
                    'confidence': 0,
                    'car_part': 'Unknown',
                    'damage_status': 'Unclear',
                    'damage_type': 'Unknown',
                    'reasoning': response_text
                }
        except json.JSONDecodeError:
            return {
                'match': 'Unclear',
                'confidence': 0,
                'car_part': 'Unknown',
                'damage_status': 'Unclear',
                'damage_type': 'Unknown',
                'reasoning': response_text
            }
    
    def batch_verify(self, 
                     image_description_pairs: List[Tuple[str, str]],
                     delay: float = 1.0) -> List[Dict]:
        """
        Verify multiple image-description pairs.
        
        Args:
            image_description_pairs: List of (image_path, description) tuples
            delay: Delay between API calls to respect rate limits
            
        Returns:
            List of verification results
        """
        results = []
        
        for i, (image_path, description) in enumerate(image_description_pairs):
            print(f"Processing {i+1}/{len(image_description_pairs)}: {image_path}")
            
            result = self.verify_image_description(image_path, description)
            results.append(result)
            
            # Respect rate limits
            if i < len(image_description_pairs) - 1:
                time.sleep(delay)
        
        return results
    
    def generate_report(self, results: List[Dict], output_file: str = None) -> str:
        """
        Generate a summary report from verification results.
        
        Args:
            results: List of verification results
            output_file: Optional file path to save report
            
        Returns:
            Report as string
        """
        total = len(results)
        matches = sum(1 for r in results if r.get('match', '').lower() == 'yes')
        errors = sum(1 for r in results if 'error' in r)
        
        avg_confidence = sum(r.get('confidence', 0) for r in results if 'confidence' in r) / max(total - errors, 1)
        
        report = f"""
=== CAR DAMAGE DETECTION REPORT ===
Total Images Processed: {total}
Successful Matches: {matches} ({matches/total*100:.1f}%)
Errors: {errors}
Average Confidence: {avg_confidence:.1f}%

=== DETAILED RESULTS ===
"""
        
        for i, result in enumerate(results, 1):
            if 'error' in result:
                report += f"\n{i}. ERROR: {result['error']}\n"
            else:
                report += f"""
{i}. {result.get('image_path', 'Unknown')}
   Description: {result.get('description', 'N/A')}
   Match: {result.get('match', 'N/A')}
   Confidence: {result.get('confidence', 0)}%
   Car Part: {result.get('car_part', 'N/A')}
   Damage: {result.get('damage_status', 'N/A')} - {result.get('damage_type', 'N/A')}
   Reasoning: {result.get('reasoning', 'N/A')[:100]}...
"""
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
                f.write("\n\n=== RAW JSON RESULTS ===\n")
                json.dump(results, f, indent=2)
        
        return report


# Example usage
if __name__ == "__main__":
    # Load API key from environment variable
    API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
    
    # Initialize detector
    detector = CarDamageDetector(API_KEY)
    
    # Example 1: Single image verification
    print("=== Single Image Verification ===")
    result = detector.verify_image_description(
        image_path="path/to/car_image.jpg",
        description="damaged front bumper with scratch"
    )
    print(json.dumps(result, indent=2))
    
    # Example 2: Batch verification
    print("\n=== Batch Verification ===")
    test_pairs = [
        ("path/to/bumper_damage.jpg", "damaged front bumper"),
        ("path/to/door_scratch.jpg", "scratched door panel"),
        ("path/to/headlight_broken.jpg", "broken headlight"),
    ]
    
    results = detector.batch_verify(test_pairs, delay=1.0)
    
    # Generate report
    report = detector.generate_report(results, output_file="damage_report.txt")
    print(report)
    
    # Example 3: Working with Kaggle dataset structure
    print("\n=== Kaggle Dataset Example ===")
    # Assuming dataset structure: images/ folder with car part images
    dataset_path = "path/to/car-parts-and-car-damages"
    
    # You would typically parse the annotations file that comes with the dataset
    # For demonstration, here's how you'd structure it:
    dataset_pairs = [
        (f"{dataset_path}/images/image1.jpg", "damaged bumper"),
        (f"{dataset_path}/images/image2.jpg", "intact door"),
        # Add more pairs based on your dataset annotations
    ]
    
    # Process dataset
    # results = detector.batch_verify(dataset_pairs, delay=1.0)
    # detector.generate_report(results, "kaggle_dataset_results.txt")