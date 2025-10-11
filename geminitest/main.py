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
    # Mapping dictionaries for car parts and damage types
    CAR_PARTS = {
        "Quarter-panel": 1,
        "Front-wheel": 2,
        "Back-window": 3,
        "Trunk": 4,
        "Front-door": 5,
        "Rocker-panel": 6,
        "Grille": 7,
        "Windshield": 8,
        "Front-window": 9,
        "Back-door": 10,
        "Headlight": 11,
        "Back-wheel": 12,
        "Back-windshield": 13,
        "Hood": 14,
        "Fender": 15,
        "Tail-light": 16,
        "License-plate": 17,
        "Front-bumper": 18,
        "Back-bumper": 19,
        "Mirror": 20,
        "Roof": 21
    }

    DAMAGE_TYPES = {
        "No-damage": 0,
        "Dent": 1,
        "Scratch": 2,
        "Broken part": 3,
        "Paint chip": 4,
        "Missing part": 5,
        "Flaking": 6,
        "Corrosion": 7,
        "Cracked": 8
    }

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
            car_parts_list = "\n".join([f"   {name}: {idx}" for name, idx in self.CAR_PARTS.items()])
            damage_types_list = "\n".join([f"   {name}: {idx}" for name, idx in self.DAMAGE_TYPES.items()])
            
            prompt_text = (
                "AI-Powered Multimodal Claims Assessment\n"
                "You are a highly-skilled and meticulous automotive insurance claims assessor. Your task is to analyze an image and a corresponding text description to determine if they match, focusing on specific car parts and visible damage. You must adhere to a strict, rigorous process, providing a detailed assessment.\n\n"
                "Task: Analyze the provided image and description to verify if the damage depicted in the image accurately matches the damage described. Your response must be objective and data-driven.\n\n"
                f"Description: {description}\n\n"
                "CAR PARTS MAPPING:\n"
                f"{car_parts_list}\n\n"
                "DAMAGE TYPES MAPPING:\n"
                f"{damage_types_list}\n\n"
                "Provide a JSON response with the following keys. Do not deviate from this format.\n"
                "1. MATCH: A single value from the set ['Exact Match', 'Partial Match', 'No Match'].\n"
                "2. CONFIDENCE: A numeric value from 0.0 to 1.0 representing your confidence in the assessment.\n"
                "3. CAR_PART: Identify the specific car part shown in the image (use exact names from the CAR PARTS MAPPING above).\n"
                "4. CAR_PART_ID: The numeric ID of the car part from the mapping (1-21).\n"
                "5. DAMAGE_STATUS: A single value from the set ['Damaged', 'Not Damaged', 'Unclear'].\n"
                "6. DAMAGE_TYPE: The primary type of damage visible (use exact names from the DAMAGE TYPES MAPPING above).\n"
                "7. DAMAGE_TYPE_ID: The numeric ID of the damage type from the mapping (0-8).\n"
                "8. REASONING: A concise, detailed explanation of your assessment, referencing specific visual evidence from the image and how it aligns or conflicts with the provided description.\n"
                "9. SEVERITY: A single value from the set ['Low', 'Medium', 'High'] based on the damage observed in the image.\n"
                "10. DAMAGE_VECTOR: A 21-element array where each index (0-20) corresponds to a car part (1-21), and the value is the damage type ID (0-8). Set the appropriate index to the damage type ID for the identified car part, and 0 for all others."
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
            
            # Add formatted output
            result['formatted_output'] = self._format_output(result)
            result['damage_vector'] = self._create_damage_vector(result)

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
    
    def _format_output(self, result: Dict) -> str:
        """Format the result with proper newlines for readability."""
        # Handle both uppercase and lowercase keys
        match = result.get('MATCH') or result.get('match', 'N/A')
        confidence = result.get('CONFIDENCE') or result.get('confidence', 0)
        car_part = result.get('CAR_PART') or result.get('car_part', 'Unknown')
        damage_status = result.get('DAMAGE_STATUS') or result.get('damage_status', 'N/A')
        damage_type = result.get('DAMAGE_TYPE') or result.get('damage_type', [])
        reasoning = result.get('REASONING') or result.get('reasoning', 'N/A')
        severity = result.get('SEVERITY') or result.get('severity', 'N/A')
        
        # Format damage types
        if isinstance(damage_type, list):
            damage_type_str = ', '.join(damage_type)
        else:
            damage_type_str = str(damage_type)
        
        formatted = f"""
{'='*60}
CAR DAMAGE ANALYSIS REPORT
{'='*60}

IMAGE: {result.get('image_path', 'N/A')}
DESCRIPTION: {result.get('description', 'N/A')}

{'-'*60}
ANALYSIS RESULTS
{'-'*60}

MATCH STATUS:     {match}
CONFIDENCE:       {confidence:.2f}
CAR PART:         {car_part}
DAMAGE STATUS:    {damage_status}
DAMAGE TYPE(S):   {damage_type_str}
SEVERITY:         {severity}

{'-'*60}
REASONING
{'-'*60}
{reasoning}

{'='*60}
"""
        return formatted
    
    def _create_damage_vector(self, result: Dict) -> List[int]:
        """Create a vector mapping each car part to its damage type.
        Returns a 21-element vector where each index corresponds to a car part
        and the value is the damage type ID (0-8)."""
        # Initialize vector with 0 (no damage) for all parts
        vector = [0] * 21
        
        # Get car part and damage type from result (handle both cases)
        car_part = result.get('CAR_PART') or result.get('car_part', '')
        damage_type = result.get('DAMAGE_TYPE') or result.get('damage_type', [])
        
        # Normalize damage type
        if isinstance(damage_type, list):
            damage_type = damage_type[0] if damage_type else 'No-damage'
        
        # Find matching car part
        car_part_id = None
        for part_name, part_id in self.CAR_PARTS.items():
            if part_name.lower() in str(car_part).lower():
                car_part_id = part_id
                break
        
        # Find matching damage type
        damage_type_id = 0  # Default to no damage
        for damage_name, damage_id in self.DAMAGE_TYPES.items():
            if damage_name.lower() in str(damage_type).lower():
                damage_type_id = damage_id
                break
        
        # Set the damage type for the identified car part
        if car_part_id is not None:
            vector[car_part_id - 1] = damage_type_id
        
        return vector
    
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