"""
Image-Description Matching Module
Validates if image matches the provided description using NVIDIA NIM VLM
"""

import os
import json
import base64
import io
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv


class DescriptionMatcher:
    # Car parts and damage types mappings
    CAR_PARTS = {
        "Quarter-panel": 1, "Front-wheel": 2, "Back-window": 3, "Trunk": 4,
        "Front-door": 5, "Rocker-panel": 6, "Grille": 7, "Windshield": 8,
        "Front-window": 9, "Back-door": 10, "Headlight": 11, "Back-wheel": 12,
        "Back-windshield": 13, "Hood": 14, "Fender": 15, "Tail-light": 16,
        "License-plate": 17, "Front-bumper": 18, "Back-bumper": 19,
        "Mirror": 20, "Roof": 21
    }

    DAMAGE_TYPES = {
        "No-damage": 0, "Dent": 1, "Scratch": 2, "Broken part": 3,
        "Paint chip": 4, "Missing part": 5, "Flaking": 6,
        "Corrosion": 7, "Cracked": 8
    }

    def __init__(self, api_key=None, model_name=None, base_url=None):
        """Initialize NVIDIA NIM VLM for image-description matching"""
        load_dotenv()

        if api_key is None:
            api_key = os.getenv("NVIDIA_API_KEY")
        if model_name is None:
            model_name = os.getenv("NVIDIA_NIM_MODEL", "meta/llama-3.2-11b-vision-instruct")
        if base_url is None:
            base_url = os.getenv("NVIDIA_NIM_BASE_URL", "https://api.build.nvidia.com/v1")

        if not api_key:
            raise ValueError("NVIDIA_API_KEY not found. Please set it in .env file or pass as argument.")

        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        print(f"Description matching model initialized using {self.model_name}!\n")

    def _parse_response(self, response_text):
        """Parse JSON response from the VLM"""
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1

            if start != -1 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
            else:
                return {
                    'MATCH': 'Unclear',
                    'CONFIDENCE': 0,
                    'CAR_PART': 'Unknown',
                    'DAMAGE_STATUS': 'Unclear',
                    'REASONING': response_text
                }
        except json.JSONDecodeError:
            return {
                'MATCH': 'Unclear',
                'CONFIDENCE': 0,
                'CAR_PART': 'Unknown',
                'DAMAGE_STATUS': 'Unclear',
                'REASONING': response_text
            }

    def _image_to_base64(self, img):
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def verify(self, image_path, description):
        """
        Verify if image matches the description

        Args:
            image_path: Path to the car damage image
            description: Text description of the damage

        Returns:
            dict: {
                'matches': bool,
                'match_type': str,
                'confidence': float,
                'car_part': str,
                'damage_status': str,
                'damage_type': str,
                'reasoning': str,
                'severity': str
            }
        """
        result = {
            'matches': False,
            'match_type': 'No Match',
            'confidence': 0.0,
            'car_part': 'Unknown',
            'damage_status': 'Unknown',
            'damage_type': 'Unknown',
            'reasoning': '',
            'severity': 'Unknown',
            'manual_check_required': False
        }

        if not os.path.exists(image_path):
            result['reasoning'] = f"Image not found: {image_path}"
            return result

        try:
            # Load image
            img = Image.open(image_path)

            # Build prompt
            car_parts_list = "\n".join([f"   {name}: {idx}" for name, idx in self.CAR_PARTS.items()])
            damage_types_list = "\n".join([f"   {name}: {idx}" for name, idx in self.DAMAGE_TYPES.items()])

            prompt_text = (
                "AI-Powered Multimodal Claims Assessment\n"
                "You are a highly-skilled automotive insurance claims assessor. Analyze the image and description to determine if they match.\n\n"
                f"Description: {description}\n\n"
                "CAR PARTS MAPPING:\n"
                f"{car_parts_list}\n\n"
                "DAMAGE TYPES MAPPING:\n"
                f"{damage_types_list}\n\n"
                "Provide a JSON response with these keys:\n"
                "1. MATCH: ['Exact Match', 'Partial Match', 'No Match']\n"
                "2. CONFIDENCE: Numeric value from 0.0 to 1.0\n"
                "3. CAR_PART: Specific car part from the mapping\n"
                "4. CAR_PART_ID: Numeric ID (1-21)\n"
                "5. DAMAGE_STATUS: ['Damaged', 'Not Damaged', 'Unclear']\n"
                "6. DAMAGE_TYPE: Type of damage from the mapping\n"
                "7. DAMAGE_TYPE_ID: Numeric ID (0-8)\n"
                "8. REASONING: Detailed explanation\n"
                "9. SEVERITY: ['Low', 'Medium', 'High']"
            )

            # Convert image to base64
            img_base64 = self._image_to_base64(img)

            # Generate response via NVIDIA NIM
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1024
            )

            response_text = response.choices[0].message.content

            # Parse response
            parsed = self._parse_response(response_text)

            # Map to result format
            match_type = parsed.get('MATCH', 'No Match')

            # Handle if match_type is a list
            if isinstance(match_type, list):
                match_type = match_type[0] if match_type else 'No Match'

            result['match_type'] = match_type

            # Check if it's an exact match
            is_exact_match = 'exact' in str(match_type).lower()
            is_partial_match = 'partial' in str(match_type).lower()

            # Check if it's a match (Exact or Partial)
            result['matches'] = is_exact_match or is_partial_match

            # Flag for manual check if partial match
            result['manual_check_required'] = is_partial_match

            result['confidence'] = parsed.get('CONFIDENCE', 0)

            # Handle car_part (could be list or string)
            car_part = parsed.get('CAR_PART', 'Unknown')
            result['car_part'] = car_part[0] if isinstance(car_part, list) else car_part

            # Handle damage_status (could be list or string)
            damage_status = parsed.get('DAMAGE_STATUS', 'Unknown')
            result['damage_status'] = damage_status[0] if isinstance(damage_status, list) else damage_status

            # Handle damage_type (could be list or string)
            damage_type = parsed.get('DAMAGE_TYPE', 'Unknown')
            result['damage_type'] = damage_type[0] if isinstance(damage_type, list) else damage_type

            result['reasoning'] = parsed.get('REASONING', 'No reasoning provided')

            # Handle severity (could be list or string)
            severity = parsed.get('SEVERITY', 'Unknown')
            result['severity'] = severity[0] if isinstance(severity, list) else severity

            result['raw_response'] = response_text

        except Exception as e:
            result['reasoning'] = f"Error during verification: {str(e)}"

        return result


if __name__ == "__main__":
    # Test
    matcher = DescriptionMatcher()
    result = matcher.verify(
        "test_damage.jpg",
        "damaged front bumper with scratch"
    )
    print(json.dumps(result, indent=2))
