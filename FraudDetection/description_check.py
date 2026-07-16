"""
Image-Description Matching Module
Validates if image matches the provided description using a VLM.
Primary: Gemini (gemini-3.5-flash). Fallback: NVIDIA NIM (Llama 3.2 Vision).
"""

import os
import json
import base64
import io
from PIL import Image
from openai import OpenAI
from google import genai
from google.genai import types as genai_types
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

    def __init__(self, api_key=None, model_name=None, base_url=None,
                 google_api_key=None, gemini_model_name=None):
        """Initialize VLM(s) for image-description matching.

        Gemini is the primary VLM; NVIDIA NIM is kept configured as a fallback
        used automatically if a Gemini call fails.
        """
        load_dotenv()

        # --- Gemini (primary) ---
        if google_api_key is None:
            google_api_key = os.getenv("GOOGLE_API_KEY")
        if gemini_model_name is None:
            gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

        self.gemini_model_name = gemini_model_name
        self.gemini_client = genai.Client(api_key=google_api_key) if google_api_key else None

        # --- NVIDIA NIM (fallback) ---
        if api_key is None:
            api_key = os.getenv("NVIDIA_API_KEY")
        if model_name is None:
            model_name = os.getenv("NVIDIA_NIM_MODEL", "meta/llama-3.2-11b-vision-instruct")
        if base_url is None:
            base_url = os.getenv("NVIDIA_NIM_BASE_URL", "https://api.build.nvidia.com/v1")

        self.model_name = model_name
        self.nim_client = OpenAI(base_url=base_url, api_key=api_key) if api_key else None

        if not self.gemini_client and not self.nim_client:
            raise ValueError(
                "Neither GOOGLE_API_KEY nor NVIDIA_API_KEY found. "
                "Please set at least one in .env file or pass as argument."
            )

        if self.gemini_client:
            print(f"Description matching model initialized using Gemini ({self.gemini_model_name}), "
                  f"with NVIDIA NIM ({self.model_name}) as fallback!\n" if self.nim_client
                  else f"Description matching model initialized using Gemini ({self.gemini_model_name})!\n")
        else:
            print(f"GOOGLE_API_KEY not set — description matching model initialized using "
                  f"NVIDIA NIM ({self.model_name}) only!\n")

    def _parse_response(self, response_text):
        """Parse VLM response — tries JSON first, then falls back to markdown key-value format."""
        import re

        # --- Attempt 1: JSON block ---
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response_text[start:end])
        except json.JSONDecodeError:
            pass

        # --- Attempt 2: **KEY:** Value markdown format ---
        # The model sometimes returns fields on a single line or with markdown formatting
        def extract(key):
            # Known keys to use as boundaries for lazy matching
            keys_pattern = r'MATCH|CONFIDENCE|CAR_PART|CAR_PART_ID|DAMAGE_STATUS|DAMAGE_TYPE|DAMAGE_TYPE_ID|REASONING|SEVERITY|CAR PART|DAMAGE STATUS|DAMAGE TYPE'
            # Match the key, optional asterisks, colon/space. 
            # Capture lazily until the next known key or end of string.
            pattern = rf'\*{{0,2}}{re.escape(key)}\*{{0,2}}[:\s]+(.*?)(?=\s*\*{{0,2}}(?:{keys_pattern})\*{{0,2}}[:\s]|$)'
            
            m = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else None

        match_val = extract('MATCH')
        confidence_val = extract('CONFIDENCE')
        car_part_val = extract('CAR_PART') or extract('CAR PART')
        damage_status_val = extract('DAMAGE_STATUS') or extract('DAMAGE STATUS')
        damage_type_val = extract('DAMAGE_TYPE') or extract('DAMAGE TYPE')
        reasoning_val = extract('REASONING')
        severity_val = extract('SEVERITY')

        if any([match_val, car_part_val, damage_status_val]):
            parsed = {
                'MATCH': match_val or 'Unclear',
                'CONFIDENCE': 0.0,
                'CAR_PART': car_part_val or 'Unknown',
                'DAMAGE_STATUS': damage_status_val or 'Unclear',
                'DAMAGE_TYPE': damage_type_val or 'Unknown',
                'REASONING': reasoning_val or response_text,
                'SEVERITY': severity_val or 'Unknown',
            }
            try:
                parsed['CONFIDENCE'] = float(confidence_val) if confidence_val else 0.0
            except (ValueError, TypeError):
                parsed['CONFIDENCE'] = 0.0
            return parsed

        # --- Fallback: return raw text as reasoning ---
        return {
            'MATCH': 'Unclear',
            'CONFIDENCE': 0,
            'CAR_PART': 'Unknown',
            'DAMAGE_STATUS': 'Unclear',
            'REASONING': response_text,
        }


    def _image_to_base64(self, img):
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _call_gemini(self, prompt_text, img):
        """Generate a response via Gemini (primary VLM)"""
        response = self.gemini_client.models.generate_content(
            model=self.gemini_model_name,
            contents=[prompt_text, img],
        )
        return response.text

    def _call_nim(self, prompt_text, img_base64):
        """Generate a response via NVIDIA NIM (fallback VLM)"""
        response = self.nim_client.chat.completions.create(
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
        return response.choices[0].message.content

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

            response_text = None
            used_fallback = False

            # Try Gemini first
            if self.gemini_client:
                try:
                    response_text = self._call_gemini(prompt_text, img)
                except Exception as gemini_error:
                    if self.nim_client:
                        used_fallback = True
                    else:
                        raise gemini_error

            # Fall back to NVIDIA NIM if Gemini is unavailable or failed
            if response_text is None and self.nim_client:
                img_base64 = self._image_to_base64(img)
                response_text = self._call_nim(prompt_text, img_base64)

            if used_fallback:
                result['used_fallback'] = True

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
