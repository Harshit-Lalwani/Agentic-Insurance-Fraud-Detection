"""
Automotive Fraud Detection System
Main entry point with Gradio web interface
"""

import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path
import gradio as gr
import numpy as np

# Enforce CUDA-capable GPU before anything else.
# CPU fallback is intentionally disabled: the pipeline runs multiple heavy
# models (Detectron2, HuggingFace, CLIP) and CPU inference is far too slow
# for an interactive Gradio app.
try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA-capable GPU not detected.\n"
            "This application requires a CUDA-capable NVIDIA GPU to run.\n"
            "Install a PyTorch build with CUDA support:\n"
            "  pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124\n"
            "Also ensure NVIDIA drivers are installed (run `nvidia-smi` to verify)."
        )
    print(f"[GPU CHECK] CUDA available: {torch.cuda.get_device_name(0)}")
except RuntimeError as e:
    print("=" * 60)
    print("FATAL: GPU requirement not satisfied")
    print("=" * 60)
    print(str(e))
    sys.exit(1)

# Import detection modules
from ai_detector import AIImageDetector
from tampering_check import TamperingDetector
from description_check import DescriptionMatcher
from duplication_check import DuplicationDetector
from combined_damage_detector import CombinedDamageDetector


class FraudDetectionPipeline:
    def __init__(self):
        """Initialize all detection modules"""
        print("=" * 60)
        print("INITIALIZING FRAUD DETECTION SYSTEM")
        print("=" * 60)
        
        # Initialize detectors
        try:
            self.ai_detector = AIImageDetector()
        except Exception as e:
            print(f"Warning: AI Detector initialization failed: {e}")
            self.ai_detector = None
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            self.tampering_detector = TamperingDetector(
                ela_model_path=os.path.join(script_dir, "Image-Tampering-Detection-using-ELA-and-Metadata-Analysis/ELA_Training/model_ela.h5"),
                weather_model_path=os.path.join(script_dir, "Image-Tampering-Detection-using-ELA-and-Metadata-Analysis/WeatherCNNTraining/Weather_Model.h5")
            )
        except Exception as e:
            print(f"Warning: Tampering Detector initialization failed: {e}")
            self.tampering_detector = None
        
        try:
            self.description_matcher = DescriptionMatcher()
        except Exception as e:
            print(f"Warning: Description Matcher initialization failed: {e}")
            self.description_matcher = None
        
        try:
            self.duplication_detector = DuplicationDetector()
        except Exception as e:
            print(f"Warning: Duplication Detector initialization failed: {e}")
            self.duplication_detector = None
        
        try:
            # Initialize Combined Damage Detector (Parts + Damage Classification)
            self.damage_detector = CombinedDamageDetector(
                parts_model_path=os.path.join(script_dir, "../damage-det/model_parts.pth"),
                damage_model_path=os.path.join(script_dir, "../damage-det/model_damage.pth"),
                confidence_threshold=0.7,  # Increased to 70% to reduce false positives
                verbose=True
            )
        except Exception as e:
            print(f"Warning: Damage Detector initialization failed: {e}")
            self.damage_detector = None
        
        # Create base directory for storing submissions
        self.base_dir = "fraud_detection_data"
        os.makedirs(self.base_dir, exist_ok=True)
        self._last_submission_path = None  # Track last submission for Gradio output
        self._last_results = None  # Structured results for UI rendering
        
        print("=" * 60)
        print("SYSTEM INITIALIZED SUCCESSFULLY")
        print("=" * 60)
        print()
    
    def _create_submission_folder(self, customer_name):
        """Create a unique folder for this submission"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{customer_name.replace(' ', '_')}_{timestamp}"
        submission_path = os.path.join(self.base_dir, folder_name)
        images_path = os.path.join(submission_path, "images")
        
        os.makedirs(images_path, exist_ok=True)
        
        return submission_path, images_path
    
    def _calculate_confidence_score(self, result):
        """
        Calculate overall confidence claim score (0-100) based on all validation checks
        
        Higher score = Higher confidence the claim is legitimate
        Lower score = Higher suspicion of fraud
        
        Scoring breakdown:
        - AI Check: 25 points (if passed)
        - Tampering Check: 25 points (if passed)
        - Description Match: 25 points (if passed, partial match = 15 points)
        - Duplication Check: 25 points (if passed)
        
        Returns:
            int: Confidence score from 0-100
        """
        score = 0
        
        # AI Check (25 points)
        if result.get('ai_check'):
            ai_check = result['ai_check']
            if not ai_check.get('is_ai_generated', True):
                # Not AI-generated = full points
                score += 25
            else:
                # Deduct based on AI probability
                ai_prob = ai_check.get('ai_percentage', 100)
                score += max(0, int(25 * (1 - ai_prob / 100)))
        elif result.get('ai_check') is None:
            # If check was skipped, give benefit of doubt (partial points)
            score += 15
        
        # Tampering Check (25 points)
        if result.get('tampering_check'):
            tampering = result['tampering_check']
            if not tampering.get('is_tampered', True):
                # Not tampered = full points
                score += 25
            else:
                # Deduct based on tampering score
                tamper_score = tampering.get('tampering_score', 100)
                score += max(0, int(25 * (1 - tamper_score / 100)))
        elif result.get('tampering_check') is None:
            # If check was skipped, give benefit of doubt
            score += 15
        
        # Description Match (25 points)
        if result.get('description_check'):
            desc_check = result['description_check']
            if desc_check.get('matches', False):
                match_type = desc_check.get('match_type', 'no_match')
                if match_type == 'strong_match':
                    score += 25
                elif match_type == 'partial_match':
                    score += 15
                else:
                    score += 10
            else:
                # No match = 0 points
                pass
        elif result.get('description_check') is None:
            # If check was skipped, give benefit of doubt
            score += 15
        
        # Duplication Check (25 points)
        if result.get('duplication_check'):
            dup_check = result['duplication_check']
            if not dup_check.get('is_duplicate', True):
                # Not a duplicate = full points
                score += 25
            else:
                # Is duplicate = 0 points
                pass
        elif result.get('duplication_check') is None:
            # If check was skipped, give benefit of doubt
            score += 15
        
        # Ensure score is within 0-100 range
        return max(0, min(100, score))
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
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
    
    def _save_metadata(self, submission_path, customer_name, car_details, descriptions, results):
        """Save submission metadata to JSON file"""
        # Convert all numpy types to native Python types
        serializable_results = self._convert_to_serializable(results)
        
        metadata = {
            "customer_name": customer_name,
            "car_details": car_details,
            "submission_date": datetime.now().isoformat(),
            "descriptions": descriptions,
            "results": serializable_results
        }
        
        metadata_path = os.path.join(submission_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path
    
    def process_submission(self, images, descriptions, customer_name, car_details):
        """
        Process a fraud detection submission
        
        Args:
            images: List of uploaded image files
            descriptions: List of descriptions (one per image)
            customer_name: Name of the customer
            car_details: Car details string
            
        Returns:
            str: Formatted results report
        """
        if not images or len(images) == 0:
            return "ERROR: No images uploaded!"
        
        if not customer_name or not customer_name.strip():
            return "ERROR: Customer name is required!"
        
        if not car_details or not car_details.strip():
            return "ERROR: Car details are required!"
        
        # Parse descriptions (comma-separated or one per line)
        if isinstance(descriptions, str):
            desc_list = [d.strip() for d in descriptions.replace('\n', ',').split(',') if d.strip()]
        else:
            desc_list = descriptions
        
        # Ensure we have descriptions for all images
        if len(desc_list) < len(images):
            # Pad with generic description
            desc_list.extend(["Car damage"] * (len(images) - len(desc_list)))
        
        # Create submission folder
        submission_path, images_path = self._create_submission_folder(customer_name)
        self._last_submission_path = submission_path  # Store for Gradio output
        
        # Build report
        report = []
        report.append("=" * 70)
        report.append("AUTOMOTIVE FRAUD DETECTION REPORT")
        report.append("=" * 70)
        report.append(f"Customer Name: {customer_name}")
        report.append(f"Car Details: {car_details}")
        report.append(f"Submission Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of Images: {len(images)}")
        report.append("=" * 70)
        report.append("")
        
        overall_result = "PASSED"
        all_results = []
        
        # Process each image
        for idx, image_file in enumerate(images):
            image_num = idx + 1
            description = desc_list[idx] if idx < len(desc_list) else "Car damage"
            
            report.append(f"\n{'='*70}")
            report.append(f"IMAGE {image_num}/{len(images)}")
            report.append(f"{'='*70}")
            report.append(f"Description: {description}")
            report.append("-" * 70)
            
            # Save image to folder
            image_filename = f"image_{image_num}_{Path(image_file.name).name}"
            image_path = os.path.join(images_path, image_filename)
            shutil.copy(image_file.name, image_path)
            
            result = {
                'image': image_filename,
                'description': description,
                'ai_check': None,
                'tampering_check': None,
                'description_check': None,
                'duplication_check': None,
                'damage_analysis': None,
                'status': 'UNKNOWN',
                'confidence_score': 0  # 0-100 confidence claim score
            }
            
            # STEP 1: AI Detection
            report.append("\n[STEP 1/5] AI GENERATION CHECK")
            report.append("-" * 70)
            
            if self.ai_detector:
                ai_result = self.ai_detector.detect(image_path)
                result['ai_check'] = ai_result
                
                report.append(f"AI Generation Probability: {ai_result['ai_percentage']:.2f}%")
                report.append(f"Verdict: {ai_result['verdict']}")
                report.append(f"Confidence: {ai_result['confidence']}")
                
                # Check if AI-generated with high confidence
                if ai_result['is_ai_generated'] and ai_result['confidence'] == 'High':
                    report.append("\nRESULT: FRAUD DETECTED - Image is AI-generated")
                    report.append("STATUS: REJECTED")
                    result['status'] = 'REJECTED_AI_GENERATED'
                    overall_result = "FAILED"
                else:
                    report.append("RESULT: Passed AI check")
            else:
                report.append("AI Detector not available - SKIPPED")
            
            # STEP 2: Tampering Detection
            report.append("\n[STEP 2/5] TAMPERING CHECK")
            report.append("-" * 70)
            
            if self.tampering_detector:
                tampering_result = self.tampering_detector.detect(image_path)
                result['tampering_check'] = tampering_result
                
                report.append(f"Tampering Score: {tampering_result['tampering_score']:.2f}%")
                report.append(f"Confidence: {tampering_result['confidence']}")
                report.append(f"ELA Prediction: {tampering_result['ela_prediction']}")
                
                # Check if tampered
                if tampering_result['is_tampered'] and tampering_result['tampering_score'] > 60:
                    report.append("\nRESULT: FRAUD DETECTED - Image appears to be tampered")
                    report.append("STATUS: REJECTED")
                    result['status'] = 'REJECTED_TAMPERED'
                    overall_result = "FAILED"
                else:
                    report.append("RESULT: Passed tampering check")
            else:
                report.append("Tampering Detector not available - SKIPPED")
            
            # STEP 3: Description Matching
            report.append("\n[STEP 3/5] DESCRIPTION MATCHING")
            report.append("-" * 70)
            
            if self.description_matcher:
                desc_result = self.description_matcher.verify(image_path, description)
                result['description_check'] = desc_result
                
                report.append(f"Match Type: {desc_result['match_type']}")
                report.append(f"Confidence: {desc_result['confidence']:.2f}")
                report.append(f"Car Part: {desc_result['car_part']}")
                report.append(f"Damage Status: {desc_result['damage_status']}")
                report.append(f"Reasoning: {desc_result['reasoning'][:200]}...")
                
                # Check if manual review is needed
                if desc_result.get('manual_check_required', False):
                    report.append("\n⚠️  WARNING: Manual check is required (Partial Match)")
                
                # Check if description matches
                if not desc_result['matches']:
                    report.append("\nRESULT: FRAUD DETECTED - Image does not match description")
                    report.append("STATUS: REJECTED")
                    result['status'] = 'REJECTED_DESCRIPTION_MISMATCH'
                    overall_result = "FAILED"
                else:
                    report.append("\nRESULT: Passed description check")
            else:
                report.append("Description Matcher not available - SKIPPED")
            
            # STEP 4: Duplication Check
            report.append("\n[STEP 4/5] DUPLICATION CHECK")
            report.append("-" * 70)
            
            if self.duplication_detector:
                # Exclude current submission folder to avoid comparing with other images in same submission
                dup_result = self.duplication_detector.check_for_duplicates(
                    image_path, 
                    self.base_dir,
                    exclude_folder=submission_path
                )
                result['duplication_check'] = dup_result
                
                report.append(f"Images checked in database: {dup_result['details']['total_images_checked']}")
                
                if dup_result['is_duplicate']:
                    report.append(f"\nDUPLICATE FOUND!")
                    report.append(f"Matches: {os.path.basename(dup_result['details']['duplicate_of'])}")
                    report.append(f"Location: {os.path.dirname(dup_result['details']['duplicate_of'])}")
                    report.append(f"Similarity: {dup_result['details']['similarity_score']:.2%}")
                    report.append(f"Detection method: {dup_result['details']['method_used'].upper()}")
                    report.append("\nRESULT: FRAUD DETECTED - Duplicate image submission")
                    report.append("STATUS: REJECTED")
                    result['status'] = 'REJECTED_DUPLICATE'
                    overall_result = "FAILED"
                else:
                    report.append("No duplicates found")
                    report.append("\nRESULT: Passed duplication check")
                    result['status'] = 'PASSED'
            else:
                report.append("Duplication Detector not available - SKIPPED")
                result['status'] = 'PASSED'
            
            # STEP 5: Part Detection & Damage Classification
            report.append("\n[STEP 5/5] PART DETECTION & DAMAGE CLASSIFICATION")
            report.append("-" * 70)
            
            if self.damage_detector:
                damage_result = self.damage_detector.detect_damage_and_parts(image_path)
                result['damage_analysis'] = damage_result
                
                if 'error' not in damage_result:
                    report.append(f"Parts Detected: {damage_result['num_parts']}")
                    report.append(f"Damage Types Found: {damage_result['num_damage']}")
                    report.append(f"Overall Damage Severity: {damage_result['overall_severity']}")
                    
                    # Show damage types
                    if damage_result['damage_detected']:
                        damage_counts = {}
                        for damage_type in damage_result['damage_detected']:
                            damage_counts[damage_type] = damage_counts.get(damage_type, 0) + 1
                        
                        report.append("\nDamage Types:")
                        for damage_type, count in damage_counts.items():
                            report.append(f"  • {damage_type}: {count} instance(s)")
                    
                    # Show damaged parts
                    damaged_parts = [p for p in damage_result['damage_analysis'] if p['is_damaged']]
                    if damaged_parts:
                        report.append(f"\nDamaged Parts ({len(damaged_parts)}):")
                        for part in damaged_parts:
                            report.append(f"  [DAMAGED] {part['part_name']} - {part['severity']} severity")
                            for dmg in part['damages']:
                                report.append(f"     - {dmg['damage_type']}: {dmg['overlap_ratio']:.1%} coverage")
                    
                    # Show repair cost estimates
                    if damage_result.get('price_estimates') and damage_result.get('total_estimated_repair_cost', 0) > 0:
                        report.append(f"\nREPAIR COST ESTIMATES:")
                        report.append("-" * 70)
                        for estimate in damage_result['price_estimates']:
                            if estimate.get('estimated_repair_cost', 0) > 0:
                                report.append(f"  {estimate['part_name']}: ₹{estimate['estimated_repair_cost']:,.2f}")
                                report.append(f"     Severity: {estimate['severity']} | Base Price: ₹{estimate['base_price']:,.2f}")
                        report.append("-" * 70)
                        report.append(f"  TOTAL ESTIMATED COST: ₹{damage_result['total_estimated_repair_cost']:,.2f}")
                    
                    # Save visualizations (now returns 3 separate images)
                    vis_base = f"image_{image_num}_damage_analysis"
                    vis_path = os.path.join(images_path, vis_base)
                    vis_paths = self.damage_detector.save_visualization(image_path, damage_result, vis_path)
                    result['damage_visualizations'] = vis_paths  # Store all 3 paths
                    
                    report.append("\n✓ Damage analysis complete")
                    report.append(f"Visualizations saved: {len(vis_paths)} images")
                else:
                    report.append(f"Error: {damage_result['error']}")
            else:
                report.append("Damage Detector not available - SKIPPED")
            
            # Calculate confidence claim score (0-100)
            confidence_score = self._calculate_confidence_score(result)
            result['confidence_score'] = confidence_score
            
            report.append(f"\n{'='*70}")
            report.append(f"CONFIDENCE CLAIM SCORE: {confidence_score}/100")
            report.append(f"{'='*70}")
            
            all_results.append(result)
        
        # Final Summary
        report.append(f"\n\n{'='*70}")
        report.append("FINAL VERDICT")
        report.append("=" * 70)
        
        passed_count = sum(1 for r in all_results if r['status'] == 'PASSED')
        rejected_count = len(all_results) - passed_count
        
        report.append(f"Images Analyzed: {len(images)}")
        report.append(f"Images Passed: {passed_count}")
        report.append(f"Images Rejected: {rejected_count}")
        report.append("")
        
        if overall_result == "PASSED":
            report.append("OVERALL STATUS: PASSED - No fraud detected")
            report.append("All images are authentic and match their descriptions")
        else:
            report.append("OVERALL STATUS: FAILED - Fraud detected")
            report.append("One or more images failed validation checks")
        
        report.append("=" * 70)
        report.append(f"\nSubmission saved to: {submission_path}")
        
        # Save metadata
        self._save_metadata(submission_path, customer_name, car_details, desc_list, all_results)
        
        # Store structured results for the Gradio UI layer
        self._last_results = all_results
        
        return "\n".join(report)


# Initialize pipeline
pipeline = FraudDetectionPipeline()


def format_box(content, status="info"):
    """Wrap content in a styled status card"""
    css_class = {
        "pass": "result-pass",
        "fail": "result-fail",
        "warning": "result-warning",
        "info": "result-info",
    }.get(status, "result-info")
    return f'<div class="{css_class}">\n\n{content}\n\n</div>'


def process_gradio_submission(images, descriptions, customer_name, car_details):
    """Renders structured outputs for each Gradio component from pipeline results"""
    try:
        report_text = pipeline.process_submission(images, descriptions, customer_name, car_details)
        all_results = getattr(pipeline, '_last_results', None) or []

        if not all_results:
            return format_box("No results generated. Please check your inputs.", "info"), "", "", "", "", "", None

        overall_passed = "OVERALL STATUS: PASSED" in report_text

        # Confidence score
        scores = [r.get('confidence_score', 0) for r in all_results]
        avg_score = sum(scores) / len(scores) if scores else 0
        filled = round(avg_score / 10)
        bar = "\u2588" * filled + "\u2591" * (10 - filled)

        if avg_score >= 80:
            conf_label = "High Confidence"
        elif avg_score >= 60:
            conf_label = "Medium Confidence"
        elif avg_score >= 40:
            conf_label = "Low Confidence"
        else:
            conf_label = "Very Low Confidence"

        total_repair = sum(
            (r.get('damage_analysis') or {}).get('total_estimated_repair_cost', 0) or 0
            for r in all_results
        )

        verdict = "CLAIM PASSED" if overall_passed else "CLAIM FAILED"
        verdict_detail = (
            "No fraud indicators detected."
            if overall_passed
            else "Fraud indicators detected — manual review required."
        )

        summary_rows = [
            f"| Customer | {customer_name} |",
            f"| Vehicle | {car_details} |",
            f"| Images Analyzed | {len(all_results)} |",
        ]
        if total_repair > 0:
            summary_rows.append(f"| Estimated Repair Cost | \u20b9{total_repair:,.0f} |")
        table = "\n".join(summary_rows)

        per_image = ""
        if len(scores) > 1:
            lines = "\n".join(f"- Image {i + 1}: {s}/100" for i, s in enumerate(scores))
            per_image = f"\n\n**Per-image scores:**\n{lines}"

        overall_content = (
            f"## {verdict}\n\n"
            f"{verdict_detail}\n\n"
            f"**Confidence Score: {avg_score:.0f} / 100**  \n"
            f"`{bar}` {conf_label}\n\n"
            f"---\n\n"
            f"| | |\n|---|---|\n{table}"
            f"{per_image}"
        )
        overall_md = format_box(overall_content, "pass" if overall_passed else "fail")

        # --- Per-check renderers ---
        def render_ai(result, n):
            ai = result.get('ai_check')
            if not ai:
                return format_box(f"**Image {n}**\n\nCheck not available.", "info")
            failed = ai.get('is_ai_generated') and ai.get('confidence') == 'High'
            verdict_tag = "FAIL" if failed else "PASS"
            return format_box(
                f"**Image {n}** &nbsp;&nbsp; `{verdict_tag}`\n\n"
                f"| Field | Value |\n|---|---|\n"
                f"| AI Probability | {ai.get('ai_percentage', 0):.1f}% |\n"
                f"| Verdict | {ai.get('verdict', '\u2014')} |\n"
                f"| Confidence | {ai.get('confidence', '\u2014')} |",
                "fail" if failed else "pass",
            )

        def render_tampering(result, n):
            t = result.get('tampering_check')
            if not t:
                return format_box(f"**Image {n}**\n\nCheck not available.", "info")
            failed = t.get('is_tampered') and (t.get('tampering_score', 0) > 60)
            verdict_tag = "FAIL" if failed else "PASS"
            details = ' \u00b7 '.join((t.get('details') or [])[:2])
            body = (
                f"**Image {n}** &nbsp;&nbsp; `{verdict_tag}`\n\n"
                f"| Field | Value |\n|---|---|\n"
                f"| Tampering Score | {t.get('tampering_score', 0):.1f}% |\n"
                f"| ELA Prediction | {t.get('ela_prediction') or '\u2014'} |\n"
                f"| Confidence | {t.get('confidence', '\u2014')} |"
            )
            if details:
                body += f"\n\n*{details}*"
            return format_box(body, "fail" if failed else "pass")

        def render_description(result, n):
            d = result.get('description_check')
            if not d:
                return format_box(f"**Image {n}**\n\nCheck not available.", "info")
            manual = d.get('manual_check_required', False)
            if manual:
                status, verdict_tag = "warning", "REVIEW"
            elif d.get('matches'):
                status, verdict_tag = "pass", "PASS"
            else:
                status, verdict_tag = "fail", "FAIL"
            raw_reason = d.get('reasoning') or ''
            # Escape any leading/trailing asterisks that could break markdown rendering
            reasoning = raw_reason.strip()
            body = (
                f"**Image {n}** &nbsp;&nbsp; `{verdict_tag}`\n\n"
                f"| Field | Value |\n|---|---|\n"
                f"| Match Type | {d.get('match_type', '\u2014')} |\n"
                f"| Confidence | {d.get('confidence', 0):.2f} |\n"
                f"| Car Part | {d.get('car_part', '\u2014')} |\n"
                f"| Damage Status | {d.get('damage_status', '\u2014')} |"
            )
            if reasoning:
                body += f"\n\n**Reasoning**\n\n{reasoning}"
            return format_box(body, status)


        def render_duplication(result, n):
            dup = result.get('duplication_check')
            if not dup:
                return format_box(f"**Image {n}**\n\nCheck not available.", "info")
            is_dup = dup.get('is_duplicate', False)
            verdict_tag = "FAIL" if is_dup else "PASS"
            det = dup.get('details', {})
            extra = ""
            if is_dup:
                extra = (
                    f"\n| Matched File | {os.path.basename(det.get('duplicate_of', ''))} |"
                    f"\n| Similarity | {det.get('similarity_score', 0):.1%} |"
                    f"\n| Method | {det.get('method_used', '').upper()} |"
                )
            return format_box(
                f"**Image {n}** &nbsp;&nbsp; `{verdict_tag}`\n\n"
                f"| Field | Value |\n|---|---|\n"
                f"| Duplicate | {'Yes' if is_dup else 'No'} |\n"
                f"| Images in Database | {det.get('total_images_checked', 0)} |"
                + extra,
                "fail" if is_dup else "pass",
            )

        def render_damage(result, n):
            dmg = result.get('damage_analysis')
            if not dmg:
                return format_box(f"**Image {n}**\n\nCheck not available.", "info")
            if 'error' in dmg:
                return format_box(f"**Image {n}** &nbsp;&nbsp; `ERROR`\n\n{dmg['error']}", "fail")
            damaged = [p for p in dmg.get('damage_analysis', []) if p.get('is_damaged')]
            parts_lines = "\n".join(
                f"- {p['part_name']} \u2014 {p['severity']} severity" for p in damaged
            )
            cost = dmg.get('total_estimated_repair_cost', 0) or 0
            body = (
                f"**Image {n}** &nbsp;&nbsp; `COMPLETE`\n\n"
                f"| Field | Value |\n|---|---|\n"
                f"| Parts Detected | {dmg.get('num_parts', 0)} |\n"
                f"| Damage Types | {dmg.get('num_damage', 0)} |\n"
                f"| Overall Severity | {dmg.get('overall_severity', 'None')} |"
            )
            if cost > 0:
                body += f"\n| Repair Estimate | \u20b9{cost:,.0f} |"
            if parts_lines:
                body += f"\n\n**Damaged Parts:**\n{parts_lines}"
            return format_box(body, "pass")

        ai_parts = [render_ai(r, i + 1) for i, r in enumerate(all_results)]
        tmp_parts = [render_tampering(r, i + 1) for i, r in enumerate(all_results)]
        desc_parts = [render_description(r, i + 1) for i, r in enumerate(all_results)]
        dup_parts = [render_duplication(r, i + 1) for i, r in enumerate(all_results)]
        dmg_parts = [render_damage(r, i + 1) for i, r in enumerate(all_results)]

        ai_md = "**AI Generation Check**\n\n" + "\n\n".join(ai_parts)
        tmp_md = "**Tampering Detection**\n\n" + "\n\n".join(tmp_parts)
        desc_md = "**Description Matching**\n\n" + "\n\n".join(desc_parts)
        dup_md = "**Duplication Check**\n\n" + "\n\n".join(dup_parts)
        dmg_md = "**Damage Analysis**\n\n" + "\n\n".join(dmg_parts)

        vis_images = []
        if getattr(pipeline, '_last_submission_path', None):
            images_dir = os.path.join(pipeline._last_submission_path, "images")
            if os.path.exists(images_dir):
                for file in sorted(os.listdir(images_dir)):
                    if file.endswith(("_original.jpg", "_parts.jpg", "_damage.jpg")):
                        vis_images.append(os.path.join(images_dir, file))

        return overall_md, ai_md, tmp_md, desc_md, dup_md, dmg_md, vis_images or None

    except Exception as e:
        err = format_box(f"**Error**\n\n{str(e)}", "fail")
        return err, "", "", "", "", "", None



_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.result-pass {
    background: #f0fdf4 !important;
    border-left: 3px solid #16a34a !important;
    border-radius: 6px; padding: 14px 18px; margin: 6px 0;
    color: #14532d !important;
}
.result-fail {
    background: #fef2f2 !important;
    border-left: 3px solid #dc2626 !important;
    border-radius: 6px; padding: 14px 18px; margin: 6px 0;
    color: #7f1d1d !important;
}
.result-warning {
    background: #fffbeb !important;
    border-left: 3px solid #d97706 !important;
    border-radius: 6px; padding: 14px 18px; margin: 6px 0;
    color: #78350f !important;
}
.result-info {
    background: #f8fafc !important;
    border-left: 3px solid #64748b !important;
    border-radius: 6px; padding: 14px 18px; margin: 6px 0;
    color: #1e293b !important;
}
"""

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    background_fill_primary="#ffffff",
    background_fill_secondary="#f8fafc",
    block_background_fill="#ffffff",
    input_background_fill="#f8fafc",
    body_text_color="#0f172a",
    body_text_color_subdued="#475569",
    block_label_text_color="#475569",
    block_border_color="#e2e8f0",
    border_color_primary="#e2e8f0",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#ffffff",
    input_border_color="#cbd5e1",
    block_shadow="0 1px 3px rgba(0,0,0,0.06)",
)

with gr.Blocks(title="SureClaim", theme=_THEME, css=_CSS) as app:
    gr.Markdown(
        """# SureClaim
        Computerised fraud detection for automotive insurance claims."""
    )

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────────────────────
        with gr.Column(scale=1):
            customer_name = gr.Textbox(
                label="Customer Name",
                placeholder="Full name",
                value="John Doe",
            )
            car_details = gr.Textbox(
                label="Vehicle",
                placeholder="Make, model, year",
                value="Toyota Camry 2020",
            )
            images = gr.File(
                label="Damage Images",
                file_count="multiple",
                file_types=["image"],
            )
            descriptions = gr.Textbox(
                label="Damage Descriptions",
                placeholder="One description per image, comma or newline separated",
                lines=3,
                value="Damaged front bumper",
            )
            submit_btn = gr.Button("Analyze", variant="primary", size="lg")

        # ── Right column: results ─────────────────────────────────────────────
        with gr.Column(scale=1):
            overall_status = gr.Markdown()

            with gr.Tabs():
                with gr.Tab("AI Detection"):
                    ai_check_output = gr.Markdown()
                with gr.Tab("Tampering"):
                    tampering_output = gr.Markdown()
                with gr.Tab("Description"):
                    description_output = gr.Markdown()
                with gr.Tab("Duplication"):
                    duplication_output = gr.Markdown()
                with gr.Tab("Damage"):
                    damage_output = gr.Markdown()
                    damage_gallery = gr.Gallery(
                        label="Damage Visualizations",
                        show_label=True,
                        columns=3,
                        object_fit="contain",
                        height="auto",
                    )

    submit_btn.click(
        fn=process_gradio_submission,
        inputs=[images, descriptions, customer_name, car_details],
        outputs=[
            overall_status,
            ai_check_output,
            tampering_output,
            description_output,
            duplication_output,
            damage_output,
            damage_gallery,
        ],
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )

