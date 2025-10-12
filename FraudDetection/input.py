"""
Automotive Fraud Detection System
Main entry point with Gradio web interface
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import gradio as gr
import numpy as np

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
        
        try:
            self.tampering_detector = TamperingDetector(
                ela_model_path="Image-Tampering-Detection-using-ELA-and-Metadata-Analysis/ELA_Training/model_ela.h5",
                weather_model_path="Image-Tampering-Detection-using-ELA-and-Metadata-Analysis/WeatherCNNTraining/Weather_Model.h5"
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
                parts_model_path="../damage-det/model_parts.pth",
                damage_model_path="../damage-det/model_damage.pth",
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
        
        return "\n".join(report)


# Initialize pipeline
pipeline = FraudDetectionPipeline()


def format_box(content, status="info"):
    """Format content in a colored box based on status"""
    box_class = {
        "pass": "pass-box",
        "fail": "fail-box",
        "warning": "warning-box",
        "info": "info-box"
    }.get(status, "info-box")
    
    return f'<div class="{box_class}">\n\n{content}\n\n</div>'


def process_gradio_submission(images, descriptions, customer_name, car_details):
    """Wrapper function for Gradio interface - returns structured outputs for each component"""
    try:
        # Run the full pipeline
        report_text = pipeline.process_submission(images, descriptions, customer_name, car_details)
        
        # Extract confidence scores from the report
        confidence_scores = []
        for line in report_text.split('\n'):
            if "CONFIDENCE CLAIM SCORE:" in line:
                # Extract score (format: "CONFIDENCE CLAIM SCORE: XX/100")
                try:
                    score = int(line.split("CONFIDENCE CLAIM SCORE:")[1].split("/")[0].strip())
                    confidence_scores.append(score)
                except:
                    pass
        
        # Calculate average confidence score
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Extract total repair costs from the report
        total_repair_costs = []
        for line in report_text.split('\n'):
            if "TOTAL ESTIMATED COST: ₹" in line:
                try:
                    # Extract cost (format: "TOTAL ESTIMATED COST: ₹X,XXX.XX")
                    cost_str = line.split("TOTAL ESTIMATED COST: ₹")[1].strip()
                    cost = float(cost_str.replace(',', ''))
                    total_repair_costs.append(cost)
                except:
                    pass
        
        # Calculate overall total repair cost
        overall_repair_cost = sum(total_repair_costs) if total_repair_costs else 0
        
        # Extract overall status
        overall_passed = "OVERALL STATUS: PASSED" in report_text
        
        # Format confidence score display with color coding
        if avg_confidence >= 80:
            confidence_color = "🟢"
            confidence_status = "HIGH CONFIDENCE"
        elif avg_confidence >= 60:
            confidence_color = "🟡"
            confidence_status = "MEDIUM CONFIDENCE"
        elif avg_confidence >= 40:
            confidence_color = "🟠"
            confidence_status = "LOW CONFIDENCE"
        else:
            confidence_color = "🔴"
            confidence_status = "VERY LOW CONFIDENCE"
        
        # Build overall status with confidence score
        overall_status_content = f"## {'✅ PASSED' if overall_passed else '❌ FAILED'}\n\n"
        overall_status_content += f"# {confidence_color} CONFIDENCE CLAIM SCORE: {avg_confidence:.0f}/100\n"
        overall_status_content += f"**Status:** {confidence_status}\n\n"
        
        # Add repair cost estimate if available
        if overall_repair_cost > 0:
            overall_status_content += f"# TOTAL REPAIR COST: ₹{overall_repair_cost:,.2f}\n\n"
        
        overall_status_content += "---\n\n"
        overall_status_content += f"**Customer:** {customer_name}\n\n"
        overall_status_content += f"**Car:** {car_details}\n\n"
        overall_status_content += f"**Images Analyzed:** {len(images) if images else 0}\n\n"
        
        # Add per-image confidence breakdown if multiple images
        if len(confidence_scores) > 1:
            overall_status_content += "\n**Per-Image Scores:**\n"
            for i, score in enumerate(confidence_scores, 1):
                cost_info = f" | Cost: ₹{total_repair_costs[i-1]:,.2f}" if i-1 < len(total_repair_costs) and total_repair_costs[i-1] > 0 else ""
                overall_status_content += f"- Image {i}: {score}/100{cost_info}\n"
        
        overall_status_md = format_box(
            overall_status_content,
            "pass" if overall_passed else "fail"
        )
        
        # Split report by image sections
        image_sections = []
        lines = report_text.split('\n')
        current_image_section = []
        
        for line in lines:
            if line.startswith("IMAGE ") and "/" in line and current_image_section:
                image_sections.append('\n'.join(current_image_section))
                current_image_section = [line]
            else:
                current_image_section.append(line)
        
        if current_image_section:
            image_sections.append('\n'.join(current_image_section))
        
        # Extract results for each check type, separating by image
        ai_check_parts = []
        tampering_parts = []
        desc_parts = []
        dup_parts = []
        damage_parts = []
        
        for idx, section in enumerate(image_sections):
            image_num = idx + 1
            
            # Extract AI Check for this image
            if "[STEP 1/5] AI GENERATION CHECK" in section:
                ai_start = section.find("[STEP 1/5] AI GENERATION CHECK")
                ai_end = section.find("[STEP 2/5]", ai_start)
                ai_section = section[ai_start:ai_end] if ai_end > 0 else section[ai_start:ai_start+500]
                
                ai_passed = "Passed AI check" in ai_section or "SKIPPED" in ai_section
                ai_status = "pass" if ai_passed else ("fail" if "FRAUD DETECTED" in ai_section else "info")
                ai_check_parts.append(format_box(f"**Image {image_num}**\n\n{ai_section}", ai_status))
            
            # Extract Tampering Check for this image
            if "[STEP 2/5] TAMPERING CHECK" in section:
                tmp_start = section.find("[STEP 2/5] TAMPERING CHECK")
                tmp_end = section.find("[STEP 3/5]", tmp_start)
                tmp_section = section[tmp_start:tmp_end] if tmp_end > 0 else section[tmp_start:tmp_start+500]
                
                tmp_passed = "Passed tampering check" in tmp_section or "SKIPPED" in tmp_section
                tmp_status = "pass" if tmp_passed else ("fail" if "FRAUD DETECTED" in tmp_section else "info")
                tampering_parts.append(format_box(f"**Image {image_num}**\n\n{tmp_section}", tmp_status))
            
            # Extract Description Matching for this image
            if "[STEP 3/5] DESCRIPTION MATCHING" in section:
                desc_start = section.find("[STEP 3/5] DESCRIPTION MATCHING")
                desc_end = section.find("[STEP 4/5]", desc_start)
                desc_section = section[desc_start:desc_end] if desc_end > 0 else section[desc_start:desc_start+500]
                
                desc_passed = "Passed description check" in desc_section or "SKIPPED" in desc_section
                desc_warning = "WARNING: Manual check is required" in desc_section
                desc_status = "warning" if desc_warning else ("pass" if desc_passed else ("fail" if "FRAUD DETECTED" in desc_section else "info"))
                desc_parts.append(format_box(f"**Image {image_num}**\n\n{desc_section}", desc_status))
            
            # Extract Duplication Check for this image
            if "[STEP 4/5] DUPLICATION CHECK" in section:
                dup_start = section.find("[STEP 4/5] DUPLICATION CHECK")
                dup_end = section.find("[STEP 5/5]", dup_start)
                dup_section = section[dup_start:dup_end] if dup_end > 0 else section[dup_start:dup_start+500]
                
                dup_passed = "No duplicates found" in dup_section or "SKIPPED" in dup_section
                dup_status = "pass" if dup_passed else ("fail" if "DUPLICATE FOUND" in dup_section else "info")
                dup_parts.append(format_box(f"**Image {image_num}**\n\n{dup_section}", dup_status))
            
            # Extract Damage Analysis for this image
            if "[STEP 5/5] PART DETECTION & DAMAGE CLASSIFICATION" in section:
                dmg_start = section.find("[STEP 5/5] PART DETECTION & DAMAGE CLASSIFICATION")
                dmg_end = section.find("FINAL VERDICT", dmg_start)
                if dmg_end == -1:
                    dmg_end = section.find("\n\n\n", dmg_start)
                dmg_section = section[dmg_start:dmg_end] if dmg_end > 0 else section[dmg_start:]
                
                # Check if damage analysis was skipped or had errors
                if "SKIPPED" in dmg_section or "not available" in dmg_section:
                    dmg_status = "info"
                elif "Error:" in dmg_section:
                    dmg_status = "fail"
                else:
                    # Successful analysis - show as pass (green) regardless of damage found
                    # The severity is informational, not a pass/fail criterion
                    dmg_status = "pass"
                
                damage_parts.append(format_box(f"**Image {image_num}**\n\n{dmg_section}", dmg_status))
        
        # Combine all parts with proper formatting
        ai_check_md = f"### 🤖 AI Generation Check\n\n" + "\n\n".join(ai_check_parts) if ai_check_parts else "No AI check performed"
        tampering_md = f"### � Tampering Check\n\n" + "\n\n".join(tampering_parts) if tampering_parts else "No tampering check performed"
        desc_md = f"### 📝 Description Matching\n\n" + "\n\n".join(desc_parts) if desc_parts else "No description check performed"
        dup_md = f"### 🔄 Duplication Check\n\n" + "\n\n".join(dup_parts) if dup_parts else "No duplication check performed"
        damage_md = f"### 🔧 Damage Analysis\n\n" + "\n\n".join(damage_parts) if damage_parts else "No damage analysis performed"
        
        # Collect visualization images (now 3 per image: original, parts, damage)
        vis_images = []
        if hasattr(pipeline, '_last_submission_path') and pipeline._last_submission_path:
            images_dir = os.path.join(pipeline._last_submission_path, "images")
            if os.path.exists(images_dir):
                for file in sorted(os.listdir(images_dir)):
                    # Collect all three types: original, parts, and damage visualizations
                    if file.endswith(("_original.jpg", "_parts.jpg", "_damage.jpg")):
                        vis_images.append(os.path.join(images_dir, file))
        
        return (
            overall_status_md,
            ai_check_md,
            tampering_md,
            desc_md,
            dup_md,
            damage_md,
            vis_images if vis_images else None
        )
        
    except Exception as e:
        error_msg = format_box(f"## ❌ ERROR\n\n{str(e)}\n\nPlease check your inputs and try again.", "fail")
        return error_msg, "", "", "", "", "", None


# Create Gradio Interface
with gr.Blocks(title="Automotive Fraud Detection System", css="""
    .pass-box { background-color: #d4edda !important; border: 2px solid #28a745 !important; border-radius: 8px; padding: 15px; margin: 10px 0; }
    .pass-box, .pass-box * { color: #000000 !important; }
    .fail-box { background-color: #f8d7da !important; border: 2px solid #dc3545 !important; border-radius: 8px; padding: 15px; margin: 10px 0; }
    .fail-box, .fail-box * { color: #000000 !important; }
    .warning-box { background-color: #fff3cd !important; border: 2px solid #ffc107 !important; border-radius: 8px; padding: 15px; margin: 10px 0; }
    .warning-box, .warning-box * { color: #000000 !important; }
    .info-box { background-color: #d1ecf1 !important; border: 2px solid #17a2b8 !important; border-radius: 8px; padding: 15px; margin: 10px 0; }
    .info-box, .info-box * { color: #000000 !important; }
""") as app:
    gr.Markdown(
        """
        # 🚗 Automotive Fraud Detection System
        
        This system analyzes automotive damage images to detect potential fraud through:
        1. **AI Generation Detection** - Identifies AI-generated fake images
        2. **Tampering Detection** - Detects image manipulation using ELA analysis
        3. **Description Matching** - Verifies images match the provided damage descriptions
        4. **Duplication Check** - Detects previously submitted images
        5. **Part Detection & Damage Classification** - Identifies car parts and damage types with severity
        
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Customer Information")
            customer_name = gr.Textbox(
                label="Customer Name",
                placeholder="Enter customer name",
                value="John Doe"
            )
            car_details = gr.Textbox(
                label="Car Details",
                placeholder="e.g., Toyota Camry 2020, License Plate: ABC-1234",
                lines=2,
                value="Toyota Camry 2020"
            )
            
            gr.Markdown("### 📸 Upload Images")
            images = gr.File(
                label="Upload Car Damage Images",
                file_count="multiple",
                file_types=["image"]
            )
            
            descriptions = gr.Textbox(
                label="Image Descriptions",
                placeholder="Enter descriptions separated by commas or new lines\ne.g., Damaged front bumper, Scratched door, Broken headlight",
                lines=4,
                value="Damaged front bumper"
            )
            
            submit_btn = gr.Button("🔍 Analyze for Fraud", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Analysis Summary")
            overall_status = gr.Markdown(label="Overall Status")
            
            gr.Markdown("### 🔬 Detailed Analysis")
            ai_check_output = gr.Markdown(label="AI Generation Check")
            tampering_output = gr.Markdown(label="Tampering Check")
            description_output = gr.Markdown(label="Description Match")
            duplication_output = gr.Markdown(label="Duplication Check")
            damage_output = gr.Markdown(label="Damage Analysis")
    
    # Add gallery for damage analysis visualizations
    with gr.Row():
        gr.Markdown("### 🎨 Damage Analysis Visualizations")
    with gr.Row():
        damage_gallery = gr.Gallery(
            label="Original Image | Car Parts Detection | Damage Analysis (per image)",
            show_label=True,
            columns=3,
            object_fit="contain",
            height="auto"
        )
    
    gr.Markdown(
        """
        ---
        ### 📖 Instructions:
        1. Enter customer name and car details
        2. Upload one or more images of car damage
        3. Provide descriptions for each image (comma or newline separated)
        4. Click "Analyze for Fraud" to process
        5. Review the **Confidence Claim Score**, **Repair Cost Estimates**, and detailed report
        
        ### 📊 Confidence Claim Score (0-100):
        - **🟢 80-100**: HIGH CONFIDENCE - Claim appears legitimate
        - **🟡 60-79**: MEDIUM CONFIDENCE - Some concerns, review recommended
        - **🟠 40-59**: LOW CONFIDENCE - Multiple red flags detected
        - **🔴 0-39**: VERY LOW CONFIDENCE - High fraud risk
        
        The score is calculated based on:
        - AI Generation Check (25 points)
        - Tampering Detection (25 points)
        - Description Matching (25 points)
        - Duplication Check (25 points)
        
        ### 💰 Repair Cost Estimation:
        - The system automatically estimates repair costs based on:
          - **Detected car parts** (Front bumper, Hood, Door, etc.)
          - **Damage severity** (Low, Medium, High)
          - **Market-based part prices** (average replacement values)
        - Cost estimates are shown for each damaged part
        - **Total repair cost** is displayed in the summary
        - Costs are calculated as: Base Part Price × Severity Multiplier
        
        ### 📝 Notes:
        - **Green boxes** = Passed validation ✅
        - **Red boxes** = Failed validation / Fraud detected ❌
        - **Yellow boxes** = Warnings / Manual review needed ⚠️
        - **Blue boxes** = Information / Skipped checks ℹ️
        - All submissions are saved with timestamp in `fraud_detection_data/` folder
        - The system performs comprehensive 5-stage validation for fraud detection
        - Damage visualizations show detected car parts with color-coded severity levels
        """
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
            damage_gallery
        ]
    )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Starting Automotive Fraud Detection System...")
    print("=" * 70)
    print("\nThe web interface will open in your browser.")
    print("You can also access it at: http://localhost:7860")
    print("\nPress Ctrl+C to stop the server.")
    print("=" * 70 + "\n")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
