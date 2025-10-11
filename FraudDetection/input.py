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

# Import detection modules
from ai_detector import AIImageDetector
from tampering_check import TamperingDetector
from description_check import DescriptionMatcher


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
        
        # Create base directory for storing submissions
        self.base_dir = "fraud_detection_data"
        os.makedirs(self.base_dir, exist_ok=True)
        
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
    
    def _save_metadata(self, submission_path, customer_name, car_details, descriptions, results):
        """Save submission metadata to JSON file"""
        metadata = {
            "customer_name": customer_name,
            "car_details": car_details,
            "submission_date": datetime.now().isoformat(),
            "descriptions": descriptions,
            "results": results
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
                'status': 'UNKNOWN'
            }
            
            # STEP 1: AI Detection
            report.append("\n[STEP 1/3] AI GENERATION CHECK")
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
                    all_results.append(result)
                    continue  # Skip further checks
                else:
                    report.append("RESULT: Passed AI check")
            else:
                report.append("AI Detector not available - SKIPPED")
            
            # STEP 2: Tampering Detection
            report.append("\n[STEP 2/3] TAMPERING CHECK")
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
                    all_results.append(result)
                    continue  # Skip further checks
                else:
                    report.append("RESULT: Passed tampering check")
            else:
                report.append("Tampering Detector not available - SKIPPED")
            
            # STEP 3: Description Matching
            report.append("\n[STEP 3/3] DESCRIPTION MATCHING")
            report.append("-" * 70)
            
            if self.description_matcher:
                desc_result = self.description_matcher.verify(image_path, description)
                result['description_check'] = desc_result
                
                report.append(f"Match Type: {desc_result['match_type']}")
                report.append(f"Confidence: {desc_result['confidence']:.2f}")
                report.append(f"Car Part: {desc_result['car_part']}")
                report.append(f"Damage Status: {desc_result['damage_status']}")
                report.append(f"Reasoning: {desc_result['reasoning'][:200]}...")
                
                # Check if description matches
                if not desc_result['matches']:
                    report.append("\nRESULT: FRAUD DETECTED - Image does not match description")
                    report.append("STATUS: REJECTED")
                    result['status'] = 'REJECTED_DESCRIPTION_MISMATCH'
                    overall_result = "FAILED"
                else:
                    report.append("\nRESULT: Passed description check")
                    result['status'] = 'PASSED'
            else:
                report.append("Description Matcher not available - SKIPPED")
                result['status'] = 'PASSED'
            
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


def process_gradio_submission(images, descriptions, customer_name, car_details):
    """Wrapper function for Gradio interface"""
    try:
        return pipeline.process_submission(images, descriptions, customer_name, car_details)
    except Exception as e:
        return f"ERROR: {str(e)}\n\nPlease check your inputs and try again."


# Create Gradio Interface
with gr.Blocks(title="Automotive Fraud Detection System", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # Automotive Fraud Detection System
        
        This system analyzes automotive damage images to detect potential fraud through:
        1. **AI Generation Detection** - Identifies AI-generated fake images
        2. **Tampering Detection** - Detects image manipulation using ELA analysis
        3. **Description Matching** - Verifies images match the provided damage descriptions
        
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Customer Information")
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
            
            gr.Markdown("### Upload Images")
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
            
            submit_btn = gr.Button("Analyze for Fraud", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### Analysis Results")
            output = gr.Textbox(
                label="Fraud Detection Report",
                lines=30,
                max_lines=50,
                show_copy_button=True
            )
    
    gr.Markdown(
        """
        ---
        ### Instructions:
        1. Enter customer name and car details
        2. Upload one or more images of car damage
        3. Provide descriptions for each image (comma or newline separated)
        4. Click "Analyze for Fraud" to process
        5. Review the detailed report
        
        ### Notes:
        - All submissions are saved with timestamp in `fraud_detection_data/` folder
        - Each submission includes images, descriptions, and analysis results
        - The system performs three-stage validation for comprehensive fraud detection
        """
    )
    
    submit_btn.click(
        fn=process_gradio_submission,
        inputs=[images, descriptions, customer_name, car_details],
        outputs=output
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
