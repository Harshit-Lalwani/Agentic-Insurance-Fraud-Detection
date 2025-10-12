"""
Combined Car Damage and Parts Detection Module
Combines car parts detection with damage detection to provide comprehensive analysis
"""

import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import json
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Import existing detectors
from car_parts_detector import CarPartsDetector

# Detectron2 imports for damage detection
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances
import torchvision


class CombinedDamageDetector:
    """Combined detector for car parts and damage analysis"""
    
    # Damage types mapping (from damage detection model)
    DAMAGE_TYPES = {
        0: "Dent",
        1: "Scratch", 
        2: "Broken part",
        3: "Paint chip",
        4: "Missing part",
        5: "Flaking",
        6: "Corrosion",
        7: "Cracked"
    }
    
    # Severity thresholds based on damage-to-part area ratio
    SEVERITY_THRESHOLDS = {
        'Low': 0.15,      # < 15% of part area
        'Medium': 0.50,   # 15-50% of part area (increased from 35%)
        'High': 0.50      # > 50% of part area (increased from 35%)
    }
    
    # Fixed base prices for car parts (in INR) - average market prices
    PART_BASE_PRICES = {
        'Front-bumper': 800,
        'Back-bumper': 750,
        'Hood': 1200,
        'Trunk': 900,
        'Front-door': 600,
        'Back-door': 550,
        'Fender': 450,
        'Quarter-panel': 800,
        'Rocker-panel': 400,
        'Headlight': 350,
        'Tail-light': 250,
        'Mirror': 200,
        'Windshield': 300,
        'Back-window': 280,
        'Front-window': 200,
        'Back-windshield': 280,
        'Grille': 350,
        'License-plate': 25,
        'Roof': 1500,
        'Front-wheel': 800,
        'Back-wheel': 800
    }
    
    # Damage severity cost multipliers
    SEVERITY_COST_MULTIPLIERS = {
        'Low': 0.25,      # 25% of part value (minor repair)
        'Medium': 0.60,   # 60% of part value (major repair)
        'High': 0.90,     # 90% of part value (near replacement)
        'None': 0.0       # No cost
    }
    
    def __init__(self, parts_model_path, damage_model_path, confidence_threshold=0.5, verbose=True):
        """
        Initialize combined detector
        
        Args:
            parts_model_path: Path to car parts detection model (.pth)
            damage_model_path: Path to damage detection model (.pth)
            confidence_threshold: Minimum confidence for detections
            verbose: Print initialization info
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize car parts detector
        if verbose:
            print("Initializing car parts detector...")
        self.parts_detector = CarPartsDetector(
            model_path=parts_model_path,
            confidence_threshold=confidence_threshold,
            verbose=verbose
        )
        
        # Initialize damage detector
        if verbose:
            print("Initializing damage detector...")
        self.damage_predictor, self.damage_cfg = self._setup_damage_detector(
            damage_model_path, confidence_threshold
        )
        
        if verbose:
            print("Combined detector initialized successfully!\n")
    
    def _setup_damage_detector(self, model_path, confidence_threshold):
        """Setup damage detection model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Damage model not found: {model_path}")
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
        
        cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # 8 damage types
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        
        predictor = DefaultPredictor(cfg)
        
        # Set up metadata for visualization
        MetadataCatalog.get("car_damage").thing_classes = list(self.DAMAGE_TYPES.values())
        
        return predictor, cfg
    
    def _calculate_repair_cost(self, part_name, severity):
        """
        Calculate estimated repair cost based on part and damage severity
        
        Args:
            part_name: Name of the damaged part
            severity: Damage severity ('Low', 'Medium', 'High', 'None')
            
        Returns:
            dict with price information
        """
        # Get base price for the part
        base_price = self.PART_BASE_PRICES.get(part_name)
        
        if base_price is None or severity == 'None':
            return {
                'part_name': part_name,
                'base_price': base_price,
                'severity': severity,
                'multiplier': self.SEVERITY_COST_MULTIPLIERS.get(severity, 0),
                'estimated_repair_cost': 0.0,
                'price_available': base_price is not None
            }
        
        # Calculate repair cost based on severity
        multiplier = self.SEVERITY_COST_MULTIPLIERS.get(severity, 0)
        repair_cost = base_price * multiplier
        
        return {
            'part_name': part_name,
            'base_price': base_price,
            'severity': severity,
            'multiplier': multiplier,
            'estimated_repair_cost': round(repair_cost, 2),
            'price_available': True
        }
    
    def _apply_nms(self, output, nms_threshold=0.1):
        """Apply Non-Maximum Suppression to damage detections"""
        if len(output["instances"]) == 0:
            return output["instances"]
        
        boxes = output["instances"].pred_boxes.tensor.cpu()
        scores = output["instances"].scores.cpu()
        classes = output["instances"].pred_classes.cpu()
        
        # Perform NMS
        keep = torchvision.ops.nms(boxes, scores, nms_threshold)
        
        # Create filtered instances
        nms_instances = Instances(output["instances"].image_size)
        nms_instances.pred_boxes = Boxes(boxes[keep])
        nms_instances.scores = scores[keep]
        nms_instances.pred_classes = classes[keep]
        
        # Copy masks if available
        if output["instances"].has("pred_masks"):
            nms_instances.pred_masks = output["instances"].pred_masks[keep]
        
        return nms_instances
    
    def _calculate_overlap_ratio(self, damage_bbox, part_bbox):
        """Calculate overlap ratio between damage and car part using bounding boxes"""
        try:
            # Extract coordinates: [x1, y1, x2, y2]
            d_x1, d_y1, d_x2, d_y2 = damage_bbox
            p_x1, p_y1, p_x2, p_y2 = part_bbox
            
            # Calculate intersection rectangle
            x1 = max(d_x1, p_x1)
            y1 = max(d_y1, p_y1)
            x2 = min(d_x2, p_x2)
            y2 = min(d_y2, p_y2)
            
            # Check if there's an intersection
            if x1 >= x2 or y1 >= y2:
                return 0.0
            
            # Calculate areas
            intersection_area = (x2 - x1) * (y2 - y1)
            part_area = (p_x2 - p_x1) * (p_y2 - p_y1)
            
            if part_area == 0:
                return 0.0
            
            # Return ratio of intersection to part area
            overlap_ratio = intersection_area / part_area
            
            # Debug output
            print(f"  Damage bbox: {damage_bbox}")
            print(f"  Part bbox: {part_bbox}")
            print(f"  Intersection area: {intersection_area:.1f}")
            print(f"  Part area: {part_area:.1f}")
            print(f"  Overlap ratio: {overlap_ratio:.3f}")
            
            return overlap_ratio
            
        except Exception as e:
            print(f"Error calculating overlap: {e}")
            return 0.0
    
    def _determine_severity(self, overlap_ratio):
        """Determine damage severity based on overlap ratio"""
        if overlap_ratio < self.SEVERITY_THRESHOLDS['Low']:
            return 'Low'
        elif overlap_ratio < self.SEVERITY_THRESHOLDS['Medium']:
            return 'Medium'
        else:
            return 'High'
    
    def detect_damage_and_parts(self, image_path, nms_threshold=0.1):
        """
        Detect both car parts and damage, then analyze their relationships
        
        Args:
            image_path: Path to the image file
            nms_threshold: IoU threshold for NMS on damage detections
            
        Returns:
            dict: Comprehensive analysis results
        """
        if not os.path.exists(image_path):
            return {'error': f"Image not found: {image_path}"}
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': f"Failed to load image: {image_path}"}
            
            # Detect car parts
            parts_result = self.parts_detector.detect(image_path, return_visualization=False)
            if 'error' in parts_result:
                return {'error': f"Parts detection failed: {parts_result['error']}"}
            
            # Detect damage
            damage_output = self.damage_predictor(image)
            damage_instances = self._apply_nms(damage_output, nms_threshold)
            
            # Process damage detections
            damage_detections = []
            if len(damage_instances) > 0:
                for i in range(len(damage_instances)):
                    damage_class = int(damage_instances.pred_classes[i])
                    damage_type = self.DAMAGE_TYPES[damage_class]
                    confidence = float(damage_instances.scores[i])
                    bbox = damage_instances.pred_boxes.tensor[i].cpu().numpy()
                    
                    damage_info = {
                        'damage_type': damage_type,
                        'confidence': confidence,
                        'bbox': bbox.tolist(),
                        'class_id': damage_class
                    }
                    
                    # Add mask if available
                    if damage_instances.has("pred_masks"):
                        damage_info['mask'] = damage_instances.pred_masks[i].cpu().numpy()
                    
                    damage_detections.append(damage_info)
            
            # Analyze damage-part relationships
            damage_analysis = self._analyze_damage_part_relationships(
                parts_result['detections'], damage_detections
            )
            
            # Calculate price estimates for damaged parts
            price_estimates = []
            total_repair_cost = 0
            
            print("\nCalculating repair costs...")
            for part_analysis in damage_analysis:
                if part_analysis['is_damaged']:
                    cost_info = self._calculate_repair_cost(
                        part_analysis['part_name'],
                        part_analysis['severity']
                    )
                    price_estimates.append(cost_info)
                    if cost_info['estimated_repair_cost'] > 0:
                        total_repair_cost += cost_info['estimated_repair_cost']
                        print(f"  {part_analysis['part_name']}: ₹{cost_info['estimated_repair_cost']:.2f} ({part_analysis['severity']} severity)")
            
            print(f"\nTotal estimated repair cost: ₹{total_repair_cost:.2f}")
            
            # Compile results
            result = {
                'image_path': image_path,
                'parts_detected': parts_result['detected_parts'],
                'num_parts': parts_result['num_parts'],
                'damage_detected': [d['damage_type'] for d in damage_detections],
                'num_damage': len(damage_detections),
                'parts_detections': parts_result['detections'],
                'damage_detections': damage_detections,
                'damage_analysis': damage_analysis,
                'overall_severity': self._calculate_overall_severity(damage_analysis),
                'price_estimates': price_estimates,
                'total_estimated_repair_cost': round(total_repair_cost, 2)
            }
            
            return result
            
        except Exception as e:
            return {'error': f"Detection failed: {str(e)}"}
    
    def _analyze_damage_part_relationships(self, parts_detections, damage_detections):
        """Analyze which parts have which damage and severity"""
        analysis = []
        
        print(f"Analyzing {len(parts_detections)} parts against {len(damage_detections)} damage detections...")
        
        for part in parts_detections:
            part_name = part['part_name']
            part_analysis = {
                'part_name': part_name,
                'damages': [],
                'total_damage_ratio': 0.0,
                'severity': 'None',
                'is_damaged': False
            }
            
            print(f"\nChecking part: {part_name}")
            
            # Check each damage detection against this part
            for i, damage in enumerate(damage_detections):
                if 'bbox' in part and 'bbox' in damage:
                    print(f"  Checking against damage {i+1}: {damage['damage_type']}")
                    overlap_ratio = self._calculate_overlap_ratio(
                        damage['bbox'], part['bbox']
                    )
                    
                    # Consider it a match if there's significant overlap
                    if overlap_ratio > 0.01:  # 1% minimum overlap threshold (lowered for testing)
                        damage_info = {
                            'damage_type': damage['damage_type'],
                            'confidence': damage['confidence'],
                            'overlap_ratio': overlap_ratio,
                            'severity': self._determine_severity(overlap_ratio)
                        }
                        part_analysis['damages'].append(damage_info)
                        part_analysis['total_damage_ratio'] += overlap_ratio
                        part_analysis['is_damaged'] = True
                        print(f"    ✓ DAMAGE FOUND: {damage['damage_type']} ({overlap_ratio:.1%} overlap)")
                    else:
                        print(f"    ✗ No significant overlap ({overlap_ratio:.1%})")
                else:
                    print(f"    ✗ Missing bbox data - part: {'bbox' in part}, damage: {'bbox' in damage}")
            
            # Determine overall severity for this part
            if part_analysis['is_damaged']:
                part_analysis['severity'] = self._determine_severity(
                    part_analysis['total_damage_ratio']
                )
            
            analysis.append(part_analysis)
        
        return analysis
        
        return analysis
    
    def _calculate_overall_severity(self, damage_analysis):
        """Calculate overall damage severity for the vehicle"""
        if not damage_analysis:
            return 'None'
        
        damaged_parts = [part for part in damage_analysis if part['is_damaged']]
        
        if not damaged_parts:
            return 'None'
        
        # Count severity levels
        high_count = sum(1 for part in damaged_parts if part['severity'] == 'High')
        medium_count = sum(1 for part in damaged_parts if part['severity'] == 'Medium')
        
        total_damaged = len(damaged_parts)
        
        # Determine overall severity
        if high_count > 0 or total_damaged > 3:
            return 'High'
        elif medium_count > 1 or total_damaged > 1:
            return 'Medium'
        else:
            return 'Low'
    
    def generate_report(self, result):
        """Generate a human-readable damage assessment report"""
        if 'error' in result:
            return f"Error: {result['error']}"
        
        report = []
        report.append("=" * 70)
        report.append("CAR DAMAGE ASSESSMENT REPORT")
        report.append("=" * 70)
        report.append(f"Image: {os.path.basename(result['image_path'])}")
        report.append(f"Parts Detected: {result['num_parts']}")
        report.append(f"Damage Types Found: {result['num_damage']}")
        report.append(f"Overall Severity: {result['overall_severity']}")
        
        # Add cost summary
        if result.get('total_estimated_repair_cost', 0) > 0:
            report.append(f"Total Estimated Repair Cost: ₹{result['total_estimated_repair_cost']:,.2f}")
        report.append("")
        
        # Summary of detected damage types
        if result['damage_detected']:
            damage_counts = {}
            for damage_type in result['damage_detected']:
                damage_counts[damage_type] = damage_counts.get(damage_type, 0) + 1
            
            report.append("DAMAGE TYPES DETECTED:")
            for damage_type, count in damage_counts.items():
                report.append(f"  • {damage_type}: {count} instance(s)")
            report.append("")
        
        # Detailed part-by-part analysis
        report.append("PART-BY-PART ANALYSIS:")
        report.append("-" * 70)
        
        damaged_parts = [part for part in result['damage_analysis'] if part['is_damaged']]
        undamaged_parts = [part for part in result['damage_analysis'] if not part['is_damaged']]
        
        if damaged_parts:
            for part in damaged_parts:
                report.append(f"\n🔴 {part['part_name']} - DAMAGED ({part['severity']} severity)")
                
                # Find price estimate for this part
                price_info = None
                for estimate in result.get('price_estimates', []):
                    if estimate['part_name'] == part['part_name']:
                        price_info = estimate
                        break
                
                # Display damage details
                for damage in part['damages']:
                    report.append(f"   • {damage['damage_type']}: {damage['overlap_ratio']:.1%} coverage")
                    report.append(f"     Confidence: {damage['confidence']:.2f}, Severity: {damage['severity']}")
                
                # Display price estimate
                if price_info and price_info['price_available']:
                    report.append(f"   💰 Base Part Price: ₹{price_info['base_price']:,.2f}")
                    report.append(f"   💰 Estimated Repair Cost: ₹{price_info['estimated_repair_cost']:,.2f}")
                    report.append(f"      (Based on {price_info['multiplier']:.0%} of part value for {price_info['severity']} severity)")
        
        if undamaged_parts:
            report.append(f"\n🟢 UNDAMAGED PARTS ({len(undamaged_parts)}):")
            # Get unique undamaged part names to avoid duplicates
            unique_undamaged_names = list(set([part['part_name'] for part in undamaged_parts]))
            unique_undamaged_names.sort()  # Sort alphabetically for consistency
            report.append(f"   {', '.join(unique_undamaged_names)}")
        
        # Cost summary
        if result.get('price_estimates'):
            report.append(f"\n" + "-" * 70)
            report.append("REPAIR COST SUMMARY:")
            parts_with_prices = [p for p in result['price_estimates'] if p['price_available'] and p['estimated_repair_cost'] > 0]
            
            if parts_with_prices:
                for price_info in parts_with_prices:
                    report.append(f"  {price_info['part_name']:.<40} ₹{price_info['estimated_repair_cost']:>10,.2f}")
                report.append(f"  {'TOTAL ESTIMATED COST':.<40} ₹{result['total_estimated_repair_cost']:>10,.2f}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def save_visualization(self, image_path, result, output_path):
        """
        Create and save 3 separate visualizations showing original, parts, and damage
        
        Args:
            image_path: Path to the input image
            result: Detection results dictionary
            output_path: Base path for output files (without extension)
            
        Returns:
            list: Paths to the 3 generated images [original, parts, damage]
        """
        if 'error' in result:
            print(f"Cannot create visualization: {result['error']}")
            return []
        
        try:
            # Load image
            image = cv2.imread(image_path)
            
            # Get base path without extension
            base_path = os.path.splitext(output_path)[0]
            
            # Generate 3 separate images
            output_paths = []
            
            # 1. Save original image
            original_path = f"{base_path}_original.jpg"
            cv2.imwrite(original_path, image)
            output_paths.append(original_path)
            print(f"Original image saved to: {original_path}")
            
            # 2. Generate and save parts detection visualization
            parts_vis = self.parts_detector.detect(image_path, return_visualization=True)
            parts_path = f"{base_path}_parts.jpg"
            if 'visualization' in parts_vis:
                cv2.imwrite(parts_path, parts_vis['visualization'])
                output_paths.append(parts_path)
                print(f"Parts detection saved to: {parts_path}")
            else:
                # Fallback to original if parts visualization fails
                cv2.imwrite(parts_path, image)
                output_paths.append(parts_path)
            
            # 3. Generate and save combined damage analysis visualization
            combined_image = self._create_combined_visualization(image, result)
            damage_path = f"{base_path}_damage.jpg"
            cv2.imwrite(damage_path, combined_image)
            output_paths.append(damage_path)
            print(f"Damage analysis saved to: {damage_path}")
            
            print(f"All visualizations saved successfully!")
            return output_paths
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return []
    
    def _create_combined_visualization(self, image, result):
        """Create visualization highlighting damaged parts"""
        vis_image = image.copy()
        
        # Draw damaged parts with different colors based on severity
        severity_colors = {
            'Low': (0, 255, 255),      # Yellow
            'Medium': (0, 165, 255),   # Orange
            'High': (0, 0, 255),       # Red
            'None': (0, 255, 0)        # Green
        }
        
        for part_analysis in result['damage_analysis']:
            if part_analysis['is_damaged']:
                # Find corresponding part detection
                part_detection = None
                for part in result['parts_detections']:
                    if part['part_name'] == part_analysis['part_name']:
                        part_detection = part
                        break
                
                if part_detection and 'bbox' in part_detection:
                    bbox = part_detection['bbox']
                    color = severity_colors[part_analysis['severity']]
                    
                    # Draw bounding box
                    cv2.rectangle(
                        vis_image,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        color, 3
                    )
                    
                    # Add label
                    label = f"{part_analysis['part_name']} ({part_analysis['severity']})"
                    cv2.putText(
                        vis_image, label,
                        (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                    )
        
        return vis_image


def main():
    """Example usage of the combined detector"""
    import sys
    
    # Check for command line arguments
    if len(sys.argv) < 2:
        print("Usage: python combined_damage_detector.py <image_path>")
        print("Example: python combined_damage_detector.py '../archive/Car damages dataset/File1/img/Car damages 25.png'")
        return
    
    # Get image path from command line argument
    test_image = sys.argv[1]
    
    # Model paths
    parts_model_path = "../part_detection_model.pth"  # Car parts model
    damage_model_path = "../damage_model.pth"  # Damage detection model
    
    # Initialize combined detector
    try:
        detector = CombinedDamageDetector(
            parts_model_path=parts_model_path,
            damage_model_path=damage_model_path,
            confidence_threshold=0.5
        )
        
        if not os.path.exists(test_image):
            print(f"Test image not found: {test_image}")
            return
        
        print("Running combined damage detection...")
        
        # Perform detection
        result = detector.detect_damage_and_parts(test_image)
        
        # Generate and print report
        report = detector.generate_report(result)
        print(report)
        
        # Save visualization
        output_dir = "combined_analysis_output"
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(test_image))[0]
        vis_path = os.path.join(output_dir, f"{base_name}_damage_analysis.jpg")
        json_path = os.path.join(output_dir, f"{base_name}_analysis.json")
        
        detector.save_visualization(test_image, result, vis_path)
        
        # Save JSON results
        # Remove numpy arrays from result for JSON serialization
        json_result = result.copy()
        for detection in json_result.get('parts_detections', []):
            if 'mask' in detection:
                del detection['mask']
        for detection in json_result.get('damage_detections', []):
            if 'mask' in detection:
                del detection['mask']
        
        with open(json_path, 'w') as f:
            json.dump(json_result, f, indent=2, default=str)
        
        print(f"\nResults saved to:")
        print(f"  - Visualization: {vis_path}")
        print(f"  - JSON Report: {json_path}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()