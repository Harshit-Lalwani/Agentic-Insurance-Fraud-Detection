"""
Car Parts Detection Module
Detects and segments car parts in images using Detectron2 Mask R-CNN
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog


class CarPartsDetector:
    """Detect car parts using trained Mask R-CNN model"""
    
    # Car parts mapping (from training annotations)
    CAR_PARTS = {
        1: "Quarter-panel",
        2: "Front-wheel",
        3: "Back-window",
        4: "Trunk",
        5: "Front-door",
        6: "Rocker-panel",
        7: "Grille",
        8: "Windshield",
        9: "Front-window",
        10: "Back-door",
        11: "Headlight",
        12: "Back-wheel",
        13: "Back-windshield",
        14: "Hood",
        15: "Fender",
        16: "Tail-light",
        17: "License-plate",
        18: "Front-bumper",
        19: "Back-bumper",
        20: "Mirror",
        21: "Roof"
    }
    
    def __init__(self, model_path, config_file=None, confidence_threshold=0.5, verbose=True):
        """
        Initialize car parts detector
        
        Args:
            model_path: Path to the trained model weights (.pth file)
            config_file: Optional Detectron2 config file path
            confidence_threshold: Minimum confidence for detections (0-1)
            verbose: Print initialization info (default: True)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.confidence_threshold = confidence_threshold
        
        # Setup Detectron2 config
        cfg = get_cfg()
        
        # Use Mask R-CNN with ResNet-50 backbone (common for car parts detection)
        if config_file:
            cfg.merge_from_file(config_file)
        else:
            cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            ))
        
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 21  # 21 car parts
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        
        # Register metadata for visualization
        MetadataCatalog.get("car_parts").set(thing_classes=list(self.CAR_PARTS.values()))
        
        if verbose:
            print(f"Car parts detector initialized!")
            print(f"Device: {cfg.MODEL.DEVICE}")
            print(f"Confidence threshold: {confidence_threshold}")
            print(f"Number of car parts classes: {len(self.CAR_PARTS)}")
            print()
    
    def detect(self, image_path, return_visualization=False):
        """
        Detect car parts in an image
        
        Args:
            image_path: Path to the image file
            return_visualization: If True, also return annotated image
            
        Returns:
            dict: {
                'detected_parts': list of detected car part names,
                'num_parts': number of detected parts,
                'detections': list of detailed detection info,
                'confidence_scores': dict mapping part names to confidence scores,
                'visualization': annotated image (if return_visualization=True)
            }
        """
        if not os.path.exists(image_path):
            return {
                'error': f"Image not found: {image_path}",
                'detected_parts': [],
                'num_parts': 0,
                'detections': []
            }
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {
                    'error': f"Failed to load image: {image_path}",
                    'detected_parts': [],
                    'num_parts': 0,
                    'detections': []
                }
            
            # Run inference
            outputs = self.predictor(img)
            
            # Extract predictions
            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
            
            # Process detections
            detections = []
            detected_parts = []
            confidence_scores = {}
            
            for idx in range(len(classes)):
                class_id = int(classes[idx]) + 1  # Add 1 because classes are 0-indexed
                part_name = self.CAR_PARTS.get(class_id, f"Unknown-{class_id}")
                confidence = float(scores[idx])
                box = boxes[idx].tolist()
                
                detection_info = {
                    'part_name': part_name,
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': box,  # [x1, y1, x2, y2]
                    'bbox_area': (box[2] - box[0]) * (box[3] - box[1])
                }
                
                if masks is not None:
                    detection_info['has_mask'] = True
                    detection_info['mask_area'] = int(masks[idx].sum())
                
                detections.append(detection_info)
                detected_parts.append(part_name)
                
                # Keep highest confidence for each part type
                if part_name not in confidence_scores or confidence > confidence_scores[part_name]:
                    confidence_scores[part_name] = confidence
            
            # Get unique parts (remove duplicates)
            unique_parts = list(set(detected_parts))
            unique_parts.sort()
            
            result = {
                'detected_parts': unique_parts,
                'num_parts': len(unique_parts),
                'total_detections': len(detections),
                'detections': detections,
                'confidence_scores': confidence_scores,
                'image_shape': img.shape
            }
            
            # Optionally create visualization
            if return_visualization:
                from detectron2.utils.visualizer import Visualizer
                
                v = Visualizer(
                    img[:, :, ::-1],
                    MetadataCatalog.get("car_parts"),
                    scale=1.0
                )
                vis = v.draw_instance_predictions(instances)
                result['visualization'] = vis.get_image()
            
            return result
            
        except Exception as e:
            return {
                'error': f"Detection failed: {str(e)}",
                'detected_parts': [],
                'num_parts': 0,
                'detections': []
            }
    
    def detect_and_save_visualization(self, image_path, output_path):
        """
        Detect car parts and save annotated image
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image
            
        Returns:
            dict: Detection results
        """
        result = self.detect(image_path, return_visualization=True)
        
        if 'visualization' in result:
            vis_img = result['visualization']
            cv2.imwrite(output_path, vis_img[:, :, ::-1])
            result['visualization_saved'] = output_path
        
        return result
    
    def get_part_statistics(self, image_path):
        """
        Get statistical information about detected parts
        
        Args:
            image_path: Path to the image
            
        Returns:
            dict: Statistics including coverage, confidence, etc.
        """
        result = self.detect(image_path)
        
        if 'error' in result:
            return result
        
        detections = result['detections']
        image_area = result['image_shape'][0] * result['image_shape'][1]
        
        stats = {
            'unique_parts': result['detected_parts'],
            'num_unique_parts': result['num_parts'],
            'total_detections': result['total_detections'],
            'average_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0,
            'max_confidence': max([d['confidence'] for d in detections]) if detections else 0,
            'min_confidence': min([d['confidence'] for d in detections]) if detections else 0,
            'total_bbox_coverage': sum([d['bbox_area'] for d in detections]) / image_area if detections else 0,
        }
        
        # Add per-part statistics
        part_stats = {}
        for part_name in result['detected_parts']:
            part_detections = [d for d in detections if d['part_name'] == part_name]
            part_stats[part_name] = {
                'count': len(part_detections),
                'avg_confidence': np.mean([d['confidence'] for d in part_detections]),
                'max_confidence': max([d['confidence'] for d in part_detections]),
            }
        
        stats['per_part_stats'] = part_stats
        
        return stats


if __name__ == "__main__":
    # Test the detector
    import sys
    import json
    
    # Model path
    model_path = "./model_final.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)
    
    try:
        # Initialize detector (suppress initialization output)
        detector = CarPartsDetector(
            model_path=model_path,
            confidence_threshold=0.5,
            verbose=False
        )
        
        # Check if test image provided
        if len(sys.argv) > 1:
            test_image = sys.argv[1]
            
            if not os.path.exists(test_image):
                print(f"Image not found: {test_image}")
                sys.exit(1)
            
            # Run detection
            result = detector.detect(test_image, return_visualization=False)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                sys.exit(1)
            
            # Display results
            print(f"\nImage: {os.path.basename(test_image)}")
            print(f"Detected {result['num_parts']} unique car parts:\n")

            if result['detected_parts']:
                for part in result['detected_parts']:
                    conf = result['confidence_scores'][part]
                    count = len([d for d in result['detections'] if d['part_name'] == part])
                    print(f"   {part:20s} {conf:6.1%} ({count}x)")

                # Get statistics
                stats = detector.get_part_statistics(test_image)
                print(f"\nAvg confidence: {stats['average_confidence']:.1%}")

                # Save visualization and JSON
                output_dir = "car_parts_output"
                os.makedirs(output_dir, exist_ok=True)

                base_name = os.path.splitext(os.path.basename(test_image))[0]
                vis_path = os.path.join(output_dir, f"{base_name}_parts_detected.jpg")
                json_path = os.path.join(output_dir, f"{base_name}_results.json")

                detector.detect_and_save_visualization(test_image, vis_path)

                # Save JSON results
                with open(json_path, 'w') as f:
                    json.dump({
                        'image': test_image,
                        'detected_parts': result['detected_parts'],
                        'num_parts': result['num_parts'],
                        'confidence_scores': result['confidence_scores'],
                        'statistics': stats,
                        'detections': result['detections']
                    }, f, indent=2)

                print(f"\nSaved: {vis_path}")
                print(f"Saved: {json_path}\n")
            else:
                print("No car parts detected\n")
            
        else:
            print("\nUsage: python car_parts_detector.py <image_path>")
            print("Output: car_parts_output/<image>_parts_detected.jpg & .json\n")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
