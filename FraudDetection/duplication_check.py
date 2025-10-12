"""
Image Duplication Detection Module
Checks if an image is a duplicate of previously submitted images
"""

import os
import sys
from pathlib import Path
from PIL import Image
import imagehash
import torch
import clip
import numpy as np


class DuplicationDetector:
    def __init__(self, hash_threshold=5, clip_threshold=0.85):
        """
        Initialize duplication detector
        
        Args:
            hash_threshold: Maximum hash difference to consider images same (default: 5)
            clip_threshold: Minimum CLIP similarity to consider images same (default: 0.85)
        """
        self.hash_threshold = hash_threshold
        self.clip_threshold = clip_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = None
        self.clip_preprocess = None
        
        print("Duplication detector initialized!")
        print(f"  Hash threshold: {hash_threshold}")
        print(f"  CLIP threshold: {clip_threshold}")
        print(f"  Device: {self.device}\n")
    
    def _load_clip_model(self):
        """Lazy load CLIP model only when needed"""
        if self.clip_model is None:
            print("  Loading CLIP model for semantic comparison...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            print("  CLIP model loaded!")
    
    def _get_hashes(self, img):
        """Calculate perceptual hashes for an image"""
        return {
            'aHash': imagehash.average_hash(img),
            'pHash': imagehash.phash(img),
            'dHash': imagehash.dhash(img)
        }
    
    def _hash_diff(self, hash1, hash2):
        """Calculate difference between two hashes"""
        return abs(hash1 - hash2)
    
    def _get_clip_embedding(self, img_path):
        """Get CLIP embedding for an image"""
        self._load_clip_model()
        image = self.clip_preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.clip_model.encode_image(image).cpu().numpy().flatten()
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def compare_two_images(self, img1_path, img2_path):
        """
        Compare two images using perceptual hashes first, then CLIP embeddings
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            
        Returns:
            dict: Comparison result
        """
        try:
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
        except Exception as e:
            return {
                'are_same': False,
                'method_used': 'error',
                'hash_scores': {},
                'clip_similarity': None,
                'error': str(e)
            }
        
        # Step 1: Check perceptual hashes (fast)
        hashes1 = self._get_hashes(img1)
        hashes2 = self._get_hashes(img2)
        hash_scores = {}
        
        for k in hashes1:
            hash_scores[k] = self._hash_diff(hashes1[k], hashes2[k])
        
        min_hash_diff = min(hash_scores.values())
        
        # If hash difference is very low, consider images same
        if min_hash_diff <= self.hash_threshold:
            return {
                'are_same': True,
                'method_used': 'hash',
                'hash_scores': hash_scores,
                'min_hash_diff': min_hash_diff,
                'clip_similarity': None
            }
        
        # Step 2: If hashes don't indicate same image, use CLIP embeddings (slower but more accurate)
        try:
            emb1 = self._get_clip_embedding(img1_path)
            emb2 = self._get_clip_embedding(img2_path)
            clip_sim = self._cosine_similarity(emb1, emb2)
            
            are_same = clip_sim >= self.clip_threshold
            
            return {
                'are_same': are_same,
                'method_used': 'clip',
                'hash_scores': hash_scores,
                'min_hash_diff': min_hash_diff,
                'clip_similarity': clip_sim
            }
        except Exception as e:
            return {
                'are_same': False,
                'method_used': 'error',
                'hash_scores': hash_scores,
                'min_hash_diff': min_hash_diff,
                'clip_similarity': None,
                'error': str(e)
            }
    
    def find_duplicates_in_database(self, image_path, database_folder, exclude_folder=None):
        """
        Check if an image is a duplicate of any image in the database
        
        Args:
            image_path: Path to the image to check
            database_folder: Path to folder containing previous submissions
            exclude_folder: Optional folder path to exclude from search (e.g., current submission folder)
            
        Returns:
            dict: {
                'is_duplicate': bool,
                'duplicate_of': str or None (path to matching image),
                'similarity_score': float,
                'method_used': str,
                'total_images_checked': int
            }
        """
        result = {
            'is_duplicate': False,
            'duplicate_of': None,
            'similarity_score': 0.0,
            'method_used': None,
            'total_images_checked': 0,
            'matches': []
        }
        
        if not os.path.exists(database_folder):
            result['error'] = f"Database folder not found: {database_folder}"
            return result
        
        # Normalize exclude_folder path if provided
        exclude_folder_abs = None
        if exclude_folder:
            exclude_folder_abs = os.path.abspath(exclude_folder)
        
        # Find all image files in database (recursively)
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
        database_images = []
        
        for root, dirs, files in os.walk(database_folder):
            # Skip the excluded folder and its subdirectories
            if exclude_folder_abs and os.path.abspath(root).startswith(exclude_folder_abs):
                continue
                
            for file in files:
                if file.lower().endswith(image_extensions):
                    full_path = os.path.join(root, file)
                    # Don't compare with itself
                    if os.path.abspath(full_path) != os.path.abspath(image_path):
                        database_images.append(full_path)
        
        result['total_images_checked'] = len(database_images)
        
        if len(database_images) == 0:
            result['error'] = "No images found in database"
            return result
        
        # Compare with each image in database
        for db_image_path in database_images:
            comparison = self.compare_two_images(image_path, db_image_path)
            
            if comparison['are_same']:
                # Found a duplicate!
                result['is_duplicate'] = True
                result['duplicate_of'] = db_image_path
                result['method_used'] = comparison['method_used']
                
                if comparison['method_used'] == 'hash':
                    result['similarity_score'] = 1.0 - (comparison['min_hash_diff'] / 64.0)
                elif comparison['method_used'] == 'clip':
                    result['similarity_score'] = comparison['clip_similarity']
                
                result['matches'].append({
                    'image_path': db_image_path,
                    'comparison': comparison
                })
                
                # For efficiency, return immediately on first match
                # If you want to find all duplicates, remove this break
                break
        
        return result
    
    def check_for_duplicates(self, image_path, database_folder, exclude_folder=None):
        """
        Simplified interface: Check if image is duplicate
        
        Args:
            image_path: Path to image to check
            database_folder: Path to database folder
            exclude_folder: Optional folder path to exclude from search (e.g., current submission folder)
            
        Returns:
            dict: {
                'is_duplicate': bool,
                'duplicate_info': str (human-readable info),
                'details': dict
            }
        """
        details = self.find_duplicates_in_database(image_path, database_folder, exclude_folder)
        
        info_lines = []
        
        if details['is_duplicate']:
            info_lines.append("DUPLICATE DETECTED!")
            info_lines.append(f"This image matches: {os.path.basename(details['duplicate_of'])}")
            info_lines.append(f"From: {os.path.dirname(details['duplicate_of'])}")
            info_lines.append(f"Similarity: {details['similarity_score']:.2%}")
            info_lines.append(f"Detection method: {details['method_used'].upper()}")
        else:
            info_lines.append("No duplicates found")
            info_lines.append(f"Checked against {details['total_images_checked']} images in database")
        
        return {
            'is_duplicate': details['is_duplicate'],
            'duplicate_info': "\n".join(info_lines),
            'details': details
        }


if __name__ == "__main__":
    # Test
    detector = DuplicationDetector()
    
    if len(sys.argv) >= 3:
        img1 = sys.argv[1]
        img2 = sys.argv[2]
        result = detector.compare_two_images(img1, img2)
        print(f"\nImages are {'SAME' if result['are_same'] else 'DIFFERENT'}")
        print(f"Method: {result['method_used']}")
        if result.get('clip_similarity'):
            print(f"CLIP similarity: {result['clip_similarity']:.4f}")
    else:
        print("Usage: python duplication_check.py <image1> <image2>")
