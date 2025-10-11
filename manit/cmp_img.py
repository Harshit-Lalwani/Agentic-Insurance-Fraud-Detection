import sys
from PIL import Image
import imagehash
import torch
import clip
from torchvision import transforms
import numpy as np

def get_hashes(img):
    return {
        'aHash': imagehash.average_hash(img),
        'pHash': imagehash.phash(img),
        'dHash': imagehash.dhash(img)
    }

def hash_diff(hash1, hash2):
    return abs(hash1 - hash2)

def get_clip_embedding(img_path, model, preprocess, device):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(image).cpu().numpy().flatten()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def are_images_similar(img1_path, img2_path, hash_threshold=5, clip_threshold=0.85, verbose=False):
    """
    Compare two images using perceptual hashes first, then CLIP embeddings if needed.
    
    Args:
        img1_path (str): Path to first image
        img2_path (str): Path to second image
        hash_threshold (int): Maximum hash difference to consider images same (default: 5)
        clip_threshold (float): Minimum CLIP similarity to consider images same (default: 0.85)
        verbose (bool): Print detailed comparison results
    
    Returns:
        dict: {
            'are_same': bool,
            'method_used': str ('hash' or 'clip'),
            'hash_scores': dict,
            'clip_similarity': float or None
        }
    """
    try:
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
    except Exception as e:
        if verbose:
            print(f"Error opening images: {e}")
        return {
            'are_same': False,
            'method_used': 'error',
            'hash_scores': {},
            'clip_similarity': None,
            'error': str(e)
        }

    # Step 1: Check perceptual hashes
    hashes1 = get_hashes(img1)
    hashes2 = get_hashes(img2)
    hash_scores = {}
    
    for k in hashes1:
        hash_scores[k] = hash_diff(hashes1[k], hashes2[k])
    
    # Check if any hash indicates very similar images
    min_hash_diff = min(hash_scores.values())
    
    if verbose:
        print("Perceptual Hash Differences:")
        for k, v in hash_scores.items():
            print(f"  {k}: {v}")
        print(f"Minimum hash difference: {min_hash_diff}")
    
    # If hash difference is very low, consider images same
    if min_hash_diff <= hash_threshold:
        if verbose:
            print(f"Images considered SAME based on hash (min diff: {min_hash_diff} <= {hash_threshold})")
        return {
            'are_same': True,
            'method_used': 'hash',
            'hash_scores': hash_scores,
            'clip_similarity': None
        }
    
    # Step 2: If hashes don't indicate same image, use CLIP embeddings
    if verbose:
        print(f"Hash differences too high (min: {min_hash_diff} > {hash_threshold}), checking CLIP embeddings...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        emb1 = get_clip_embedding(img1_path, model, preprocess, device)
        emb2 = get_clip_embedding(img2_path, model, preprocess, device)
        clip_sim = cosine_similarity(emb1, emb2)
        
        if verbose:
            print(f"CLIP Cosine Similarity: {clip_sim:.4f}")
        
        are_same = clip_sim >= clip_threshold
        
        if verbose:
            print(f"Images considered {'SAME' if are_same else 'DIFFERENT'} based on CLIP (similarity: {clip_sim:.4f} {'≥' if are_same else '<'} {clip_threshold})")
        
        return {
            'are_same': are_same,
            'method_used': 'clip',
            'hash_scores': hash_scores,
            'clip_similarity': clip_sim
        }
        
    except Exception as e:
        if verbose:
            print(f"Error with CLIP embeddings: {e}")
        return {
            'are_same': False,
            'method_used': 'error',
            'hash_scores': hash_scores,
            'clip_similarity': None,
            'error': str(e)
        }

def main(img1_path, img2_path):
    """Command line interface for image comparison"""
    result = are_images_similar(img1_path, img2_path, verbose=True)
    
    print("\n" + "="*50)
    print("FINAL VERDICT:")
    print(f"Images are {'SAME' if result['are_same'] else 'DIFFERENT'}")
    print(f"Method used: {result['method_used'].upper()}")
    if 'error' in result:
        print(f"Error: {result['error']}")
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python cmp_img.py <image1> <image2>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])