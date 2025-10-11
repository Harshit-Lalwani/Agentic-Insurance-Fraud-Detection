from cmp_img import are_images_similar

# Basic usage
result = are_images_similar("image1.jpg", "image2.jpg")
print(f"Are same: {result['are_same']}")
print(f"Method used: {result['method_used']}")

# With custom thresholds and verbose output
result = are_images_similar(
    "image1.jpg", "image2.jpg", 
    hash_threshold=3, 
    clip_threshold=0.9, 
    verbose=True
)

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