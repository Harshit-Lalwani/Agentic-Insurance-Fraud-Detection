import os
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from cmp_img import are_images_similar
import tempfile
import shutil

def apply_crop_resize(img, severity=0.1):
    """Apply slight crop and resize"""
    width, height = img.size
    crop_amount = int(min(width, height) * severity)
    
    # Crop from random corner
    left = random.randint(0, crop_amount)
    top = random.randint(0, crop_amount)
    right = width - random.randint(0, crop_amount)
    bottom = height - random.randint(0, crop_amount)
    
    cropped = img.crop((left, top, right, bottom))
    # Resize back to original size
    return cropped.resize((width, height), Image.LANCZOS)

def apply_color_brightness_tweak(img, brightness_factor=None, color_factor=None):
    """Apply color and brightness adjustments"""
    if brightness_factor is None:
        brightness_factor = random.uniform(0.8, 1.2)
    if color_factor is None:
        color_factor = random.uniform(0.8, 1.2)
    
    # Brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    
    # Color saturation
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(color_factor)
    
    return img

def apply_blur_noise(img, blur_radius=1, noise_factor=10):
    """Apply Gaussian blur and noise"""
    # Convert to numpy array
    img_array = np.array(img)
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_factor, img_array.shape).astype(np.uint8)
    noisy_img = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Apply Gaussian blur
    if len(noisy_img.shape) == 3:
        blurred = cv2.GaussianBlur(noisy_img, (blur_radius*2+1, blur_radius*2+1), 0)
    else:
        blurred = cv2.GaussianBlur(noisy_img, (blur_radius*2+1, blur_radius*2+1), 0)
    
    return Image.fromarray(blurred)

def apply_flip_rotate(img):
    """Apply random flip or rotation"""
    operations = [
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
        lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
        lambda x: x.rotate(random.uniform(-5, 5), expand=False),
        lambda x: x.rotate(90),
        lambda x: x.rotate(180),
        lambda x: x.rotate(270)
    ]
    
    operation = random.choice(operations)
    return operation(img)

def apply_text_overlay(img, text="WATERMARK"):
    """Add text overlay/watermark"""
    # Create a copy to draw on
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to use a default font, fall back to default if not available
    try:
        font_size = max(20, min(img.size) // 20)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Random position
    width, height = img.size
    text_width = len(text) * 10  # Rough estimate
    x = random.randint(0, max(1, width - text_width))
    y = random.randint(0, max(1, height - 30))
    
    # Semi-transparent text
    draw.text((x, y), text, fill=(255, 255, 255, 128), font=font)
    
    return img_copy

def apply_recompression(img, quality=None):
    """Apply JPEG recompression"""
    if quality is None:
        quality = random.randint(40, 80)
    
    # Save to temporary file with compression
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        img.save(tmp_file.name, 'JPEG', quality=quality)
        compressed_img = Image.open(tmp_file.name)
        compressed_img.load()  # Load the image data
        os.unlink(tmp_file.name)  # Clean up temp file
        return compressed_img

def test_augmentation_detection(image_folder, num_tests=10):
    """Test the image comparison script with various augmentations"""
    
    # Get list of images
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) == 0:
        print("No images found in the folder!")
        return
    
    # Augmentation functions with descriptions
    augmentations = [
        ("Crop/Resize", apply_crop_resize),
        ("Color/Brightness", apply_color_brightness_tweak),
        ("Blur/Noise", apply_blur_noise),
        ("Flip/Rotate", apply_flip_rotate),
        ("Text Overlay", apply_text_overlay),
        ("Recompression", apply_recompression)
    ]
    
    results = {name: {"hash_detected": 0, "clip_detected": 0, "total": 0} for name, _ in augmentations}
    overall_results = {"hash_detected": 0, "clip_detected": 0, "total": 0}
    
    print("="*80)
    print("TESTING IMAGE COMPARISON WITH AUGMENTATIONS")
    print("="*80)
    
    # Test each augmentation type
    for aug_name, aug_func in augmentations:
        print(f"\nTesting {aug_name}...")
        print("-" * 40)
        
        for i in range(min(num_tests, len(image_files))):
            # Select random image
            img_file = random.choice(image_files)
            img_path = os.path.join(image_folder, img_file)
            
            try:
                # Load and apply augmentation
                original_img = Image.open(img_path)
                if original_img.mode != 'RGB':
                    original_img = original_img.convert('RGB')
                
                augmented_img = aug_func(original_img)
                
                # Save augmented image temporarily
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    augmented_img.save(tmp_file.name, 'JPEG')
                    augmented_path = tmp_file.name
                
                # Test comparison
                result = are_images_similar(img_path, augmented_path, verbose=False)
                
                results[aug_name]["total"] += 1
                overall_results["total"] += 1
                
                if result["are_same"]:
                    if result["method_used"] == "hash":
                        results[aug_name]["hash_detected"] += 1
                        overall_results["hash_detected"] += 1
                        status = "✓ HASH"
                    elif result["method_used"] == "clip":
                        results[aug_name]["clip_detected"] += 1
                        overall_results["clip_detected"] += 1
                        status = "✓ CLIP"
                    else:
                        status = "✗ ERROR"
                else:
                    status = "✗ MISSED"
                
                print(f"  Test {i+1}: {os.path.basename(img_file)} -> {status}")
                
                # Clean up
                os.unlink(augmented_path)
                
            except Exception as e:
                print(f"  Test {i+1}: ERROR - {e}")
                continue
    
    # Print summary
    print("\n" + "="*80)
    print("DETECTION RESULTS SUMMARY")
    print("="*80)
    
    for aug_name in results:
        total = results[aug_name]["total"]
        hash_det = results[aug_name]["hash_detected"]
        clip_det = results[aug_name]["clip_detected"]
        
        if total > 0:
            hash_rate = (hash_det / total) * 100
            clip_rate = (clip_det / total) * 100
            total_rate = ((hash_det + clip_det) / total) * 100
            
            print(f"{aug_name:15} | Hash: {hash_det:2}/{total} ({hash_rate:5.1f}%) | "
                  f"CLIP: {clip_det:2}/{total} ({clip_rate:5.1f}%) | "
                  f"Total: {hash_det + clip_det:2}/{total} ({total_rate:5.1f}%)")
    
    # Overall score
    total = overall_results["total"]
    hash_det = overall_results["hash_detected"]
    clip_det = overall_results["clip_detected"]
    
    if total > 0:
        hash_rate = (hash_det / total) * 100
        clip_rate = (clip_det / total) * 100
        total_rate = ((hash_det + clip_det) / total) * 100
        
        print("-" * 80)
        print(f"{'OVERALL':15} | Hash: {hash_det:2}/{total} ({hash_rate:5.1f}%) | "
              f"CLIP: {clip_det:2}/{total} ({clip_rate:5.1f}%) | "
              f"Total: {hash_det + clip_det:2}/{total} ({total_rate:5.1f}%)")
        
        print("\n" + "="*80)
        print("FINAL SCORE")
        print("="*80)
        print(f"Overall Detection Rate: {total_rate:.1f}%")
        
        if total_rate >= 90:
            grade = "A+ (Excellent)"
        elif total_rate >= 80:
            grade = "A (Very Good)"
        elif total_rate >= 70:
            grade = "B (Good)"
        elif total_rate >= 60:
            grade = "C (Fair)"
        else:
            grade = "D (Needs Improvement)"
        
        print(f"Performance Grade: {grade}")
        print(f"Hash method efficiency: {hash_rate:.1f}% (faster method)")
        print(f"CLIP method usage: {clip_rate:.1f}% (slower but more accurate)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python test_augmentation.py <image_folder>")
        sys.exit(1)
    
    image_folder = sys.argv[1]
    if not os.path.isdir(image_folder):
        print(f"Error: {image_folder} is not a valid directory")
        sys.exit(1)
    
    test_augmentation_detection(image_folder, num_tests=5)  # Test 5 images per augmentation type