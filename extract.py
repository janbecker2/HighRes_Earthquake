import json
import os
from PIL import Image

# 1. --- Configuration ---
# Update this to your exact Windows path where the unzipped Roboflow data is
input_base_dir = r"C:\Users\job02\Downloads\damage2.coco-segmentation"

# Where you want the final cropped images to go
output_base_dir = r"C:\Users\job02\Downloads\training_images"

# --- NEW: Minimum Resolution Thresholds (in pixels) ---
# If a building's bounding box is smaller than this, it gets skipped.
# 20 pixels at 0.5m/px = 10 meters (a small house)
MIN_WIDTH = 64
MIN_HEIGHT = 64

# The splits provided by Roboflow
splits = ["train", "valid", "test"]

print(f"Starting extraction (Filtering out buildings smaller than {MIN_WIDTH}x{MIN_HEIGHT} pixels)...")

# Create the two main output folders immediately
os.makedirs(os.path.join(output_base_dir, "undestroyed"), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, "destroyed"), exist_ok=True)

# 2. --- Loop through each split (train, valid, test) ---
for split in splits:
    split_dir = os.path.join(input_base_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"Skipping '{split}' - folder not found.")
        continue
        
    json_path = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.exists(json_path):
        print(f"Skipping '{split}' - no _annotations.coco.json found.")
        continue
        
    print(f"\nProcessing '{split}' split...")
    
    # Load the COCO JSON for this specific split
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
        
    # Map category IDs to names (e.g., 1: 'destroyed', 2: 'no-damage')
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Map image IDs to file names for quick lookup
    images = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # 3. --- Extract and Crop ---
    count = 0
    skipped_size = 0
    
    for i, ann in enumerate(coco_data['annotations']):
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox'] # COCO format: [x_min, y_min, width, height]
        
        width = bbox[2]
        height = bbox[3]
        
        # --- NEW: Size Filter ---
        # Skip invalid bounding boxes OR buildings that are too small
        if width < MIN_WIDTH or height < MIN_HEIGHT:
            skipped_size += 1
            continue
            
        file_name = images[image_id]
        cat_name = categories[category_id]
        
        if cat_name == "destroyed":
            folder_name = "destroyed"
        elif cat_name == "no-damage":
            folder_name = "undestroyed"
        else:
            continue # Überspringt major-damage und minor-damage komplett!
        
        img_path = os.path.join(split_dir, file_name)
        if not os.path.exists(img_path):
            continue
            
        try:
            with Image.open(img_path) as img:
                # Calculate crop coordinates: (left, upper, right, lower)
                left, upper = bbox[0], bbox[1]
                right, lower = bbox[0] + width, bbox[1] + height
                
                cropped_img = img.crop((left, upper, right, lower))
                
                # Safety check: convert to RGB to prevent errors when saving RGBA/Palette as JPEG
                if cropped_img.mode in ("RGBA", "P"):
                    cropped_img = cropped_img.convert("RGB")
                
                # Save the chip (Adding 'split' name to prevent overwriting files with the same name)
                save_name = f"{split}_{file_name.split('.')[0]}_crop_{i}.jpg"
                save_path = os.path.join(output_base_dir, folder_name, save_name)
                cropped_img.save(save_path)
                count += 1
                
        except Exception as e:
            print(f"  Error processing {file_name}: {e}")
            
    print(f"  -> Extracted {count} buildings. (Skipped {skipped_size} for being too small).")

print("\nAll done! Your filtered, combined dataset is ready.")