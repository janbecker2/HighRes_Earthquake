import json
import os
from PIL import Image

# extracting data to match our training data format


input_base_dir = r"C:\Users\job02\Downloads\damage2.coco-segmentation"

output_base_dir = r"C:\Users\job02\Downloads\training_images"


MIN_WIDTH = 64
MIN_HEIGHT = 64

splits = ["train", "valid", "test"]

print(f"Starting extraction")

os.makedirs(os.path.join(output_base_dir, "undestroyed"), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, "destroyed"), exist_ok=True)

# iterate through each split (train, valid, test)
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
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
        
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    images = {img['id']: img['file_name'] for img in coco_data['images']}
    
    count = 0
    skipped_size = 0
    
    for i, ann in enumerate(coco_data['annotations']):
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox'] # COCO format: [x_min, y_min, width, height]
        
        width = bbox[2]
        height = bbox[3]

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
            continue 
        
        img_path = os.path.join(split_dir, file_name)
        if not os.path.exists(img_path):
            continue
            
        try:
            with Image.open(img_path) as img:
                left, upper = bbox[0], bbox[1]
                right, lower = bbox[0] + width, bbox[1] + height
                
                cropped_img = img.crop((left, upper, right, lower))
                
                if cropped_img.mode in ("RGBA", "P"):
                    cropped_img = cropped_img.convert("RGB")
                
                save_name = f"{split}_{file_name.split('.')[0]}_crop_{i}.jpg"
                save_path = os.path.join(output_base_dir, folder_name, save_name)
                cropped_img.save(save_path)
                count += 1
                
        except Exception as e:
            print(f"  Error processing {file_name}: {e}")
            
    print(f"  -> Extracted {count} buildings. (Skipped {skipped_size} for being too small).")

