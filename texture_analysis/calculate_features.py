# preprocess.py
from pathlib import Path
import numpy as np
import rasterio
from skimage.transform import resize
from concurrent.futures import ProcessPoolExecutor

from data_prep import load_image
from glcm import compute_glcm_texture
from sobel import compute_sobel_texture

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
BAND_ORDER = ("glcm_energy", "glcm_homogeneity", "sobel_magnitude")

# ---------------------------------------------------------------
# Texture computation
# ---------------------------------------------------------------
def compute_textures(img, nodata=None):
    """Return dict of texture arrays for one image."""
    glcm_result = compute_glcm_texture(img, nodata=nodata)
    sobel_result = compute_sobel_texture(img, nodata=nodata)
    return {
        "glcm_energy": glcm_result["energy"],
        "glcm_homogeneity": glcm_result["homogeneity"],
        "sobel_magnitude": sobel_result["magnitude"],
    }

# ---------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------
def rgb_to_uint8(arr):
    """Keep original RGB bands as 0-255."""
    arr = np.nan_to_num(arr.astype("float32"), nan=0.0)
    return np.clip(arr, 0, 255).astype("uint8")

def scale_local_uint8(arr):
    """Stretch texture band using LOCAL (per-image) min/max -> uint8 [0,255]."""
    arr = np.nan_to_num(arr.astype("float32"), nan=0.0)
    vmin, vmax = np.min(arr), np.max(arr)
    
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr)
        
    return np.clip(arr * 255.0, 0, 255).astype("uint8")

# ---------------------------------------------------------------
# STEP 2: build one 6-band uint8 stack (Uses local scaling now)
# ---------------------------------------------------------------
def make_stack(image_path, out_path, include_original=True):
    img, profile = load_image(Path(image_path))
    ref_shape = img.shape[-2:]

    tex = compute_textures(img, nodata=profile.get("nodata"))

    bands, band_names = [], []

    # 1) Original image bands (assumes standard 0-255 image data)
    if include_original:
        if img.ndim == 3:
            for b in range(img.shape[0]):
                bands.append(rgb_to_uint8(img[b]))
                band_names.append(f"band_{b + 1}")
        else:
            bands.append(rgb_to_uint8(img))
            band_names.append("band_1")

    # 2) Texture bands -- upsample to full res, scale LOCALLY to 0-255
    for k in BAND_ORDER:
        arr = tex[k]
        if arr.shape != tuple(ref_shape):
            arr = resize(arr, ref_shape, order=1, preserve_range=True, anti_aliasing=False)
        
        # Scale locally based on this specific image's min and max
        bands.append(scale_local_uint8(arr))
        band_names.append(k)

    stacked = np.stack(bands, axis=0)

    # --- UPDATED PROFILE LOGIC ---
    # Start fresh to avoid JPEG/TIFF driver compatibility crashes
    prof = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": stacked.shape[0],
        "height": ref_shape[0],
        "width": ref_shape[1],
        "nodata": None,
    }
    
    # Safely copy over geospatial data ONLY if it exists (for TIFs)
    if "crs" in profile and profile["crs"]:
        prof["crs"] = profile["crs"]
    if "transform" in profile and profile["transform"]:
        prof["transform"] = profile["transform"]
    # -----------------------------

    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(stacked)
        for i, bname in enumerate(band_names, start=1):
            dst.set_band_description(i, bname)

    return stacked

# ---------------------------------------------------------------
# Multiprocessing Wrapper 
# ---------------------------------------------------------------
def process_single_file(args):
    p, out_dir = args
    
    # Include the original extension in the stem to prevent collisions
    # e.g., image.jpg -> image_jpg_stack.tif
    safe_stem = p.name.replace('.', '_')
    out_p = out_dir / f"{safe_stem}_stack.tif"
    temp_p = out_dir / f"{safe_stem}_temp.tif"

    # Skip if the final file already exists (allows you to pause/resume)
    if out_p.exists():
        return

    try:
        # rasterio.Env() is crucial for stable multiprocessing in GDAL
        with rasterio.Env():
            make_stack(p, temp_p)
        
        # Atomic rename: this guarantees no corrupted files are ever finalized
        temp_p.rename(out_p)
        print(f"Saved: {out_p.name}")
        
    except Exception as e:
        print(f"FAILED {p.name}: {e}")
        # Clean up the broken temporary file
        if temp_p.exists():
            temp_p.unlink()

# ---------------------------------------------------------------
# MAIN ROUTINE
# ---------------------------------------------------------------
if __name__ == "__main__":

    IN_DIR = Path(r"C:\Users\job02\Downloads\training_images\damaged")         # input folder
    OUT_DIR = Path(r"C:\Users\job02\Downloads\training_images_textures\damaged") # new output folder
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find ALL TIFs and JPGs (case-insensitive)
    image_files = list(
        set(IN_DIR.glob("*.jpg")) |
        set(IN_DIR.glob("*.jpeg")) |
        set(IN_DIR.glob("*.JPG")) |
        set(IN_DIR.glob("*.JPEG")) |
        set(IN_DIR.glob("*.tif")) |
        set(IN_DIR.glob("*.tiff")) |
        set(IN_DIR.glob("*.TIF"))
    )

    if not image_files:
        print(f"\nNo images found in {IN_DIR}")
    else:
        print(f"\nFound {len(image_files)} images in {IN_DIR}")
        print(f"Building stacks using multiprocessing...")
        
        # Package the arguments (No more stats variable to pass!)
        tasks = [(p, OUT_DIR) for p in image_files]

        # Process all files in parallel
        with ProcessPoolExecutor() as executor:
            executor.map(process_single_file, tasks)

        print(f"\nDone. All stacks saved to {OUT_DIR}")

    # TEST_FILE = Path(r"C:\Users\job02\Downloads\study1.tif") 
    
    # # 2. Where to save the test output
    # OUT_DIR = Path(r"C:\Users\job02\Downloads\study1_textures.tif")
    # OUT_DIR.mkdir(parents=True, exist_ok=True)

    # out_p = OUT_DIR / f"{TEST_FILE.stem}_test_stack.tif"

    # print(f"\n--- Testing: {TEST_FILE.name} ---")
    
    # if not TEST_FILE.exists():
    #     print(f"  -> File not found! Check the path: {TEST_FILE}")
    # else:
    #     try:
    #         # 3. Process the single file
    #         stacked = make_stack(TEST_FILE, out_p)
            
    #         # 4. Print the results to verify it worked and isn't just zeros
    #         print(f"  -> SUCCESS! Saved to: {out_p.name}")
    #         print(f"  -> Stack Shape: {stacked.shape}")
    #         print(f"  -> Data Type: {stacked.dtype}")
    #         print(f"  -> Min/Max Values: {stacked.min()} / {stacked.max()}")
            
    #     except Exception as e:
    #         print(f"  -> FAILED: {e}")