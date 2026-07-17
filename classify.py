"""
Standalone inference script (PIXEL-WISE) with Building Masking:
- Loads the saved VGG16 model
- Loads Microsoft Building Footprints and creates a binary mask
- Classifies raster pixel-by-pixel, SKIPPING areas with no buildings
- Saves a georeferenced GeoTIFF
- Plots the classification result
"""

# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window
from rasterio.features import rasterize

import geopandas as gpd
import tensorflow as tf

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH       = r"C:\Users\job02\Downloads\vgg16_destroyed.keras"
INFER_RASTER     = r"C:\Users\job02\Downloads\aoi_clips\aoi_clips\after_aoi_clip.tif"
OUTPUT_RASTER    = r"C:\Users\job02\Downloads\aoi_clips\aoi_clips\classification.tif"

# ADD YOUR BUILDING FOOTPRINTS FILE HERE (GeoJSON, Shapefile, or GeoPackage)
BUILDINGS_VECTOR = r"C:\Users\job02\Downloads\ms_buildings_pre_earthquake.geojson"

CHIP_SIZE = 128          
THRESHOLD = 0.5          
STRIDE    = 16           
MANUAL_CRS = "EPSG:32637"

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# ============================================================
# PREPARE DATA & MASK
# ============================================================
print("\nPreparing building mask...")
with rasterio.open(INFER_RASTER) as src:
    width, height = src.width, src.height
    profile = src.profile.copy()
    raster_crs = src.crs or CRS.from_user_input(MANUAL_CRS)
    profile.update(count=1, dtype="uint8", nodata=255, crs=raster_crs)

    # Load buildings
    buildings_gdf = gpd.read_file(BUILDINGS_VECTOR)
    
    # Ensure CRS matches the raster
    if buildings_gdf.crs != raster_crs:
        print(f"Reprojecting buildings from {buildings_gdf.crs} to {raster_crs}...")
        buildings_gdf = buildings_gdf.to_crs(raster_crs)
    
    # Rasterize building footprints (1 for building, 0 for background)
    shapes = ((geom, 1) for geom in buildings_gdf.geometry if geom is not None)
    building_mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=src.transform,
        fill=0,
        dtype="uint8"
    )

# ============================================================
# PIXEL-WISE INFERENCE (Masked)
# ============================================================
print("\nStarting masked pixel-wise classification...")

with rasterio.open(INFER_RASTER) as src:
    prob_sum = np.zeros((height, width), dtype="float32")
    count    = np.zeros((height, width), dtype="float32")

    tiles, coords = [], []
    skipped_windows = 0

    for r0 in range(0, height - CHIP_SIZE + 1, STRIDE):
        for c0 in range(0, width - CHIP_SIZE + 1, STRIDE):
            
            # Check building mask first to save processing time
            mask_chip = building_mask[r0:r0 + CHIP_SIZE, c0:c0 + CHIP_SIZE]
            if mask_chip.max() == 0:
                skipped_windows += 1
                continue # Skip windows completely devoid of buildings

            window = Window(c0, r0, CHIP_SIZE, CHIP_SIZE)
            chip = src.read([1, 2, 3], window=window).astype("float32")
            chip = np.transpose(chip, (1, 2, 0))
            if chip.max() > 1.0:
                chip = chip / 255.0
            
            tiles.append(chip)
            coords.append((r0, c0))

    print(f"Total windows to process: {len(tiles)} (Skipped {skipped_windows} empty windows)")

    BATCH = 64
    all_probs = []
    for i in range(0, len(tiles), BATCH):
        batch = np.array(tiles[i:i + BATCH], dtype="float32")
        p = model.predict(batch, verbose=0).ravel()
        all_probs.extend(p)
        if (i // BATCH) % 20 == 0:
            print(f"  processed {min(i + BATCH, len(tiles))}/{len(tiles)} windows")

    # accumulate probabilities
    for (r0, c0), p in zip(coords, all_probs):
        prob_sum[r0:r0 + CHIP_SIZE, c0:c0 + CHIP_SIZE] += p
        count[r0:r0 + CHIP_SIZE, c0:c0 + CHIP_SIZE]    += 1.0

    # Average valid predictions
    valid = count > 0
    prob_map = np.full((height, width), np.nan, dtype="float32")
    prob_map[valid] = prob_sum[valid] / count[valid]

    # Enforce precise pixel-level building mask (clip edges of overlapping windows)
    no_building = (building_mask == 0)
    prob_map[no_building] = np.nan

    # Threshold -> hard class map
    pred_map = np.full((height, width), 255, dtype="uint8")
    valid_buildings = (valid) & (~no_building)
    pred_map[valid_buildings] = (prob_map[valid_buildings] >= THRESHOLD).astype("uint8")

print("Classification complete.")

# ============================================================
# WRITE RESULT
# ============================================================
with rasterio.open(OUTPUT_RASTER, "w", **profile) as dst:
    dst.write(pred_map, 1)
print(f"Saved masked classification map to:\n{OUTPUT_RASTER}")

# ============================================================
# PLOT RESULT
# ============================================================
plot_class = np.ma.masked_equal(pred_map, 255)
plot_prob  = np.ma.masked_invalid(prob_map)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

with rasterio.open(INFER_RASTER) as src:
    rgb = src.read([1, 2, 3]).astype("float32")
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.transpose(rgb, (1, 2, 0))

axes[0].imshow(rgb)
axes[0].set_title("Input image (RGB)")
axes[0].axis("off")

im1 = axes[1].imshow(plot_prob, cmap="RdYlGn_r", vmin=0, vmax=1)
axes[1].set_title("Probability of 'destroyed' (Buildings Only)")
axes[1].axis("off")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(plot_class, cmap="RdYlGn_r", vmin=0, vmax=1)
axes[2].set_title(f"Classification (Buildings Only, thr={THRESHOLD})")
axes[2].axis("off")
cbar = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=[0, 1])
cbar.ax.set_yticklabels(["Not destroyed", "Destroyed"])

plt.tight_layout()
plt.show()

# --- Print class distribution ---
unique, counts = np.unique(pred_map[pred_map != 255], return_counts=True)
print("\nBuilding status distribution:")
for u, c in zip(unique, counts):
    name = "Destroyed" if u == 1 else "Not destroyed"
    print(f"  {name} ({u}): {c} pixels")