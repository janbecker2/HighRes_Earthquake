"""
Standalone inference script (PIXEL-WISE):
- Loads the saved VGG16 model
- Classifies a raster pixel-by-pixel using overlapping windows + averaging
- Saves a georeferenced GeoTIFF
- Plots the classification result
"""

# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

import rasterio
from rasterio.windows import Window

import tensorflow as tf

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH    = r"C:\Users\job02\Downloads\vgg16_destroyed.keras"
INFER_RASTER  = r"C:\Users\job02\Downloads\aoi_clips\aoi_clips\after_aoi_clip.tif"
OUTPUT_RASTER = r"C:\Users\job02\Downloads\aoi_clips\aoi_clips\classification.tif"

CHIP_SIZE = 128          # must match training
THRESHOLD = 0.5          # sigmoid cutoff for class 1

# Pixel-wise smoothness (smaller = smoother & slower):
#   32 = quick test, 16 = good, 8 = very smooth, 1 = true per-pixel (VERY slow)
STRIDE = 16


# ============================================================
# LOAD MODEL
# ============================================================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")


# ============================================================
# PIXEL-WISE INFERENCE (overlapping windows + averaging)
# ============================================================
# Each window -> ONE probability, added to EVERY pixel in that window.
# Overlapping windows are averaged so every pixel gets a smooth,
# per-pixel probability instead of a blocky tile label.
print("\nStarting pixel-wise classification...")

with rasterio.open(INFER_RASTER) as src:
    width, height = src.width, src.height
    profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=255)

    # accumulate summed probabilities and hit-counts per pixel
    prob_sum = np.zeros((height, width), dtype="float32")
    count    = np.zeros((height, width), dtype="float32")

    # collect all windows
    tiles, coords = [], []
    for r0 in range(0, height - CHIP_SIZE + 1, STRIDE):
        for c0 in range(0, width - CHIP_SIZE + 1, STRIDE):
            window = Window(c0, r0, CHIP_SIZE, CHIP_SIZE)
            chip = src.read([1, 2, 3], window=window).astype("float32")
            chip = np.transpose(chip, (1, 2, 0))
            if chip.max() > 1.0:
                chip = chip / 255.0
            tiles.append(chip)
            coords.append((r0, c0))

    print(f"Total windows: {len(tiles)} (STRIDE={STRIDE})")

    # predict in batches to save memory
    BATCH = 64
    all_probs = []
    for i in range(0, len(tiles), BATCH):
        batch = np.array(tiles[i:i + BATCH], dtype="float32")
        p = model.predict(batch, verbose=0).ravel()
        all_probs.extend(p)
        if (i // BATCH) % 20 == 0:
            print(f"  processed {min(i + BATCH, len(tiles))}/{len(tiles)} windows")

    # accumulate probabilities into the map
    for (r0, c0), p in zip(coords, all_probs):
        prob_sum[r0:r0 + CHIP_SIZE, c0:c0 + CHIP_SIZE] += p
        count[r0:r0 + CHIP_SIZE, c0:c0 + CHIP_SIZE]    += 1.0

    # average (avoid divide-by-zero)
    valid = count > 0
    prob_map = np.full((height, width), np.nan, dtype="float32")
    prob_map[valid] = prob_sum[valid] / count[valid]

    # threshold -> hard class map
    pred_map = np.full((height, width), 255, dtype="uint8")
    pred_map[valid] = (prob_map[valid] >= THRESHOLD).astype("uint8")

print("Classification complete.")


# ============================================================
# WRITE RESULT
# ============================================================
with rasterio.open(OUTPUT_RASTER, "w", **profile) as dst:
    dst.write(pred_map, 1)
print(f"Saved classification map to:\n{OUTPUT_RASTER}")


# ============================================================
# PLOT RESULT
# ============================================================
plot_class = np.ma.masked_equal(pred_map, 255)
plot_prob  = np.ma.masked_invalid(prob_map)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# --- Left: original RGB image ---
with rasterio.open(INFER_RASTER) as src:
    rgb = src.read([1, 2, 3]).astype("float32")
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.transpose(rgb, (1, 2, 0))
axes[0].imshow(rgb)
axes[0].set_title("Input image (RGB)")
axes[0].axis("off")

# --- Middle: probability map ---
im1 = axes[1].imshow(plot_prob, cmap="RdYlGn_r", vmin=0, vmax=1)
axes[1].set_title("Probability of 'destroyed'")
axes[1].axis("off")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# --- Right: hard classification ---
im2 = axes[2].imshow(plot_class, cmap="RdYlGn_r", vmin=0, vmax=1)
axes[2].set_title(f"Classification (thr={THRESHOLD})")
axes[2].axis("off")
cbar = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=[0, 1])
cbar.ax.set_yticklabels(["Not destroyed", "Destroyed"])

plt.tight_layout()
plt.show()


# --- Print class distribution ---
unique, counts = np.unique(pred_map[pred_map != 255], return_counts=True)
print("\nClass distribution:")
for u, c in zip(unique, counts):
    name = "Destroyed" if u == 1 else "Not destroyed"
    print(f"  {name} ({u}): {c} pixels")