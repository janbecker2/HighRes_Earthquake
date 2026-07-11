import matplotlib.pyplot as plt
import rasterio
import numpy as np
from skimage.feature import graycomatrix, graycoprops

RASTER = r"C:\Users\job02\Downloads\aoi_clips\aoi_clips\after_aoi_clip.tif"
WIN = 15
STEP = 2
LEVELS = 32
PROP = 'energy'   # <- change to 'dissimilarity', 'correlation', 'ASM', etc.

with rasterio.open(RASTER) as src:
    nodata = src.nodata
    rgb = src.read([1, 2, 3]).astype(np.float32)

gray = rgb.mean(axis=0)
mask = rgb[0] == nodata
gray = np.nan_to_num(np.where(mask, 0, gray))
gray_u8 = np.clip(gray / 255 * (LEVELS - 1), 0, LEVELS - 1).astype(np.uint8)

h, w = gray_u8.shape
out_h = (h - WIN) // STEP + 1
out_w = (w - WIN) // STEP + 1

prop_map = np.full((out_h, out_w), np.nan, dtype=np.float32)
homogeneity = np.full((out_h, out_w), np.nan, dtype=np.float32)

for i in range(out_h):
    for j in range(out_w):
        y, x = i * STEP, j * STEP
        win = gray_u8[y:y+WIN, x:x+WIN]
        if mask[y:y+WIN, x:x+WIN].any():
            continue
        glcm = graycomatrix(win, distances=[1], angles=[0], levels=LEVELS, symmetric=True, normed=True)
        prop_map[i, j] = graycoprops(glcm, PROP)[0, 0]
        homogeneity[i, j] = graycoprops(glcm, 'homogeneity')[0, 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
im0 = axes[0].imshow(prop_map, cmap='inferno')
axes[0].set_title(PROP.capitalize(), fontweight='bold')
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(homogeneity, cmap='viridis')
axes[1].set_title('Homogeneity', fontweight='bold')
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046)

plt.tight_layout()
plt.show()