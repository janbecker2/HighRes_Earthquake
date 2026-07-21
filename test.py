# test_workflow.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from data_prep import load_image
from glcm import compute_glcm_texture
from sobel import compute_sobel_texture

TEST_FILE = Path("aoi_clips/after_aoi_clip.tif")

img, profile = load_image(TEST_FILE)
print("Shape:", img.shape)
print("Dtype:", img.dtype)
print("Nodata:", profile.get("nodata"))

glcm_result = compute_glcm_texture(img, nodata=profile.get("nodata"))
print("GLCM keys:", glcm_result.keys())
for k, v in glcm_result.items():
    print(f"{k}: shape={v.shape}, min={np.nanmin(v):.4f}, max={np.nanmax(v):.4f}")

sobel_result = compute_sobel_texture(img, nodata=profile.get("nodata"))
print("Sobel keys:", sobel_result.keys())
print("Magnitude shape:", sobel_result["magnitude"].shape)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(glcm_result["energy"], cmap="inferno")
axes[0].set_title("GLCM Energy")
axes[1].imshow(glcm_result["homogeneity"], cmap="viridis")
axes[1].set_title("GLCM Homogeneity")
axes[2].imshow(sobel_result["magnitude"], cmap="gray")
axes[2].set_title("Sobel Magnitude")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()