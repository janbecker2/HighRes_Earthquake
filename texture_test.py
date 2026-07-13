import numpy as np
import rasterio
import matplotlib.pyplot as plt

# -------------------------------------------------
# Bilder laden
# -------------------------------------------------
before_path = "aoi_clips/before_aoi_clip.tif"
after_path = "aoi_clips/after_aoi_clip.tif"

with rasterio.open(before_path) as src:
    before_image = src.read()  # shape: (bands, rows, cols)

with rasterio.open(after_path) as src:
    after_image = src.read()

# RGB-Plot (Band 1,2,3 als R,G,B)
def plot_rgb(img, title=""):
    rgb = img[:3, :, :]  # erste 3 Bänder
    rgb = np.moveaxis(rgb, 0, -1)  # (rows, cols, 3)
    # Skalierung auf 0–1, falls nötig (hier einfach angenommen: passt)
    plt.figure()
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

plot_rgb(before_image, "Before")
plot_rgb(after_image, "After")


# -------------------------------------------------
# Sobel-Operator
# -------------------------------------------------
# Kernel X und Y (wie in R)
KernelX = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)

KernelY = np.array([[ 1, 2, 1],
                    [ 0, 0, 0],
                    [-1,-2,-1]], dtype=float)

print("KernelX:\n", KernelX)
print("KernelY:\n", KernelY)


def apply_sobel(band: np.ndarray):
    """
    band: 2D-Array (rows, cols)
    returns: Sobel-Magnitude (rows, cols)
    """
    from scipy.signal import convolve2d

    # convolve2d mit 'same' Größe, Randbehandlung mit 0 (kann man anpassen)
    SobelX = convolve2d(band, KernelX, mode='same', boundary='fill', fillvalue=0)
    SobelY = convolve2d(band, KernelY, mode='same', boundary='fill', fillvalue=0)

    Sobel = np.sqrt(SobelX**2 + SobelY**2)
    return Sobel, SobelX, SobelY


# Wir nehmen Band 1 (Index 0) wie in R: before_image[[1]]
before_band = before_image[0, :, :].astype(float)
after_band = after_image[0, :, :].astype(float)

before_Sobel, before_SobelX, before_SobelY = apply_sobel(before_band)
after_Sobel, after_SobelX, after_SobelY = apply_sobel(after_band)

def plot_grey(img, title=""):
    img = np.nan_to_num(img, nan=0.0)

    # Auf 0–1 normalisieren
    vmin = img.min()
    vmax = img.max()
    if vmax > vmin:
        img_norm = (img - vmin) / (vmax - vmin)
    else:
        img_norm = np.zeros_like(img)

    plt.figure()
    plt.imshow(img_norm, cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

plot_grey(before_Sobel, "Before Sobel")
plot_grey(after_Sobel, "After Sobel")


# -------------------------------------------------
# Einfache Change Analysis
# -------------------------------------------------
result_change = np.abs(after_Sobel - before_Sobel)

plot_grey(result_change, "Change analysis")

# Schwellenwert: mean + sd (wie in R)
mean_val = np.nanmean(result_change)
sd_val = np.nanstd(result_change)
result_thresh = mean_val + sd_val

result_change_mask = result_change > result_thresh

plt.figure()
plt.imshow(result_change_mask, cmap='gray')
plt.title("Change mask")
plt.axis("off")
plt.tight_layout()

plt.show()