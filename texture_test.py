import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# -------------------------------------------------
# Bilder laden
# -------------------------------------------------
before_path = "aoi_clips/before_aoi_clip.tif"
after_path = "aoi_clips/after_aoi_clip.tif"

with rasterio.open(before_path) as src:
    before_image = src.read()  # (bands, rows, cols)
    before_nodata = src.nodata  # vermutlich -2147483648

with rasterio.open(after_path) as src:
    after_image = src.read()
    after_nodata = src.nodata

print("before_nodata:", before_nodata)
print("after_nodata:", after_nodata)

# -------------------------------------------------
# NoData maskieren und Band auf float skalieren
# -------------------------------------------------
# Wir nehmen Band 1
before_band = before_image[0, :, :].astype("float32")
after_band = after_image[0, :, :].astype("float32")

# NoData zu NaN setzen
if before_nodata is not None:
    before_band[before_band == before_nodata] = np.nan
if after_nodata is not None:
    after_band[after_band == after_nodata] = np.nan

# Gültige Werte auf 0–1 bringen (hier: max gültiger Wert = 255)
# Zuerst gültiges Maximum bestimmen (ohne NaN)
valid_max_before = np.nanmax(before_band)
valid_max_after = np.nanmax(after_band)

print("valid_max_before:", valid_max_before)
print("valid_max_after:", valid_max_after)

# Skalieren (typisch für 8‑bit-artige Daten, die in int32 gespeichert sind)
before_band = before_band / 255.0
after_band = after_band / 255.0


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
    # NaN mit Median füllen (statt 0), sonst extremer Randgradient
    valid = band[~np.isnan(band)]
    if valid.size == 0:
        raise ValueError("No valid pixels in band")

    median_val = float(np.median(valid))
    band_filled = np.nan_to_num(band, nan=median_val)

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

def plot_grey(img, title="", percentile=(1, 99)):
    """
    Sobel-Graustufen-Plot mit Percentil-Clipping für besseren Kontrast.
    """
    img = np.nan_to_num(img, nan=0.0)

    lo, hi = percentile
    vmin = np.percentile(img, lo)
    vmax = np.percentile(img, hi)

    img_norm = (img - vmin) / (vmax - vmin + 1e-12)
    img_norm = np.clip(img_norm, 0, 1)

    plt.figure()
    plt.imshow(img_norm, cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

plot_grey(before_Sobel, "Before Sobel")
plot_grey(after_Sobel, "After Sobel")