import numpy as np
from skimage.filters import sobel as skimage_sobel


def compute_sobel_texture(img, band=0, nodata=None, scale=255.0):
    img = np.asarray(img)

    if img.ndim == 3:
        band_data = img[band, :, :].astype("float32")
    else:
        band_data = img.astype("float32")

    if nodata is not None:
        band_data[band_data == nodata] = np.nan

    band_data = band_data / scale

    valid = band_data[~np.isnan(band_data)]
    if valid.size == 0:
        raise ValueError("Keine gültigen Pixel im Band")

    median_val = float(np.median(valid))
    band_filled = np.nan_to_num(band_data, nan=median_val)

    magnitude = skimage_sobel(band_filled)
    return {"magnitude": magnitude}