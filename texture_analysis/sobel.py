import numpy as np
from skimage.filters import sobel as skimage_sobel


def compute_sobel_texture(img, band=0, nodata=None, scale=None, to_gray=False):
    img = np.asarray(img)

    # auto-detect scale 
    if scale is None:
        if np.issubdtype(img.dtype, np.integer):
            scale = float(np.iinfo(img.dtype).max)   
        else:
            scale = 1.0

    # select band and convert to grayscale
    if img.ndim == 3:
        if to_gray and img.shape[0] == 3:
            # luminance conversion (Rec. 601)
            band_data = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).astype("float32")
        else:
            band_data = img[band, :, :].astype("float32")
    else:
        band_data = img.astype("float32")

    if nodata is not None:
        band_data[band_data == nodata] = np.nan

    # normalize
    band_data = band_data / scale

    # check for valid pixels
    valid = band_data[~np.isnan(band_data)]
    if valid.size == 0:
        raise ValueError("Keine gültigen Pixel im Band")

    # fill NaNs with median so Sobel doesn't propagate them
    median_val = float(np.median(valid))
    band_filled = np.nan_to_num(band_data, nan=median_val)

    magnitude = skimage_sobel(band_filled)
    return {"magnitude": magnitude}
