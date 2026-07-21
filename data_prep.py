from pathlib import Path
import numpy as np
import rasterio

import glcm
import sobel


# Function to load images from hard drive
def load_image(path, band=None):

    # check if file exists or path is correct
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    # check if single band is set, else return complete image
    with rasterio.open(path) as src:
        profile = src.profile
        if band is not None:
            img = src.read(band)
        else:
            img = src.read()
            if img.shape[0] == 1:
                img = img[0]

    return img, profile


# Load all supported images from folder (tif + jpg by default)
def load_tifs(folder, patterns=("*.tif", "*.tiff", "*.jpg", "*.jpeg"),
              band=None, stack=False):
    folder = Path(folder)

    # allow a single string pattern too, for backward compatibility
    if isinstance(patterns, str):
        patterns = (patterns,)

    # collect files matching any of the patterns
    files = []
    for pat in patterns:
        files.extend(folder.glob(pat))
    files = sorted(set(files))  # set() avoids duplicates on case-insensitive systems

    if not files:
        raise FileNotFoundError(
            f"No files found with patterns {patterns} in {folder}"
        )

    # get images, profiles and names and store
    images, profiles, names = [], [], []
    for f in files:
        img, profile = load_image(f, band=band)
        images.append(img)
        profiles.append(profile)
        names.append(f.name)

    if stack:
        images = np.stack(images, axis=0)  # set stack=False if images have different heights/widths

    return {"images": images, "filenames": names, "profiles": profiles}


# Debugging functions, delete later
# result = load_tifs("aoi_clips/", stack=True)          # loads tif + jpg
# result = load_tifs("aoi_clips/", patterns="*.jpg")    # only jpg
# print(result["images"].shape)
# print(result["filenames"])


# Call texture analysis functions
def process_images(images, profiles=None, method="both", **kwargs):

    if isinstance(images, np.ndarray):
        images = [images[i] for i in range(images.shape[0])]

    results = []
    for img, profile in zip(images, profiles or [{}] * len(images)):
        if method == "glcm":
            res = glcm.compute_glcm_texture(img, nodata=profile.get("nodata"), **kwargs)
        elif method == "sobel":
            res = sobel.compute_sobel_texture(img, nodata=profile.get("nodata"), **kwargs)
        elif method == "both":
            res = {
                "glcm": glcm.compute_glcm_texture(img, nodata=profile.get("nodata"), **kwargs.get("glcm", {})),
                "sobel": sobel.compute_sobel_texture(img, nodata=profile.get("nodata"), **kwargs)
            }
        else:
            raise ValueError(f"Unknown Method: {method}")

        results.append(res)

    return results