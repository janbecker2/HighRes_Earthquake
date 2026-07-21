import numpy as np
from skimage.feature import graycomatrix, graycoprops


def compute_glcm_texture(img, win=15, step=2, levels=32,
                         props=("energy", "homogeneity"), nodata=None, scale=None):
    img = np.asarray(img, dtype=np.float32)

    # auto-detect scale from ORIGINAL dtype (before float cast) if not given
    # note: pass scale explicitly for float TIFFs if needed
    if scale is None:
        # img was already cast to float32, so infer from value range instead
        max_val = np.nanmax(img)
        if max_val <= 1.0:
            scale = 1.0            # already normalized floats
        elif max_val <= 255:
            scale = 255.0          # 8-bit
        else:
            scale = 65535.0        # 16-bit

    if img.ndim == 3:
        gray = img.mean(axis=0)
        mask = np.any(img == nodata, axis=0) if nodata is not None \
            else np.zeros_like(gray, dtype=bool)
    else:
        gray = img
        mask = img == nodata if nodata is not None \
            else np.zeros_like(gray, dtype=bool)

    gray = np.nan_to_num(np.where(mask, 0, gray))
    gray_u8 = np.clip(gray / scale * (levels - 1), 0, levels - 1).astype(np.uint8)

    h, w = gray_u8.shape
    out_h = (h - win) // step + 1
    out_w = (w - win) // step + 1

    prop_maps = {p: np.full((out_h, out_w), np.nan, dtype=np.float32) for p in props}

    for i in range(out_h):
        for j in range(out_w):
            y, x = i * step, j * step
            window = gray_u8[y:y + win, x:x + win]
            if mask[y:y + win, x:x + win].any():
                continue
            glcm = graycomatrix(window, distances=[1], angles=[0],
                                levels=levels, symmetric=True, normed=True)
            for p in props:
                prop_maps[p][i, j] = graycoprops(glcm, p)[0, 0]

    return prop_maps