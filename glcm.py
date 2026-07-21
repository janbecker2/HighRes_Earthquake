import numpy as np
from skimage.feature import graycomatrix, graycoprops


def compute_glcm_texture(img, win=15, step=2, levels=32, props=("energy", "homogeneity"), nodata=None):
    """
    Berechnet GLCM-Texturkarten per Sliding Window für ein RGB- oder Graustufenbild.

    Parameter
    ---------
    img : np.ndarray
        Bild als (bands, H, W) [RGB] oder (H, W) [Graustufen], so wie von
        load_image()/load_tifs() zurückgegeben.
    win : int
        Fenstergröße für das Sliding Window.
    step : int
        Schrittweite zwischen den Fenstern.
    levels : int
        Anzahl der Graustufen für die GLCM-Quantisierung.
    props : tuple[str]
        Welche GLCM-Eigenschaften berechnet werden sollen (z.B. "energy", "homogeneity").
    nodata : float oder None
        Nodata-Wert, um leere/ungültige Pixel zu maskieren.

    Returns
    -------
    dict mit einer Textur-Karte (np.ndarray) pro angeforderter Eigenschaft.
    """
    img = np.asarray(img, dtype=np.float32)

    if img.ndim == 3:
        gray = img.mean(axis=0)
        mask = img[0] == nodata if nodata is not None else np.zeros_like(gray, dtype=bool)
    else:
        gray = img
        mask = img == nodata if nodata is not None else np.zeros_like(gray, dtype=bool)

    gray = np.nan_to_num(np.where(mask, 0, gray))
    gray_u8 = np.clip(gray / 255 * (levels - 1), 0, levels - 1).astype(np.uint8)

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
            glcm = graycomatrix(window, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)
            for p in props:
                prop_maps[p][i, j] = graycoprops(glcm, p)[0, 0]

    return prop_maps