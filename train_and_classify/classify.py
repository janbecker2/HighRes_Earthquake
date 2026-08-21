import numpy as np

import torch
import torch.nn as nn
from torchvision import models

import geopandas as gpd
import rasterio
from rasterio.windows import Window, from_bounds
from shapely.geometry import box


# config parameters and paths
class Config:
    CHIP_SIZE    = 224          # image size


    MODEL_PATH   = r"C:\Users\job02\Downloads\resnet50_damaged_wo_b1.pth"

    INFER_RASTER = r"C:\Users\job02\Downloads\study1_textures.tif\study1_test_stack.tif"
    FOOTPRINTS   = r"C:\Users\job02\Downloads\2023Turkey_earthquake_data\2023Turkey_earthquake_data\GBA_building_footprint\Turkey_GBA_building_data.shp"
    FOOT_LAYER   = None
    OUTPUT_GPKG  = r"C:\Users\job02\Downloads\buildings_classified.gpkg"
    PAD_FRAC     = 0.15
    BATCH_INFER  = 64
    DEFAULT_THRESHOLD = 0.5    # Fallback if no threshold metadata found

    NORM_MEAN = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    NORM_STD  = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    CLASS_NAMES  = {0: "destroyed", 1: "undestroyed"}


cfg = Config()



def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


# building model
def build_model(device, freeze_backbone=False):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)

    old_conv = model.conv1
    # Create a new layer with 6 input channels instead of 3
    model.conv1 = nn.Conv2d(6, old_conv.out_channels, 
                            kernel_size=old_conv.kernel_size, 
                            stride=old_conv.stride, 
                            padding=old_conv.padding, 
                            bias=old_conv.bias is not None)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_feats, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, 1),
    )
    return model.to(device)


# load model with helper function
def load_model(device, model_path=None):
    model_path = model_path or cfg.MODEL_PATH
    checkpoint = torch.load(model_path, map_location=device)

    model = build_model(device, freeze_backbone=False)

    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        optimal_threshold = checkpoint.get("optimal_threshold", cfg.DEFAULT_THRESHOLD)
    else:
        model.load_state_dict(checkpoint)
        optimal_threshold = cfg.DEFAULT_THRESHOLD

    model.eval()
    print(f"Loaded model from: {model_path}")
    print(f"Using loaded Optimal Decision Threshold: {optimal_threshold:.4f}")
    return model, optimal_threshold


# scale raster to match model training
def scale_raster_array(arr, src_dtype=None):
    arr = np.nan_to_num(arr.astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
    dtype_str = str(src_dtype) if src_dtype is not None else None

    if dtype_str == "uint8":
        pass  # already 0-255
    elif dtype_str == "uint16":
        arr = arr / 65535.0 * 255.0
    elif dtype_str == "int16":
        arr = np.clip(arr, 0, None) / 32767.0 * 255.0
    elif dtype_str in ("float32", "float64") or dtype_str is None:
        m = arr.max()
        if m <= 1.5:
            arr = arr * 255.0          # already 0-1 float
        elif m <= 255.5:
            pass                        # already 0-255
        else:
            arr = arr / max(m, 1.0) * 255.0  # fallback: per-chip max scaling
    else:
        m = arr.max()
        arr = arr / max(m, 1.0) * 255.0

    return np.clip(arr, 0, 255)


# convert a raster chip to a tensor for model input
def _chip_to_tensor(chip_hwc, chip_size, src_dtype=None):
    chip = scale_raster_array(chip_hwc, src_dtype=src_dtype)  # -> 0-255 float32
    chip = chip / 255.0                                        # -> 0-1

    t = torch.from_numpy(chip).unsqueeze(0)    # (1,6,H,W)
    t = torch.nn.functional.interpolate(
        t, size=(chip_size, chip_size), mode="bilinear", align_corners=False
    ).squeeze(0)                               # (6,chip,chip)

    mean = torch.tensor(cfg.NORM_MEAN).view(6, 1, 1)
    std  = torch.tensor(cfg.NORM_STD).view(6, 1, 1)
    t = (t - mean) / std
    return t


# classify footprints in raster
def classify_footprints(model, device, threshold=None,
                        infer_raster=None, footprints=None,
                        output_gpkg=None, foot_layer=None,
                        pad_frac=None, batch_infer=None, class_names=None):
    infer_raster = infer_raster or cfg.INFER_RASTER
    footprints   = footprints   or cfg.FOOTPRINTS
    output_gpkg  = output_gpkg  or cfg.OUTPUT_GPKG
    foot_layer   = foot_layer   if foot_layer is not None else cfg.FOOT_LAYER
    pad_frac     = pad_frac     if pad_frac    is not None else cfg.PAD_FRAC
    batch_infer  = batch_infer  or cfg.BATCH_INFER
    threshold    = threshold    if threshold   is not None else cfg.DEFAULT_THRESHOLD
    class_names  = class_names  or cfg.CLASS_NAMES

    class_for_1 = class_names[1]
    class_for_0 = class_names[0]

    with rasterio.open(infer_raster) as src:
        raster_crs    = src.crs
        raster_bounds = src.bounds
        raster_dtype  = src.dtypes[0]
        H, W = src.height, src.width

    if foot_layer is not None:
        foot_crs = gpd.read_file(footprints, rows=1, layer=foot_layer).crs
    else:
        foot_crs = gpd.read_file(footprints, rows=1).crs

    raster_box = gpd.GeoDataFrame(geometry=[box(*raster_bounds)], crs=raster_crs)
    if foot_crs is not None and raster_box.crs != foot_crs:
        raster_box_ft = raster_box.to_crs(foot_crs)
    else:
        raster_box_ft = raster_box
    bbox_tuple = tuple(raster_box_ft.total_bounds)

    print(f"Reading footprints within raster bbox...")
    if foot_layer is not None:
        gdf = gpd.read_file(footprints, layer=foot_layer, bbox=bbox_tuple)
    else:
        gdf = gpd.read_file(footprints, bbox=bbox_tuple)

    if len(gdf) == 0:
        raise RuntimeError("No footprints fall within the raster extent.")

    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    raster_box_rcrs = box(*raster_bounds)
    gdf = gdf[gdf.geometry.intersects(raster_box_rcrs)].reset_index(drop=True)
    print(f"Footprints intersecting raster: {len(gdf)}")

    model.eval()
    probs_out = np.full(len(gdf), np.nan, dtype="float32")

    with rasterio.open(infer_raster) as src:
        batch_tensors, batch_idx = [], []

        def flush():
            if not batch_tensors:
                return
            x = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                logits = model(x).squeeze(1)
                p = torch.sigmoid(logits).cpu().numpy()  # logits -> probabilities
            for gi, pv in zip(batch_idx, p):
                probs_out[gi] = pv
            batch_tensors.clear()
            batch_idx.clear()

        for gi, geom in enumerate(gdf.geometry):
            if geom is None or geom.is_empty:
                continue

            minx, miny, maxx, maxy = geom.bounds
            dx = (maxx - minx) * pad_frac
            dy = (maxy - miny) * pad_frac
            minx, maxx = minx - dx, maxx + dx
            miny, maxy = miny - dy, maxy + dy

            try:
                win = from_bounds(minx, miny, maxx, maxy, src.transform)
            except Exception:
                continue

            col_off = max(0, int(np.floor(win.col_off)))
            row_off = max(0, int(np.floor(win.row_off)))
            w = min(int(np.ceil(win.width)),  W - col_off)
            h = min(int(np.ceil(win.height)), H - row_off)
            if w < 2 or h < 2:
                continue

            window = Window(col_off, row_off, w, h)
            chip = src.read(window=window).astype("float32")
            
            if chip.shape[1] < 2 or chip.shape[2] < 2:
                continue

            batch_tensors.append(_chip_to_tensor(chip, cfg.CHIP_SIZE, src_dtype=raster_dtype))
            batch_idx.append(gi)

            if len(batch_tensors) >= batch_infer:
                flush()
            if (gi + 1) % 2000 == 0:
                print(f"  {gi+1}/{len(gdf)} buildings processed")

        flush()

    gdf["prob"] = probs_out
    gdf["pred_class"] = np.where(
        np.isnan(probs_out), "no_data",
        np.where(probs_out >= threshold, class_for_1, class_for_0)
    )

    n_ok  = int(np.isfinite(probs_out).sum())
    n_dmg = int((gdf["pred_class"] == class_for_1).sum())
    print(f"Classified {n_ok}/{len(gdf)} buildings "
          f"-> {n_dmg} '{class_for_1}', {n_ok - n_dmg} '{class_for_0}'")

    gdf.to_file(output_gpkg, driver="GPKG")
    return gdf

# run inference
def do_inference():
    device = get_device()
    model, optimal_threshold = load_model(device)
    return classify_footprints(model, device, threshold=optimal_threshold)


if __name__ == "__main__":
    do_inference()
