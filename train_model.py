"""
Full pipeline: VGG16 transfer-learning binary classifier
1) Train on shapefile + raster
2) Save the model
3) Pixel-wise classification of a new raster -> georeferenced GeoTIFF
"""

# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

import geopandas as gpd
import rasterio
from rasterio.windows import Window

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import RMSprop

# ============================================================
# >>> CONFIG — EDIT THESE <<<
# ============================================================
TRAIN_SHP = r"C:\Users\job02\Downloads\Training_Data.shp"
VAL_SHP   = r"C:\Users\job02\Downloads\Validation_Data.shp"
RASTER    = r"C:\Users\job02\Downloads\ea_kahramanmaras.tif"

LABEL_COL  = "destroyed"
CHIP_SIZE  = 128
BATCH_SIZE = 16
EPOCHS     = 15

MODEL_PATH = r"C:\Users\job02\Downloads\vgg16_destroyed.keras"

INFER_RASTER  = r"C:\Users\job02\Downloads\aoi_clips\aoi_clips\after_aoi_clip.tif"
OUTPUT_RASTER = r"C:\Users\job02\Downloads\aoi_clips\aoi_clips\classification.tif"
THRESHOLD = 0.5

# Pixel-wise method:
#   STRIDE small  -> smoother, per-pixel-like result (slower)
#   STRIDE = 16 or 8 is a good compromise between speed & smoothness
STRIDE = 16


# ============================================================
# DATA LOADING: shapefile + raster -> image chips + labels
# ============================================================
def extract_chips(shp_path, raster_path, label_col, size=128):
    gdf = gpd.read_file(shp_path)

    matches = [c for c in gdf.columns if c.lower() == label_col.lower()]
    if not matches:
        raise KeyError(
            f"Label column '{label_col}' not found in {shp_path}. "
            f"Available columns: {list(gdf.columns)}"
        )
    actual_col = matches[0]
    print(f"{shp_path}: using label column '{actual_col}'")

    X, y = [], []
    half = size // 2

    with rasterio.open(raster_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            cx, cy = geom.centroid.x, geom.centroid.y
            r, c = src.index(cx, cy)
            r0, c0 = r - half, c - half

            if r0 < 0 or c0 < 0 or r0 + size > src.height or c0 + size > src.width:
                continue

            window = Window(c0, r0, size, size)
            chip = src.read([1, 2, 3], window=window).astype("float32")
            chip = np.transpose(chip, (1, 2, 0))

            X.append(chip)
            y.append(float(row[actual_col]))

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32")

    if X.size > 0 and X.max() > 1.0:
        X /= 255.0

    print(f"{shp_path}: extracted {len(X)} chips of shape "
          f"{X.shape[1:] if len(X) else '(none)'}, "
          f"labels: {np.unique(y) if len(y) else '(none)'}")
    return X, y


# Build the datasets
X_train, y_train = extract_chips(TRAIN_SHP, RASTER, LABEL_COL, CHIP_SIZE)
X_val,   y_val   = extract_chips(VAL_SHP,   RASTER, LABEL_COL, CHIP_SIZE)

if len(X_train) == 0 or len(X_val) == 0:
    raise RuntimeError(
        "No chips were extracted. Likely causes: shapefile points fall "
        "outside the raster extent, or the CRS/raster path is wrong."
    )

training_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

validation_dataset = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# MODEL: VGG16 (frozen) + classification head
# ============================================================
vgg16_feat_extr = VGG16(
    include_top=False,
    input_shape=(CHIP_SIZE, CHIP_SIZE, 3),
    weights="imagenet"
)
vgg16_feat_extr.trainable = False

x = vgg16_feat_extr.layers[14].output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
outputs = Dense(1, activation="sigmoid")(x)

pretrained_model = Model(inputs=vgg16_feat_extr.input, outputs=outputs)
pretrained_model.summary()


# ============================================================
# COMPILE + TRAIN
# ============================================================
pretrained_model.compile(
    optimizer=RMSprop(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

diagnostics = pretrained_model.fit(
    training_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset
)


# ============================================================
# SAVE MODEL
# ============================================================
pretrained_model.save(MODEL_PATH)
print(f"Model saved to:\n{MODEL_PATH}")


# ============================================================
# PLOT TRAINING RESULTS
# ============================================================
history = diagnostics.history

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history["loss"], label="train loss")
plt.plot(history["val_loss"], label="val loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss")
plt.subplot(1, 2, 2)
plt.plot(history["accuracy"], label="train acc")
plt.plot(history["val_accuracy"], label="val acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy")
plt.tight_layout()
plt.show()
