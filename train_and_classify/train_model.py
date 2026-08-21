import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.transforms import v2
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split

import geopandas as gpd
import rasterio
from rasterio.windows import Window, from_bounds
from shapely.geometry import box
from PIL import Image

# set parameters and paths
class Config:
    # --- Training data (one subfolder per class) ---
    TRAIN_DIR    = r"C:\Users\job02\Downloads\training_images_textures_without_b2"
    VAL_SPLIT    = 0.2
    CHIP_SIZE    = 224          
    BATCH_SIZE   = 32
    EPOCHS       = 40
    LR           = 1e-4         
    WEIGHT_DECAY = 1e-4         
    FREEZE_BACKBONE = False    
    PATIENCE     = 6            
    SEED         = 42
    SELECT_ON    = "val_f1"    
    USE_AMP      = True         

    MODEL_PATH   = r"C:\Users\job02\Downloads\resnet50_damaged_wo_b2.pth"

    INFER_RASTER = r"C:\Users\job02\Downloads\aoi_karamankaras.tif"
    FOOTPRINTS   = r"C:\Users\job02\Downloads\2023Turkey_earthquake_data\2023Turkey_earthquake_data\GBA_building_footprint\Turkey_GBA_building_data.shp"
    FOOT_LAYER   = None
    OUTPUT_GPKG  = r"C:\Users\job02\Downloads\buildings_classified.gpkg"
    PAD_FRAC     = 0.15
    BATCH_INFER  = 64
    DEFAULT_THRESHOLD = 0.5   

    # ImageNet normalization 
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD  = [0.229, 0.224, 0.225]

    CLASS_NAMES  = {0: "damaged", 1: "undamaged"}


cfg = Config()

# use gpu if available
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device

# build model with 6 band input layer
def build_model(device, freeze_backbone=None):
    freeze_backbone = cfg.FREEZE_BACKBONE if freeze_backbone is None else freeze_backbone
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)

    old_conv = model.conv1
    new_conv = nn.Conv2d(6, old_conv.out_channels, 
                         kernel_size=old_conv.kernel_size, 
                         stride=old_conv.stride, 
                         padding=old_conv.padding, 
                         bias=old_conv.bias is not None)

    with torch.no_grad():
        # Keep original RGB weights for channels 0, 1, 2
        new_conv.weight[:, :3, :, :] = old_conv.weight
        
        new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

    model.conv1 = new_conv

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.conv1.parameters():
            p.requires_grad = True

    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_feats, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, 1),
    )

    return model.to(device)

# build training optimizer with automatic learning curve
def build_optimizer(model):
    param_groups = [
        {"params": model.conv1.parameters(),  "lr": cfg.LR * 0.1},
        {"params": model.bn1.parameters(),    "lr": cfg.LR * 0.1},
        {"params": model.layer1.parameters(), "lr": cfg.LR * 0.1},
        {"params": model.layer2.parameters(), "lr": cfg.LR * 0.1},
        {"params": model.layer3.parameters(), "lr": cfg.LR * 0.3},
        {"params": model.layer4.parameters(), "lr": cfg.LR * 0.5},
        {"params": model.fc.parameters(),     "lr": cfg.LR * 10},
    ]
    param_groups = [g for g in param_groups if any(p.requires_grad for p in g["params"])]
    return torch.optim.Adam(param_groups, weight_decay=cfg.WEIGHT_DECAY)

# load existing model (not used in this script)
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

# scale raster for model imput
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
            arr = arr * 255.0         
        elif m <= 255.5:
            pass                        
        else:
            arr = arr / max(m, 1.0) * 255.0
    else:
        m = arr.max()
        arr = arr / max(m, 1.0) * 255.0

    return np.clip(arr, 0, 255)

# load raster as tensor
def rasterio_tensor_loader(path):
    with rasterio.open(path) as src:
        arr = src.read(masked=True) 
        
    arr = np.ma.filled(arr, 0).astype("float32")
    arr = arr / 255.0 
    
    return torch.from_numpy(arr)


# loaders to access training data
def get_dataloaders():
    train_tf = v2.Compose([
        v2.RandomResizedCrop(cfg.CHIP_SIZE, scale=(0.7, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(180),
        v2.Normalize(mean=[0.5]*6, std=[0.5]*6), 
    ])
    
    eval_tf = v2.Compose([
        v2.Resize((cfg.CHIP_SIZE, cfg.CHIP_SIZE)),
        v2.Normalize(mean=[0.5]*6, std=[0.5]*6),
    ])

    full_ds_train = datasets.ImageFolder(cfg.TRAIN_DIR, transform=train_tf,
                                         loader=rasterio_tensor_loader)
    full_ds_eval  = datasets.ImageFolder(cfg.TRAIN_DIR, transform=eval_tf,
                                         loader=rasterio_tensor_loader)

    print(f"Classes found: {full_ds_train.classes}")
    print(f"Class -> index: {full_ds_train.class_to_idx}")
    print(f"Total images: {len(full_ds_train)}")

    if len(full_ds_train.classes) < 2:
        print("Only one class folder found")

    cfg.CLASS_NAMES = {v: k for k, v in full_ds_train.class_to_idx.items()}

    labels = np.array(full_ds_train.targets)
    for idx, name in cfg.CLASS_NAMES.items():
        print(f"  class '{name}' (idx {idx}): {(labels == idx).sum()} images")

    all_idx = np.arange(len(full_ds_train))
    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=cfg.VAL_SPLIT,
        random_state=cfg.SEED,
        stratify=labels,
    )

    train_ds = Subset(full_ds_train, train_idx)
    val_ds   = Subset(full_ds_eval, val_idx)
    print(f"Split -> train: {len(train_ds)}, val: {len(val_ds)}")

    train_labels = labels[train_idx]
    for idx, name in cfg.CLASS_NAMES.items():
        print(f"  train class '{name}': {(train_labels == idx).sum()} images")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=True)

    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    print(f"Class balance in train set -> class 0: {n_neg}, class 1: {n_pos}, "
          f"pos_weight for BCEWithLogitsLoss: {pos_weight.item():.3f}")

    return train_loader, val_loader, pos_weight


# run epoch with optimization
def run_epoch(model, loader, criterion, optimizer, device, train=True, scaler=None):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_logits, all_targets = [], []

    torch.set_grad_enabled(train)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.float().to(device, non_blocking=True).unsqueeze(1)

        if train:
            optimizer.zero_grad()

        if scaler is not None and train:
            with torch.amp.autocast('cuda'):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(logits.detach())
        total_loss += loss.item() * xb.size(0)
        correct += ((probs >= 0.5).float() == yb).sum().item()
        total += xb.size(0)

        all_logits.extend(probs.cpu().numpy())
        all_targets.extend(yb.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_logits), np.array(all_targets)

# find best decision threshold based on val f1 score
def find_best_threshold(val_probs, val_targets):
    precisions, recalls, thresholds = precision_recall_curve(val_targets, val_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)

    if best_idx < len(thresholds):
        best_thresh = float(thresholds[best_idx])
    else:
        best_thresh = 0.5

    print(f"\n--- Scientific Threshold Optimization ---")
    print(f"Optimal Threshold (Max F1-Score): {best_thresh:.4f}")
    print(f"Max Validation F1-Score:           {f1_scores[best_idx]:.4f}")
    return best_thresh, float(f1_scores[best_idx])

# actual training of model
def train_model(model, train_loader, val_loader, pos_weight, device):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    scaler = torch.cuda.amp.GradScaler() if (cfg.USE_AMP and device.type == "cuda") else None

    best_score = float("inf") if cfg.SELECT_ON == "val_loss" else -float("inf")
    bad_epochs = 0
    best_model_state = None
    best_val_probs, best_val_targets = None, None

    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    for epoch in range(cfg.EPOCHS):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, True, scaler)
        va_loss, va_acc, va_probs, va_targets = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step(va_loss)

        va_f1 = f1_score(va_targets, (va_probs >= 0.5).astype(int), zero_division=0)

        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_accuracy"].append(va_acc)

        lr_now = optimizer.param_groups[-1]["lr"]  # head LR
        print(f"Epoch {epoch+1:02d}/{cfg.EPOCHS} "
              f"- loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"- val_loss {va_loss:.4f} val_acc {va_acc:.4f} val_f1 {va_f1:.4f} "
              f"- lr {lr_now:.1e}")

        current_score = va_loss if cfg.SELECT_ON == "val_loss" else va_f1
        improved = (current_score < best_score) if cfg.SELECT_ON == "val_loss" else (current_score > best_score)

        if improved:
            best_score = current_score
            bad_epochs = 0
            best_model_state = model.state_dict().copy()
            best_val_probs = va_probs
            best_val_targets = va_targets
            print(f"   ✓ New best ({cfg.SELECT_ON} {best_score:.4f}) — checkpoint cached")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.PATIENCE:
                print(f"   Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        optimal_threshold, best_f1 = find_best_threshold(best_val_probs, best_val_targets)
        save_model_with_metadata(model, optimal_threshold, best_f1)

    return history


def save_model_with_metadata(model, threshold, val_f1=None, model_path=None):
    model_path = model_path or cfg.MODEL_PATH
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimal_threshold": threshold,
        "val_f1": val_f1,
    }
    torch.save(checkpoint, model_path)
    print(f"Model and Optimal Threshold ({threshold:.4f})")


def plot_history(history):
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

# executef
def do_training():
    device = get_device()
    train_loader, val_loader, pos_weight = get_dataloaders()
    model = build_model(device)
    history = train_model(model, train_loader, val_loader, pos_weight, device)
    plot_history(history)
    return model, device


if __name__ == "__main__":

    trained_model, dev = (None, None)

    trained_model, dev = do_training()
