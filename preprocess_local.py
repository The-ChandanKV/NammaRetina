"""
NammaRetina - Preprocessing Pipeline v3 (Final)
================================================
Saves processed PNG images (uint8, 0-255) ready for TensorFlow training.
Normalization is NOT applied here — handled by TensorFlow during training.

Pipeline: Verify → Crop Black Borders → Resize 224×224 → Ben Graham → Save PNG

Output structure:
    processed/
    ├── train_images/   (2930 PNGs)
    ├── val_images/     (366 PNGs)
    ├── test_images/    (366 PNGs)
    ├── train.csv
    ├── valid.csv
    └── test.csv
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "APTOS-19")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")

TARGET_SIZE = 224
SEVERITY_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}

# =============================================================================
# Preprocessing Functions
# =============================================================================

def verify_image(img_path):
    """Step 1: Verify image exists, is readable, has 3 channels, reasonable size."""
    if not os.path.exists(img_path):
        return False, "file_not_found"
    img = cv2.imread(img_path)
    if img is None:
        return False, "unreadable"
    if len(img.shape) != 3 or img.shape[2] != 3:
        return False, f"wrong_channels_{img.shape}"
    h, w = img.shape[:2]
    if h < 50 or w < 50:
        return False, f"too_small_{w}x{h}"
    if np.mean(img) < 5:
        return False, "mostly_black"
    return True, "ok"


def crop_black_borders(img):
    """Step 2: Remove black background around the circular retina."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 50:
            img = img[y:y+h, x:x+w]
    return img


def ben_graham_preprocess(img, sigmaX=10):
    """Step 3: Enhance retinal features (vessels, lesions) by subtracting local mean."""
    img = cv2.addWeighted(
        img, 4,
        cv2.GaussianBlur(img, (0, 0), sigmaX), -4,
        128
    )
    return img


def preprocess_single(img_path):
    """
    Full pipeline for one image:
    Verify → Crop → Resize → Ben Graham
    Returns uint8 image (0-255) or None.
    """
    is_valid, reason = verify_image(img_path)
    if not is_valid:
        return None, reason

    img = cv2.imread(img_path)
    img = crop_black_borders(img)
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    img = ben_graham_preprocess(img, sigmaX=10)

    # Clip to valid uint8 range (Ben Graham can produce values outside 0-255)
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img, "ok"


# =============================================================================
# Comparison Figure Generator
# =============================================================================

def generate_comparison_figure(df, input_dir, output_path):
    """Generate a figure showing: Original → Cropped → Ben Graham for 5 samples."""
    fig, axes = plt.subplots(5, 3, figsize=(15, 22))
    fig.suptitle("Preprocessing Comparison: Original → Cropped → Ben Graham",
                 fontsize=16, fontweight="bold", y=0.98)

    # Pick one sample from each class
    samples = []
    for label in range(5):
        class_imgs = df[df["diagnosis"] == label]["id_code"].values
        if len(class_imgs) > 0:
            samples.append((np.random.choice(class_imgs), label))

    for row, (img_id, label) in enumerate(samples):
        img_path = os.path.join(input_dir, f"{img_id}.png")
        original = cv2.imread(img_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Stage 1: Original
        axes[row][0].imshow(original_rgb)
        axes[row][0].set_title(
            f"Original ({original.shape[1]}×{original.shape[0]})\n"
            f"Class {label}: {SEVERITY_NAMES[label]}",
            fontsize=10, fontweight="bold"
        )
        axes[row][0].axis("off")

        # Stage 2: Cropped + Resized
        cropped = crop_black_borders(original.copy())
        resized = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE))
        axes[row][1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        axes[row][1].set_title("Cropped + Resized (224×224)", fontsize=10)
        axes[row][1].axis("off")

        # Stage 3: Ben Graham
        processed = ben_graham_preprocess(resized.copy(), sigmaX=10)
        processed = np.clip(processed, 0, 255).astype(np.uint8)
        axes[row][2].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        axes[row][2].set_title("Ben Graham Processed", fontsize=10)
        axes[row][2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Comparison figure saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    splits = [
        ("train_images", "train_images/train_images", "train_1.csv", "train.csv"),
        ("val_images",   "val_images/val_images",     "valid.csv",   "valid.csv"),
        ("test_images",  "test_images/test_images",   "test.csv",    "test.csv"),
    ]

    print("=" * 65)
    print("  NammaRetina — Preprocessing Pipeline v3 (Final)")
    print("  Pipeline: Verify → Crop → Resize 224×224 → Ben Graham → PNG")
    print("  Output: uint8 [0-255] PNG images (no normalization)")
    print("=" * 65)

    # ── Clean old processed folder ──
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print("\n  🗑️  Removed old processed/ folder")

    report = {}

    for out_folder, src_folder, src_csv, dst_csv in splits:
        out_dir = os.path.join(OUTPUT_DIR, out_folder)
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(DATASET_DIR, src_csv)
        df = pd.read_csv(csv_path)
        input_dir = os.path.join(DATASET_DIR, src_folder)

        split_name = out_folder.replace("_images", "")
        print(f"\n{'─' * 65}")
        print(f"  📂 {split_name.upper()} — {len(df)} images")
        print(f"{'─' * 65}")

        ok_count = 0
        failed = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
            img_id = row["id_code"]
            img_path = os.path.join(input_dir, f"{img_id}.png")
            save_path = os.path.join(out_dir, f"{img_id}.png")

            processed, status = preprocess_single(img_path)

            if processed is not None:
                cv2.imwrite(save_path, processed)
                ok_count += 1
            else:
                failed.append({"id_code": img_id, "reason": status})

        # Copy CSV as-is (no modifications to labels or filenames)
        dst_csv_path = os.path.join(OUTPUT_DIR, dst_csv)
        shutil.copy2(csv_path, dst_csv_path)

        report[split_name] = {
            "expected": len(df),
            "saved": ok_count,
            "failed": len(failed),
            "failed_list": failed,
            "csv_rows": len(df),
        }

        if failed:
            print(f"\n  ❌ {len(failed)} failed:")
            for f in failed:
                print(f"     {f['id_code']}: {f['reason']}")

    # ── Generate comparison figure ──
    print(f"\n{'─' * 65}")
    print("  📊 Generating preprocessing comparison figure...")
    print(f"{'─' * 65}")

    train_csv = pd.read_csv(os.path.join(DATASET_DIR, "train_1.csv"))
    train_src = os.path.join(DATASET_DIR, "train_images", "train_images")
    fig_path = os.path.join(OUTPUT_DIR, "preprocessing_comparison.png")
    generate_comparison_figure(train_csv, train_src, fig_path)

    # ══════════════════════════════════════════════════════════════════
    #  VERIFICATION & FINAL REPORT
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'═' * 65}")
    print("  VERIFICATION & FINAL PREPROCESSING REPORT")
    print(f"{'═' * 65}")

    all_pass = True

    for split_name, out_folder in [("train", "train_images"),
                                    ("val", "val_images"),
                                    ("test", "test_images")]:
        out_dir = os.path.join(OUTPUT_DIR, out_folder)
        png_files = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        info = report[split_name]

        print(f"\n  ┌─ {split_name.upper()} {'─' * (55 - len(split_name))}")

        # Count check
        count_ok = len(png_files) == info["expected"]
        print(f"  │  Image count:     {len(png_files)} / {info['expected']}  "
              f"{'✅' if count_ok else '❌ MISMATCH'}")
        all_pass = all_pass and count_ok

        # CSV consistency
        csv_name = "train.csv" if split_name == "train" else "valid.csv" if split_name == "val" else "test.csv"
        csv_df = pd.read_csv(os.path.join(OUTPUT_DIR, csv_name))
        csv_ids = set(csv_df["id_code"].astype(str) + ".png")
        file_ids = set(png_files)
        csv_match = csv_ids == file_ids
        print(f"  │  CSV consistency: {len(csv_df)} rows  "
              f"{'✅ all match' if csv_match else '❌ MISMATCH'}")
        all_pass = all_pass and csv_match

        # Missing images
        missing = csv_ids - file_ids
        print(f"  │  Missing images:  {len(missing)}  {'✅' if len(missing) == 0 else '❌'}")
        all_pass = all_pass and (len(missing) == 0)

        # Corrupted check
        print(f"  │  Corrupted:       {info['failed']}  "
              f"{'✅' if info['failed'] == 0 else '⚠️  removed from output'}")

        # Sample image verification
        sample_path = os.path.join(out_dir, png_files[0])
        sample = cv2.imread(sample_path)
        if sample is not None:
            h, w, c = sample.shape
            shape_ok = (h == 224 and w == 224 and c == 3)
            dtype_ok = sample.dtype == np.uint8
            print(f"  │  Resolution:      {w}×{h}×{c}  "
                  f"{'✅' if shape_ok else '❌ WRONG SIZE'}")
            print(f"  │  Pixel dtype:     {sample.dtype}  "
                  f"{'✅' if dtype_ok else '❌ WRONG TYPE'}")
            print(f"  │  Pixel range:     [{sample.min()}, {sample.max()}]  ✅ uint8")
            print(f"  │  Image format:    PNG  ✅")
            all_pass = all_pass and shape_ok and dtype_ok
        else:
            print(f"  │  ❌ Could not read sample image!")
            all_pass = False

        print(f"  └{'─' * 62}")

    # Overall summary
    print(f"\n{'═' * 65}")
    if all_pass:
        print("  ✅ ALL VERIFICATIONS PASSED")
    else:
        print("  ❌ SOME CHECKS FAILED — review above")
    print(f"{'═' * 65}")

    # Folder structure
    print(f"\n  📁 Output structure:")
    print(f"     processed/")
    for item in sorted(os.listdir(OUTPUT_DIR)):
        item_path = os.path.join(OUTPUT_DIR, item)
        if os.path.isdir(item_path):
            count = len([f for f in os.listdir(item_path) if f.endswith(".png")])
            print(f"     ├── {item}/          ({count} PNGs)")
        else:
            size_kb = os.path.getsize(item_path) / 1024
            print(f"     ├── {item}    ({size_kb:.0f} KB)")

    # Total size
    total_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fn in os.walk(OUTPUT_DIR) for f in fn
    ) / (1024 * 1024)
    print(f"\n  📦 Total size: {total_mb:.1f} MB")
    print(f"     (Ready to zip & upload to Google Colab)")

    print(f"\n{'═' * 65}")
    print(f"  PREPROCESSING SUMMARY")
    print(f"{'═' * 65}")
    print(f"  ✅ Crop Black Borders    — contour-based retina isolation")
    print(f"  ✅ Resize                — 224 × 224 × 3")
    print(f"  ✅ Ben Graham            — retinal features enhanced")
    print(f"  ✅ Saved as PNG          — uint8 [0-255]")
    print(f"  ✅ No normalization      — handled by TensorFlow at training time")
    print(f"  ✅ CSVs preserved        — original labels & filenames intact")
    print(f"  ✅ Comparison figure     — preprocessing_comparison.png")
    print(f"{'═' * 65}")
