"""Keep source identities and duplicate groups separate across train/val/test."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from retinal_preprocessing import prepare_retina

SPLITS = {"train": "train_1.csv", "validation": "valid.csv", "test": "test.csv"}
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


def difference_hash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def build_manifest(image_root, split_root, near_distance=-1):
    """Start from committed splits; quarantine duplicate groups into one split.

    Priority is test > validation > train, so test examples never move into
    training. Retain one image per group. This is image-level isolation, not
    patient-level isolation: APTOS metadata does not expose patient IDs.
    """
    source_labels = None
    source_csv = Path(image_root) / "train.csv"
    if source_csv.exists():
        source = pd.read_csv(source_csv, dtype={"id_code": str})
        if source.id_code.duplicated().any():
            raise ValueError("Source metadata contains duplicate image IDs")
        source_labels = dict(zip(source.id_code, source.diagnosis.astype(int)))
    files = {}
    for path in sorted(Path(image_root).rglob("*")):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            if path.stem in files:
                raise ValueError(f"Ambiguous image ID: {path.stem}")
            files[path.stem] = path.resolve()
    records = []
    seen = set()
    for split, filename in SPLITS.items():
        table = pd.read_csv(Path(split_root) / filename, dtype={"id_code": str})
        for row in table.itertuples(index=False):
            if row.id_code in seen:
                raise ValueError("The committed splits contain overlapping image IDs")
            seen.add(row.id_code)
            if row.id_code not in files:
                raise FileNotFoundError(f"Image missing for ID {row.id_code}")
            label = int(row.diagnosis)
            if label not in range(5):
                raise ValueError(f"Invalid severity for {row.id_code}")
            if source_labels is not None and source_labels.get(row.id_code) != label:
                raise ValueError(f"Source label differs from the committed split: {row.id_code}")
            image = cv2.imread(str(files[row.id_code]))
            if image is None or min(image.shape[:2]) < 50:
                raise ValueError(f"Unreadable or undersized image: {row.id_code}")
            records.append({"id_code": row.id_code, "label": label, "split": split,
                            "path": str(files[row.id_code]),
                            "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                            "dhash": difference_hash(image)})

    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, record in enumerate(records):
        for j in range(i):
            other = records[j]
            if record["pixel_sha256"] == other["pixel_sha256"] or (
                near_distance >= 0 and (record["dhash"] ^ other["dhash"]).bit_count() <= near_distance
            ):
                parent[find(i)] = find(j)

    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    priority = {"train": 0, "validation": 1, "test": 2}
    conflicts, removed = [], []
    retained = []
    for indices in groups.values():
        group_id = min(records[i]["id_code"] for i in indices)
        # Do not learn from duplicate images with inconsistent labels.
        if len({records[i]["label"] for i in indices}) > 1:
            conflicts.extend(records[i]["id_code"] for i in indices)
            for i in indices:
                removed.append(dict(records[i], reason="conflicting_duplicate_labels", group_id=group_id))
            continue
        winner = max(indices, key=lambda i: (priority[records[i]["split"]], records[i]["id_code"]))
        retained.append(dict(records[winner], group_id=group_id))
        for i in indices:
            if i != winner:
                removed.append(dict(records[i], reason="duplicate_group", group_id=group_id))
    manifest = pd.DataFrame(retained).sort_values(["split", "id_code"]).reset_index(drop=True)
    for split in SPLITS:
        labels = set(manifest.loc[manifest.split == split, "label"])
        if labels != set(range(5)):
            raise ValueError(f"All five classes must remain in the {split} split; got {labels}")
    audit = {"source_images": len(records), "retained_images": len(retained),
             "removed_images": len(removed), "conflicting_label_images": len(conflicts),
             "near_duplicate_hamming_threshold": near_distance,
             "patient_level_isolation": False,
             "counts": {s: {str(k): int(v) for k, v in manifest[manifest.split == s].label.value_counts().sort_index().items()} for s in SPLITS},
             "removed": removed}
    return manifest, audit


def load_pixels(frame, size):
    """Exactly match production preprocessing, including the legacy [-1,1] API."""
    pixels = []
    for path in frame.path:
        image = cv2.imread(str(path))
        image = prepare_retina(image, size)
        pixels.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return np.asarray(pixels, dtype=np.uint8)


def input_tensor(pixels):
    return pixels.astype(np.float32) / 127.5 - 1.0
