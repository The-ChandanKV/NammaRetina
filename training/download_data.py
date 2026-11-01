"""Download the versioned public APTOS 224px derivative used for this experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

URL = "https://www.kaggle.com/api/v1/datasets/download/sovitrath/diabetic-retinopathy-224x224-2019-data?datasetVersionNumber=4"
SHA256 = "8a6c20b4b645efa026e1695511e5eadf8a8bf0dd5bbf5d906f35b53919020ce3"
SOURCE = "https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data"


def archive_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def extract_checked(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if not (destination / info.filename).resolve().is_relative_to(destination):
                raise ValueError("Unsafe dataset archive entry")
        bundle.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("training/data"))
    parser.add_argument('--dataset', choices=['aptos224', 'eyepacs224'], default='aptos224')
    args = parser.parse_args()
    eyepacs = args.dataset == 'eyepacs224'
    url = 'https://www.kaggle.com/api/v1/datasets/download/sovitrath/diabetic-retinopathy-2015-data-colored-resized' if eyepacs else URL
    checksum = '2e81874412fcf34de4e9ae996f9573e6429bb920bc177fca853fe3ef3ca6f417' if eyepacs else SHA256
    expected_count = 35126 if eyepacs else 3662
    args.destination.mkdir(parents=True, exist_ok=True)
    archive = args.destination / (args.dataset + ".zip")
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        digest = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
                    digest.update(block)
                    size += len(block)
                    if size % (25 * 1024 * 1024) == 0:
                        print(f"Downloaded {size / 1024**2:.0f} MiB", flush=True)
            if digest.hexdigest() != checksum:
                raise ValueError("Dataset checksum changed; inspect the source before using this experiment")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    elif archive_digest(archive) != checksum:
        raise ValueError("Existing dataset archive has the wrong checksum")
    destination = args.destination / args.dataset
    extract_checked(archive, destination)
    count = len(list(destination.rglob("*.png")))
    if count != expected_count:
        raise ValueError(f"Expected {expected_count} images, found {count}")
    metadata = {"source": "https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized" if eyepacs else SOURCE, "version": None if eyepacs else 4, "download_url": url,
                "upstream": "https://www.kaggle.com/competitions/diabetic-retinopathy-detection" if eyepacs else "https://www.kaggle.com/competitions/aptos2019-blindness-detection",
                "archive_sha256": checksum, "image_count": count,
                "preprocessing": "Resized color images; shared Ben Graham preprocessing is applied by the trainer."}
    (args.destination / ("eyepacs_source.json" if eyepacs else "source.json")).write_text(json.dumps(metadata, indent=2))
    print(f"Verified {count} images at {destination}", flush=True)


if __name__ == "__main__":
    main()
