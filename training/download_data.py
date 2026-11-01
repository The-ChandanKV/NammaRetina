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
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    archive = args.destination / "aptos224.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        digest = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(URL, timeout=60) as source, temporary.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
                    digest.update(block)
                    size += len(block)
                    if size % (25 * 1024 * 1024) == 0:
                        print(f"Downloaded {size / 1024**2:.0f} MiB", flush=True)
            if digest.hexdigest() != SHA256:
                raise ValueError("Dataset checksum changed; inspect the source before using this experiment")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    elif hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Existing dataset archive has the wrong checksum")
    destination = args.destination / "aptos224"
    extract_checked(archive, destination)
    count = len(list(destination.rglob("*.png")))
    if count != 3662:
        raise ValueError(f"Expected 3,662 images, found {count}")
    metadata = {"source": SOURCE, "version": 4, "download_url": URL,
                "upstream": "https://www.kaggle.com/competitions/aptos2019-blindness-detection",
                "archive_sha256": SHA256, "image_count": count,
                "preprocessing": "Resized color images; shared Ben Graham preprocessing is applied by the trainer."}
    (args.destination / "source.json").write_text(json.dumps(metadata, indent=2))
    print(f"Verified {count} images at {destination}", flush=True)


if __name__ == "__main__":
    main()
