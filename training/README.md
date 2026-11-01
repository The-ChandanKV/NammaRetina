# Training a NammaRetina candidate

The new trainer fixes EfficientNet's input normalization, audits exact duplicate
images before training, and selects checkpoints using validation accuracy. It
exports a model compatible with the current app and checks inference and real
Grad-CAM after serialization. A higher score is a measured outcome, not a promise.

## Completed CPU experiment

The frozen-backbone candidate trained on 2,789 images and was selected on 361
validation images. It was then evaluated on 354 held-out images:

| Metric | Measured result |
|---|---|
| Test accuracy | 82.77% (293/354) |
| Image-level bootstrap 95% accuracy interval | 78.81–86.44% |
| Macro-F1 | 0.618 |
| Quadratic-weighted kappa | 0.840 |
| Balanced accuracy | 60.58% |
| Severe / proliferative class recall | 30.77% / 36.67% |
| Save/reload maximum probability difference | 0.0 |

The low rare-class recalls matter: overall accuracy is dominated by easier and
more common classes. This is a research baseline for further fine-tuning and
external evaluation, not a clinically validated release. The matching legacy
normalization recipe obtained 47.65% validation accuracy; the corrected recipe
obtained 77.56% on the same validation partition. This comparison includes fixing
pretrained normalization-state finalization, and does not recreate the missing
historical trained artifact.

The candidate is locally available at
`training/outputs/cpu_candidate/efficientnetb0_dr.keras` (about 18.08 MiB). Its SHA256 is
`839b2f9c88a60b2273d5043f9a5eb53634c03d86da464a7aa589e4d54097803b`.
Committed metrics and experiment metadata are in [results/](results/); image data
and weights remain local and ignored. Load, prediction, and genuine Grad-CAM
smoke checks passed with TensorFlow 2.16.2 / Keras 3.3.3.

## Dataset

Source: [APTOS 224px color derivative, version 4](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data),
originally from [APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data).
The download is approximately 250 MB and contains 3,662 labeled images. It is a
resized copy of APTOS, not additional independent training data. Follow the
provider's applicable dataset terms for your intended use.

```bash
python -m training.download_data
```

The downloader pins the version and verifies SHA256. Training starts from the
committed `APTOS-19` split IDs. Identical decoded images cannot cross splits:
test takes precedence over validation, and validation takes precedence over
training. Groups with conflicting labels are quarantined entirely. This changes
the old split membership and must be disclosed with results. The current audit
retains 2,789 training, 361 validation, and 354 test images. Perceptual hashes are
not used for automatic filtering because distinct retinas can look similar.
No patient IDs are available, so patient-level isolation and elimination of all
near-duplicates are not established.

## Recommended full training: a GPU notebook

Open [Train_NammaRetina_GPU.ipynb](Train_NammaRetina_GPU.ipynb) in Google Colab,
select a T4 GPU runtime, and run the cells in order. The notebook clones this
repository, installs the training requirements, downloads the dataset, trains,
evaluates, and copies the artifacts to your own Google Drive. Use a Python 3.11
or 3.12 runtime for the pinned TensorFlow release. If Colab defaults to a newer
Python, choose an available past runtime in **Runtime → Change runtime type**
([official instructions](https://research.google.com/colaboratory/runtime-version-faq.html)). If dependency installation
requires restarting the runtime, restart and run from the first cell again.

The same command works in a GPU environment without a notebook:

```bash
python -m training.train \
  --data-root training/data/aptos224 \
  --run-dir training/outputs/gpu_candidate \
  --mode finetune --require-gpu \
  --head-epochs 40 --fine-epochs 20 --batch-size 32
```

The trainer first learns a classification head from cached ImageNet features.
It then fine-tunes the final convolutional layers with batch normalization
frozen. Conservative augmentation is applied in pixel space, before the app's
`[-1,1]` input transform. Square-root class weights temper class imbalance.
The fine-tuning checkpoint cannot overwrite a better frozen-head checkpoint.
Use `--no-test` while developing recipes. Do not choose hyperparameters using
the held-out test results.

## CPU baseline and local Jupyter

On this Intel Mac, a separate Python 3.11 environment has been installed at
`.venv-training/`; the app environment is unchanged. TensorFlow 2.16.2 is used
because official newer wheels do not support macOS x86.

```bash
.venv-training/bin/python -m training.train \
  --data-root training/data/aptos224 \
  --run-dir training/outputs/cpu_candidate --mode frozen

.venv-training/bin/jupyter lab
```

On another machine, create a Python 3.11 environment and install
`training/requirements.txt` first. A CPU baseline trains the classifier but does
not fine-tune the ImageNet backbone. Actual training speed depends on hardware.

## Why normalization changed inside the model

[EfficientNet already includes input preprocessing](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0)
and expects `[0,255]` pixels. The old notebook fed `[-1,1]` directly into it. The
new exported model contains a built-in `Rescaling(127.5, offset=127.5)` adapter
before EfficientNet, restoring its expected input range. The current app still
supplies shared Ben Graham image preparation and `[-1,1]` inputs, so an old and
a new model do not require incompatible changes to production preprocessing.
No custom Python layer is required to load the exported artifact.

## Read the result before promotion

The output directory contains:

| File | Purpose |
|---|---|
| `efficientnetb0_dr.keras` | Selected, production-compatible model |
| `metrics.json` | Validation/test accuracy, balanced accuracy, macro-F1, QWK, all five recalls, confusion matrix, referable-DR sensitivity/specificity, severe undergrading, and image-level bootstrap accuracy interval |
| `model_metadata.json` | Input/class contract, framework versions, seed, dataset and manifest fingerprints |
| `model.sha256` | Integrity digest for Render |
| `manifest.csv`, `data_audit.json` | Exact image membership and exclusions |
| `test_predictions.csv` | Auditable per-image held-out scores |
| `validation_history.jsonl` | Selection history without test-set tuning |

The old notebook records approximately 75.41% accuracy, but its trained artifact
is absent. That score is not a controlled baseline against the new model: split
membership changes after duplicate quarantine. Do not advertise the numerical
difference as a validated improvement. An APTOS test score alone also does not
establish clinical accuracy or generalization to other cameras or populations.

## Render production steps

1. Keep the selected `.keras` file and its metrics together. Do not commit image
   data or trained weights to Git. Compare validation results and every class's
   recall; use an unchanged independent evaluation set for later comparisons.
2. Upload the selected artifact to a stable HTTPS file host, or to Google Drive
   with file-download access. A Google Drive file link is supported by the app.
   Preserve the previous model URL/checksum to permit rollback.
3. In the existing Render service's environment set `MODEL_URL` to that download
   URL and `MODEL_SHA256` to the 64 hexadecimal characters in `model.sha256`.
4. Deploy the code containing checksum support and restart/redeploy the service.
   Do not attempt to train inside the Render web process. Training is separate.
5. Open `https://nammaretina.onrender.com/health`. Check `model_available: true`
   and that `model_sha256` exactly matches the exported digest. The endpoint
   returns 503 if either the database or model is unavailable.
6. Check `database_available: true` as well. If it is false, restore the Supabase
   project and valid `SUPABASE_URL`, `SUPABASE_KEY`, and `SUPABASE_SERVICE_KEY` as
   needed. Database connectivity is independent of model loading.
7. Sign in with an approved account, analyze a known validation image, open the
   XAI page, and download its PDF. Confirm that the model runs and real Grad-CAM
   is available. This smoke test confirms integration, not clinical validation.

If a better GPU candidate is trained later, repeat promotion with its own URL
and checksum. Full GPU training, external validation, and release review are
separate from the CPU baseline.
