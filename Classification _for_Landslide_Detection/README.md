# Integrated Landslide Detection Pipeline — Step-by-Step README

## Overview
This notebook implements a full pipeline for the **Classification for Landslide Detection** challenge on Zindi. It covers:
- data loading and exploratory data analysis (EDA),
- composite RGB image creation from multi-band `.npy` tiles,
- data augmentation and DataLoader creation using FastAI + Albumentations,
- model training with stratified cross-validation (TTA + OOF),
- evaluation and submission generation.

The pipeline was written to run in Kaggle/Colab environments but can be adapted to local machines. Default paths assume a Kaggle input layout.

---

## Quick links (paths used in the notebook)
- `PipelineConfig.BASE_DIR` — base dataset location (default: `/kaggle/input/slideandseekclasificationlandslidedetectiondataset`)
- `PipelineConfig.OUTPUT_DIR` — where composite PNGs are written (default: `/kaggle/working/processed_images`)
- Final submission saved to: `/kaggle/working/Baseline_{MODEL_NAME}_submission.csv`

---

## Requirements / Environment
Install the packages used by the notebook. Using a conda environment or pip is fine.

```bash
pip install numpy pandas matplotlib seaborn tqdm pillow scikit-learn albumentations fastai timm torch torchvision
pip install git+https://github.com/fastai/fastai --upgrade   # if you need latest fastai
```

> On Kaggle, most packages (numpy, pandas, fastai, torch) are preinstalled. Adjust the `BASE_DIR` if your dataset path differs.

---

## Files expected (from competition)
- `Train.csv` — contains columns `ID`, `label` (0/1), etc.
- `Test.csv` — contains `ID` to predict
- `train_data/*.npy` — per-sample multi-band NumPy arrays
- `test_data/*.npy` — same as above for test

Make sure `PipelineConfig.BAND_LABELS` corresponds to the band order inside the `.npy` files.

---

## Configuration (edit before running)
Open the `PipelineConfig` class at the top of the notebook and change values as needed:
- `BASE_DIR`: path to the competition dataset
- `OUTPUT_DIR`: path to save generated PNGs (must be writeable)
- `COMPOSITE_SCHEMES`: list of band combinations used to build RGB images
- `BAND_LABELS`: names and order of all bands found in the `.npy` files
- `IMAGE_SIZE`, `BATCH_SIZE`, `EPOCHS`, `MODEL_NAME`, `N_SPLITS`, `TTA_ROUNDS` etc.

Example:
```python
PipelineConfig.BASE_DIR = '/kaggle/input/slideandseekclasificationlandslidedetectiondataset'
PipelineConfig.OUTPUT_DIR = '/kaggle/working/processed_images'
PipelineConfig.MODEL_NAME = 'beitv2_large_patch16_224.in1k_ft_in22k_in1k'
```

---

## Step-by-step run instructions

### Step 0 — (Optional) Use GPU, set runtime
- On Kaggle: GPU is selected by default when using GPU-enabled kernel.
- On Colab: Runtime → Change runtime type → GPU.
- Ensure PyTorch detects GPU: `torch.cuda.is_available()`

### Step 1 — Set reproducible seed
The notebook calls `set_seed(PipelineConfig.RANDOM_STATE)` to ensure reproducibility (affects NumPy, random, and PyTorch seeds). Keep the seed fixed or change for experiments.

### Step 2 — Load metadata CSVs
The notebook reads `Train.csv` and `Test.csv` from `PipelineConfig.BASE_DIR`:
```python
train_df = pd.read_csv(PipelineConfig.TRAIN_CSV)
test_df  = pd.read_csv(PipelineConfig.TEST_CSV)
```
It adds an `npy_path` column pointing to each sample’s `.npy` file.

### Step 3 — Exploratory Data Analysis (EDA)
- A class distribution plot is shown using `sns.countplot`.
- One sample is loaded and each raw band (per `PipelineConfig.BAND_LABELS`) is plotted for inspection. Check for NaNs, dynamic ranges, and band alignment.

### Step 4 — Create composite images
The `build_and_store_composites()` function does the following for each sample:
1. Loads the `.npy` tile (expected shape H x W x C)
2. Selects the band indices for a given composite scheme (e.g., `['nir','red','green']`)
3. Normalizes each composite to 0–255 and saves as a PNG under `OUTPUT_DIR/{train|test}/{scheme}/{ID}.png`

Run the composites generation (already in the notebook):
```python
build_and_store_composites(train_df, PipelineConfig.COMPOSITE_SCHEMES, dirs, 'train')
build_and_store_composites(test_df, PipelineConfig.COMPOSITE_SCHEMES, dirs, 'test')
```

**Notes & tips:**
- Normalization is per-tile (min/max). If you prefer per-band global scaling, replace the normalization step.
- For large datasets, generating PNGs can take time and disk space. Consider generating only for the chosen scheme to save time/disk: set `COMPOSITE_SCHEMES = [['nir','red','green']]`.

### Step 5 — Prepare cross-validation folds
Stratified K-Fold is used to assign folds into `train_df['fold']`. Default `N_SPLITS=5`. Change if required.

### Step 6 — Data augmentation + FastAI DataLoaders
- Albumentations transforms are defined for train and validation (resize + flips).
- A custom `AlbAug` wrapper (a `fastai` `RandTransform`) converts Albumentations transforms to FastAI-compatible transforms.
- `DataBlock` is created using `ImageBlock` and `CategoryBlock` with `get_x=ColReader('img_path')` and `get_y=ColReader('label')`.

To obtain DataLoaders for a fold, call:
```python
dls = get_dls(train_df, fold)
```

**Note:** `batch_tfms = [Normalize.from_stats(*imagenet_stats)]` uses ImageNet normalization; this assumes your composites are somewhat compatible with ImageNet-style pretraining. For different pretrained weights, adjust as needed.

### Step 7 — Model training with Stratified CV, TTA, OOF
For each fold:
1. Build DataLoaders for the fold.
2. Create a `vision_learner` using `PipelineConfig.MODEL_NAME` and metrics `[accuracy, F1Score(average='binary')]`.
3. Use `lr_find` to suggest a learning rate, then `fine_tune` for `PipelineConfig.EPOCHS` epochs.
4. Generate OOF predictions using `learner.tta(n=PipelineConfig.TTA_ROUNDS)` on the validation set.
5. Generate test predictions (TTA) and append them to `all_fold_outputs`.
6. Save best model using `SaveModelCallback` monitoring `f1_score`.

Snippet from notebook:
```python
learner = vision_learner(...)
lr_suggest = learner.lr_find(suggest_funcs=(valley, ))[0]
learner.fine_tune(PipelineConfig.EPOCHS, lr_suggest)
preds, _ = learner.tta(dl=dls.valid, n=PipelineConfig.TTA_ROUNDS)
```
**Tips:**
- Fine-tuning heavy models (e.g., BEiTv2 large) may require a lot of GPU memory. Reduce `BATCH_SIZE` or switch to a smaller model if you run out of memory.
- If `vision_learner` does not support the `MODEL_NAME` via fastai/timm, use `timm.create_model` and wrap it manually or choose a supported backbone.

### Step 8 — Out-of-fold evaluation
- The notebook constructs `oof_preds` and `pred_label` using a threshold of 0.5 on the second class probability (`x[1]`).
- Computes overall F1 using `sklearn.metrics.f1_score` and displays a confusion matrix via `ConfusionMatrixDisplay`.

### Step 9 — Submission generation
- The test fold predictions (collected across folds) are stacked and averaged:
```python
stacked = np.stack(all_fold_outputs, axis=0).mean(axis=0)
test_df['label'] = np.argmax(stacked, axis=1)
submission = test_df[['ID','label']].rename(columns={'label':'target'})
submission.to_csv(output_csv, index=False)
```
- `output_csv` is saved under `/kaggle/working/` with the model name included.

---

## Troubleshooting & Common Issues

### 1. Missing `.npy` files or wrong `BASE_DIR`
Double-check `TRAIN_CSV`/`TEST_CSV` and `RAW_TRAIN_DATA`/`RAW_TEST_DATA` paths. Print sample paths to debug:
```python
print(train_df['npy_path'].head())
```
### 2. Memory / CUDA OOM
- Reduce `BATCH_SIZE` or `IMAGE_SIZE`.
- Use a smaller backbone (e.g., `resnet34` or `efficientnet_b0`) while prototyping.
- Free memory between folds: `del learner, dls; gc.collect(); torch.cuda.empty_cache()` (already included in the notebook).

### 3. Model not found in `vision_learner`
- `PipelineConfig.MODEL_NAME` must be a model name available via `timm`. If `vision_learner` cannot load it, either install a compatible `timm` version, or pick a supported backbone name, e.g., `"resnet34"`.

### 4. Albumentations compatibility with FastAI
- The custom `AlbAug` wrapper is included in the notebook. If you change Albumentations transforms, ensure the wrapper still returns PIL images or fastai PILImage-compatible types.

### 5. Inconsistent normalization between train/test
- The notebook uses per-tile min-max normalization and ImageNet normalization before input to the model. If you observe distribution shift, consider building global statistics for normalization.

---

## Output / Artifacts produced by the notebook
- `OUTPUT_DIR` — composite PNGs for train/test by scheme (e.g. `processed_images/train/nir_red_green/ID.png`)
- Saved models (if `SaveModelCallback` triggers) in the model folder (`/kaggle/working/`)
- OOF probabilities and `pred_label` added to `train_df` (in-memory)
- Confusion matrix plot
- Final submission CSV: `/kaggle/working/Baseline_{MODEL_NAME}_submission.csv`

---

## Reproducibility & Notes
- The notebook sets seeds for python, numpy and torch. However, certain operations may still be nondeterministic (some CUDA ops). For strict determinism, ensure `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` (done in `set_seed`).

- If you plan to run multiple experiments, change `PipelineConfig.MODEL_NAME` and `OUTPUT_DIR` to avoid overwriting artifacts.

---

## Where to go next / Improvements
- Try additional augmentations (brightness/contrast, gaussian noise) for robustness.
- Replace per-tile normalization with per-band statistics or use percentile clipping to reduce outliers.
- Try ensembling diverse model architectures and alternative stacking approaches for better leaderboard performance.
- Use learning rate scheduling and longer fine-tuning if resources allow.

---

## License
This project is provided under the MIT License. Feel free to reuse and adapt for the Zindi challenge.

---

If you'd like, I can also:
- write this README to a file in the notebook workspace (`/mnt/data/README_step_by_step.md`), or
- extract the exact notebook section headings and include them as a structured table of contents.
