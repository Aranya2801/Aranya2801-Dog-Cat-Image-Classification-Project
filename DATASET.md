# 📦 Dataset Information

## Microsoft Cats vs Dogs (ASIRRA)

The primary dataset used to train DogCat Vision.

| Property | Value |
|----------|-------|
| **Name** | Microsoft Cats vs Dogs |
| **Original Source** | Microsoft Research (ASIRRA CAPTCHA dataset) |
| **Kaggle Link** | https://www.kaggle.com/c/dogs-vs-cats/data |
| **HuggingFace** | https://huggingface.co/datasets/microsoft/cats_vs_dogs |
| **Total Images** | **25,000** (12,500 cats + 12,500 dogs) |
| **Image Format** | JPEG (RGB) |
| **Resolution** | Variable — resized to 380×380 for training |
| **License** | Kaggle Terms of Service |
| **Year** | 2013 (updated 2020 on HuggingFace) |

---

## How to Download

### Method 1: Kaggle CLI (Recommended)

```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Create API credentials
#    Go to: https://www.kaggle.com/settings > "API" > "Create New Token"
#    Place kaggle.json at: ~/.kaggle/kaggle.json (Linux/Mac) or %USERPROFILE%\.kaggle\kaggle.json (Windows)

# 3. Download and prepare
python scripts/download_dataset.py --method kaggle
```

### Method 2: HuggingFace (No Account Needed)

```bash
python scripts/download_dataset.py --method huggingface
```

### Method 3: Direct Kaggle Download

1. Visit: https://www.kaggle.com/datasets/salader/dogs-vs-cats
2. Sign in to Kaggle (free account)
3. Click "Download" (861 MB ZIP)
4. Extract to `data/raw/`
5. Run: `python scripts/prepare_data.py`

---

## Dataset Split (After Preparation)

| Split | Cats | Dogs | Total | Percentage |
|-------|------|------|-------|------------|
| **Train** | 10,000 | 10,000 | **20,000** | 80% |
| **Val** | 1,250 | 1,250 | **2,500** | 10% |
| **Test** | 1,250 | 1,250 | **2,500** | 10% |
| **Total** | 12,500 | 12,500 | **25,000** | 100% |

---

## Sample Images

The `data/sample/` directory contains 20 curated sample images (10 cats, 10 dogs) for quick testing
without requiring the full dataset download.

---

## Data Preprocessing Pipeline

```
Raw Image (variable size, JPEG)
    │
    ▼
Resize to 380×380 (bilinear interpolation)
    │
    ▼
[Training only] Albumentations Augmentations:
    ├─ RandomResizedCrop (scale 0.8–1.0)
    ├─ HorizontalFlip (p=0.5)
    ├─ ColorJitter (brightness, contrast, saturation)
    ├─ GaussianBlur (p=0.2)
    ├─ CoarseDropout / Cutout (p=0.3)
    └─ 15 more transforms...
    │
    ▼
Normalize (ImageNet stats)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    │
    ▼
Convert to Tensor → Model Input (B, 3, 380, 380)
```

---

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{microsoft_cats_vs_dogs,
  title  = {Microsoft Cats vs Dogs Dataset},
  author = {Microsoft Research},
  year   = {2013},
  url    = {https://www.microsoft.com/en-us/download/details.aspx?id=54765}
}
```
