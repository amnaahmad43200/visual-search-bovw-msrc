# Setup guide (EEE3032 Visual Search)

This project compares image retrieval methods on the **MSRC-v2** dataset. It starts with a global RGB histogram baseline, develops SIFT and Bag-of-Visual-Words (BoVW) methods, and then explores optional deep features and fusion.

The MSRC-v2 image dataset is **not included** in the coursework ZIP. 
## 1. Project folder structure

 The notebook names and support files shown here are included in the supplied project ZIP. Cache, vocabulary, feature and result folders are generated or populated as the notebooks run; some may already contain files in the supplied ZIP.

```text
MyDrive/
└── cvpr_coursework/
    ├── EEE3032_Coursework_Starter_v3/
    │   ├── 01_baseline_and_detected_sift.ipynb
    │   ├── 02_dense_sift.ipynb
    │   ├── 03_descriptor_variants.ipynb
    │   ├── 04_visual_vocabulary.ipynb
    │   ├── 05_bovw_histograms.ipynb
    │   ├── 06_Development_Evaluation.ipynb
    │   ├── 07_deep_descriptor_extraction.ipynb
    │   ├── 08_deep_bovw_late_fusion.ipynb
    │   ├── 09_query_adaptive_rank_fusion.ipynb
    │   ├── 10_final_test_evaluation.ipynb
    │   │
    │   ├── EEE3032_Coursework_Starter.ipynb       # supplementary
    │   ├── EEE3032_Dense_SIFT_Diagnostic.ipynb    # supplementary
    │   ├── EEE3032_Optional_Deep_Descriptors.ipynb # supplementary
    │   │
    │   ├── cvpr/
    │   │   ├── __init__.py
    │   │   ├── baseline.py
    │   │   ├── dataset.py
    │   │   ├── deep.py
    │   │   ├── evaluation.py
    │   │   ├── personal.py
    │   │   ├── plotting.py
    │   │   └── retrieval.py
    │   │
    │   ├── visual_search/
    │   │   ├── __init__.py
    │   │   ├── cache.py
    │   │   ├── encoding.py
    │   │   ├── features.py
    │   │   └── vocabulary.py
    │   │
    │   ├── sift_cache/              # generated/reused SIFT descriptors
    │   ├── bovw_vocabularies/       # trained visual vocabularies
    │   ├── bovw_features/           # encoded BoVW features and IDF data
    │   ├── deep_features/           # optional deep-feature caches
    │   ├── results/                 # CSV files, plots and retrieval examples
    │   │
    │   ├── split_manifest.csv
    │   ├── student_methods.py
    │   ├── run_baseline.py
    │   ├── precompute_deep_descriptors.py
    │   ├── requirements.txt
    │   ├── requirements-deep.txt
    │   ├── detected_vs_dense_sift_summary.csv
    │   ├── sift_keypoint_summary.csv
    │   ├── stage3_opponent_sift_summary.csv
    │   ├── stage4_vocabulary_summary.csv
    │   └── README.md
    │
    └── MSRC_ObjCategImageDatabase_v2/
        └── Images/
            ├── 1_*.bmp
            ├── 2_*.bmp
            ├── 3_*.bmp
            ├── ...
            └── 20_*.bmp
```



## 2. Set the paths

The numbered notebooks contain a setup cell with two paths. **Check and update both paths in each notebook you run (Stages 1–10).** Search the notebook for `PROJECT_ROOT` and `DATASET_SOURCE`.

For the Google Drive folder layout shown above, use:

```python
from pathlib import Path

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/cvpr_coursework/EEE3032_Coursework_Starter_v3"
)
DATASET_SOURCE = Path(
    "/content/drive/MyDrive/cvpr_coursework/MSRC_ObjCategImageDatabase_v2"
)
```

`PROJECT_ROOT` must point to the `EEE3032_Coursework_Starter_v3` folder containing `cvpr/`, `visual_search/` and `split_manifest.csv`.

`DATASET_SOURCE` can point to any one of the following:

- the extracted `MSRC_ObjCategImageDatabase_v2` folder;
- its `Images` folder;
- the original MSRC-v2 ZIP file.

If the dataset remains zipped, change only `DATASET_SOURCE`, for example:

```python
DATASET_SOURCE = Path(
    "/content/drive/MyDrive/cvpr_coursework/MSRC_ObjCategImageDatabase_v2.zip"
)
```

For a local computer replace the two paths with your locations. For example,:

```python
PROJECT_ROOT = Path(r"C:\Users\YourName\Documents\EEE3032_Coursework_Starter_v3")
DATASET_SOURCE = Path(r"C:\Users\YourName\Documents\MSRC_ObjCategImageDatabase_v2")
```

Do not change the manifest location. The notebooks load it from `PROJECT_ROOT / "split_manifest.csv"`.

## 3. Install the packages

Open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
python -m pip install "scikit-learn>=1.2"
```

The Stage 4 vocabulary notebook uses scikit-learn for MiniBatchKMeans. Install it even if it is not listed in the copy of `requirements.txt` you received.

For the optional deep-feature stages (7–9), also run:

```bash
python -m pip install -r requirements-deep.txt
```

In Google Colab, mount Google Drive and run the equivalent commands in a notebook cell, using the full path to the requirements files. The numbered notebooks attempt to mount Drive automatically when running in Colab.

## 4. Run the notebooks in order

Run the numbered notebooks from top to bottom:

| Stage | Notebook | What it does |
|---|---|---|
| 1 | `01_baseline_and_detected_sift.ipynb` | Runs the RGB baseline and extracts detected SIFT features. |
| 2 | `02_dense_sift.ipynb` | Extracts SIFT features on a regular grid. |
| 3 | `03_descriptor_variants.ipynb` | Creates RootSIFT and OpponentSIFT variants. |
| 4 | `04_visual_vocabulary.ipynb` | Trains visual vocabularies using gallery descriptors only. |
| 5 | `05_bovw_histograms.ipynb` | Creates BoVW histograms and applies gallery-fitted IDF weights. |
| 6 | `06_Development_Evaluation.ipynb` | Compares the baseline and BoVW methods on development queries. |
| 7 | `07_deep_descriptor_extraction.ipynb` | Extracts frozen MobileNetV3-Small features (optional). |
| 8 | `08_deep_bovw_late_fusion.ipynb` | Tests fixed-weight fusion of BoVW and deep features (optional). |
| 9 | `09_query_adaptive_rank_fusion.ipynb` | Tests normalised, rank-based and query-adaptive fusion (optional). |
| 10 | `10_final_test_evaluation.ipynb` | Evaluates the selected core BoVW method on the held-out test queries. |

Stages 1–6 and 10 form the core workflow. Stages 7–9 are the optional deep-descriptor extension. Run Stage 7 before Stage 8, and Stage 8 before Stage 9. Stage 10 should be run only after the core method and its settings have been selected using Stage 6 development results.

The three supplementary notebooks are not part of the numbered workflow.

## 5. Dataset split and evaluation

The supplied `split_manifest.csv` defines the fixed split:

| Split | Images | Purpose |
|---|---:|---|
| Gallery | 391 | Search database; the only split used to fit the visual vocabulary and IDF weights. |
| Development queries | 100 | Compare methods and select settings. |
| Final test queries | 100 | Evaluate the frozen core method. |

Do not fit the visual vocabulary, IDF weights or other data-dependent settings using development or test queries. Use development results to make decisions. Do not repeatedly inspect the test results or change the method after the final test evaluation.

The notebooks report Precision@10 (P@10), Average Precision (AP), mean Average Precision (mAP), group-wise results and retrieval examples. Stage 10 also creates a 20 × 20 confusion matrix based on the group of each query's top-ranked retrieved image.

## 6. Generated files

Generated files are saved under `PROJECT_ROOT`:

| Folder | Contents |
|---|---|
| `sift_cache/` | Cached detected, dense and opponent-colour SIFT descriptors. |
| `bovw_vocabularies/` | Trained visual vocabularies. |
| `bovw_features/` | BoVW histograms and IDF information. |
| `deep_features/` | Cached deep descriptors for the optional extension. |
| `results/` | Evaluation CSV files, heatmaps, confusion matrices and ranked retrieval figures. |

Keep these folders between runs. The notebooks reuse valid cached files where supported. If a required cache or result file is missing, rerun the earlier notebook that creates it before continuing.

## 7. Troubleshooting

- **Project folder not found:** check that `PROJECT_ROOT` points to the folder containing `cvpr/` and `visual_search/`.
- **Manifest not found:** check that `split_manifest.csv` is directly inside the project folder.
- **Images not found:** check that `DATASET_SOURCE` points to the extracted dataset folder, its `Images` folder, or the dataset ZIP.
- **Missing vocabulary or BoVW features:** run Stages 1–5 before Stage 6 or Stage 10.
- **Missing deep features:** install `requirements-deep.txt` and run Stage 7 before Stages 8–9.
- **Missing fusion files:** run Stage 8 before Stage 9.

If using a local computer, make sure the Jupyter kernel is running in the same Python environment where the packages were installed.
