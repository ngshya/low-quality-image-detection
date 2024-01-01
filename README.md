# low-quality-image-detection
Low Quality Image Detection with Machine Learning

## Dataset

The images are available [here](https://www.kaggle.com/datasets/kwentar/blur-dataset). Once downloaded the .zip folder, extract its contents ( defocused_blurred, motion_blurred, sharp) inside a folder called `dataset_dms` (unless you have chosen another name). 

## Synthetic data generation

```bash
python 01_synthetic_low_quality_images.py
```

A folder called `dataset_synthetic` (unless you have chosen another name) will be created with generated low quality images.

## Features generation

```bash
python 02_features_generation.py
```

A CSV file called `df_public.csv` (unless another name is used) will be generated. 