# low-quality-image-detection
Low Quality Image Detection with Machine Learningc

## Environment setup

Create a Python environment with `Python 3.10.13` and the packages listed in `requirements.txt`. Check [here](https://docs.python.org/3/library/venv.html) or, if you are using Anaconda, [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment) for the detailed steps. 

## Dataset

The images are available [here](https://www.kaggle.com/datasets/kwentar/blur-dataset) and [here](https://drive.google.com/file/d/1RObmCDPeQ1Lg-V6u7dT02Pf0qH-QMcTp/view). Once downloaded the .zip folder, extract its contents ( defocused_blurred, motion_blurred, sharp) inside a folder called `dataset_dms` (unless you have chosen another name). 

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


