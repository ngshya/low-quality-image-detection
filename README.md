# Low Quality Image Detection

This repository contains some experimental Python code designed for the detection of low-quality images through a machine learning approach. 
In particular, the algorithm will perform 
- sharp (good quality) images detection, 
- defocused blur detection, 
- motion blur detection, 
- horizontal/vertical bands detection, 
- noise detection, 
- excessive exposure detection, 
- glare detection, 
- dark photo detection, 
- uninformative constant colour detection. 

## Environment setup

Create and use a Python environment with `Python 3.10.13` and the packages listed in `requirements.txt`. Check [here](https://docs.python.org/3/library/venv.html) or, if you are using Anaconda, [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment) for the detailed steps. 

## Dataset

The images are available [here](https://www.kaggle.com/datasets/kwentar/blur-dataset) and [here](https://drive.google.com/file/d/1RObmCDPeQ1Lg-V6u7dT02Pf0qH-QMcTp/view). Once downloaded the .zip folder, extract its contents ( defocused_blurred, motion_blurred, sharp) inside a folder called `dataset_dms` (unless another location is chosen). 

## Synthetic data generation

```bash
python 01_synthetic_low_quality_images.py
```

A folder called `dataset_synthetic` (unless another location is chosen) will be created with generated low quality images.

## Features generation

```bash
python 02_features_generation.py
```

A CSV file called `df_public.csv` (unless another name is used) will be generated. 

## Models training

Look at `03_models_{name}.ipynb` notebooks as examples. 

## Just give me the script!

```bash
python detect.py {image path}
```