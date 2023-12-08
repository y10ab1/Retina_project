# Retina project

## Quick start

### 1. Prepare conda environment
```bash
conda create -n retina python=3.9
conda activate retina
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare data
Download data from kaggle: https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification/data
Put it in `dataset` folder.

### 4. Run training
```bash
python main.py
```

### 5. Run tensorboard to see results
```bash
tensorboard --logdir=src/logs
```

