# Diabetic Retinopathy Detection

This repository implements a deep learning approach for diabetic retinopathy (DR) detection using DenseNet121 and ResNet18, evaluated on the ODIR-5K dataset.

## Files

- **`train.py`**: Main script for training, validation, and testing.  
- **`loader.py`**: Custom data loader for preprocessing the ODIR dataset.  
- **`baseline_models.py`**: Defines DenseNet121 and ResNet18 architectures.  

<!-- ## Requirements

Install dependencies with:
```bash
pip install -r requirements.txt -->

# Usage
* Prepare .npz files for the dataset.
* Train the models by running **`python train.py`**