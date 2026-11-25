"""Downloads the original full-size PKU PCB defect dataset (693 labelled photos) from Kaggle.

This is the dataset the final defect model was trained on (see training/train_pcb_yolo_tiles_colab.ipynb).
No account or key is needed.

    pip install kagglehub
    python3 download_dataset.py
"""
import kagglehub

path = kagglehub.dataset_download("akhatova/pcb-defects")
print("Dataset downloaded to:", path)
