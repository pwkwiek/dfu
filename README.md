# Diabetic Foot Ulcer Segmentation using U-Net with LAB + ERI

*More Than RGB—Because Wounds Have Layers.*

---

This project implements a **deep learning-based semantic segmentation model** for automatic detection of diabetic foot ulcers from clinical foot images. A custom **U-Net** architecture is trained to generate **pixel-level segmentation masks**, accurately outlining ulcer regions instead of simply classifying whether an ulcer is present.

To improve robustness under varying lighting conditions and skin tones, the model uses a **4-channel input** consisting of the **LAB color space** and the **Erythema Redness Index (ERI)**, which enhances redness associated with inflamed tissue. 

## Features

* Pixel-level diabetic foot ulcer segmentation
* Custom U-Net architecture
* LAB + ERI preprocessing for improved wound visualization
* Data augmentation for better generalization
* Dice Loss + Binary Cross Entropy + Gradient Consistency Loss
* Evaluation using Dice Score, IoU, and Pixel Accuracy
* Generates binary masks, probability maps, and overlay visualizations of detected ulcers.  

## Detection Pipeline

1. Clinical foot images are converted from RGB to **LAB color space**.
2. An additional **ERI (Erythema Redness Index)** channel is computed to emphasize inflamed tissue.
3. The resulting 4-channel image is normalized and passed into the U-Net model.
4. The network predicts a probability map for ulcer pixels.
5. A threshold is applied to create the final binary segmentation mask.
6. The predicted mask can be overlaid on the original image for visualization.  

## Project Structure

```text
.
├── main.ipynb                 # Complete notebook for training, evaluation, and testing
├── dfu_lab_eri_unet.py        # Model architecture, preprocessing, dataset, training, and evaluation
├── predict_unet_lab_eri.py    # Loads trained model and predicts masks on new images
├── unet_lab_eri_best.pt       # Saved trained model (generated after training)
└── README.md
```

### File Descriptions

* **main.ipynb**
  Demonstrates the complete workflow, including dataset preparation, model training, evaluation, visualization of results, and inference examples. 

* **dfu_lab_eri_unet.py**
  Contains the complete implementation of the segmentation pipeline, including:

  * LAB + ERI preprocessing
  * Dataset loading
  * Data augmentation
  * Custom U-Net architecture
  * Loss functions
  * Training and validation loops
  * Performance metrics
  * Model checkpoint saving. 

* **predict_unet_lab_eri.py**
  Loads a trained checkpoint and performs inference on unseen clinical foot images. The script generates:

  * Binary ulcer masks
  * Probability maps (optional)
  * Red overlay images showing detected ulcer regions on the original photograph. 

## Model Output

The model produces:

* Binary segmentation masks
* Ulcer probability maps
* Overlay images highlighting detected ulcer regions

These outputs assist clinicians in visualizing wound boundaries and estimating ulcer size for further analysis. 

## Evaluation Metrics

The model is evaluated using:

* Dice Score
* Intersection over Union (IoU)
* Pixel Accuracy
* Validation Loss 

## Workflow

```text
Clinical Foot Image
        │
        ▼
LAB + ERI Preprocessing
        │
        ▼
Custom U-Net
        │
        ▼
Probability Map
        │
        ▼
Binary Segmentation Mask
        │
        ▼
Overlay on Original Image
```

This project demonstrates how semantic segmentation can accurately localize diabetic foot ulcers, providing precise wound boundaries that can support clinical assessment and monitoring.
